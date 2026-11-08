//! fork: the static pass's draw list. Every pass that draws retained regions — the colour pass and
//! its depth prepass, the water mirror, the shadow cascades, the local-light atlas, the GI
//! voxelizer — gathers its admitted runs into a [`GxBatch`], which sorts them by state, fuses runs
//! that continue each other's index range, and issues them.
//!
//! The sort is the saving on both targets: [`TrackedRenderPass`] drops a `set_*` that repeats the
//! bound state, so consecutive draws that share a pipeline, bind group and buffers cost one bind,
//! and the browser, which validates every call it is sent, gains the most. On native, where the
//! device draws a multi-draw in hardware (`MULTI_DRAW_INDIRECT_COUNT`: wgpu 29 has no separate
//! `MULTI_DRAW_INDIRECT` feature and emulates the call as a loop without it), the runs left with
//! one shared state go as one `multi_draw_indexed_indirect` over this frame's indirect buffer.
//!
//! A batch holds no pass, so recording passes on several threads only needs each thread to own
//! its batch: the scratch comes from a locked pool and the indirect range from an atomic cursor.
//!
//! What a texture binding array would add (not done): the bind group is per region today — it
//! carries the region's origin uniform and record table beside the pool class's texture array —
//! so no two regions ever share a state and a multi-draw stays inside one region. One bind group
//! for every pool class (`binding_array<texture_2d_array<f32>>`), the origins and records in one
//! storage buffer and the region picked by `first_instance` (`INDIRECT_FIRST_INSTANCE`) would let
//! a whole pipeline bucket go as one multi-draw across regions; it also needs the regions' indices
//! in one shared index buffer, which `assemble_region` copies per region now.

use super::{GxCellDraw, GxCellGpu, GxRun, GxSel};
use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{
    BindGroupId, Buffer, BufferDescriptor, BufferId, BufferUsages, DrawIndexedIndirectArgs,
    IndexFormat, RenderPipeline,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use crate::modern::caps::HwCaps;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};

/// Bytes of one indirect draw's arguments.
const ARGS_SIZE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;

/// The indirect buffer's floor, in draws: what a first frame grows it to at least.
const MIN_ARGS: u32 = 4096;

/// The pipeline keys `(cutout, two_sided)` in [`pipe_class`] order: a four-entry pipeline table
/// is built by mapping these.
pub(super) const PIPE_KEYS: [(bool, bool); 4] =
    [(false, false), (false, true), (true, false), (true, true)];

/// The four-entry pipeline table's index of `(cutout, two_sided)`, as a batch's pipeline class
/// (an `Option`, the shape every `push_*` class function takes).
pub(super) fn pipe_class(cutout: bool, two_sided: bool) -> Option<u8> {
    Some(u8::from(cutout) * 2 + u8::from(two_sided))
}

/// One draw of a batch: its state, then its index range.
#[derive(Clone, Copy)]
struct GxDraw {
    /// Draw-order phase: the colour pass's WMOs before its doodads, 1.12's drain order; 0 elsewhere.
    phase: u8,
    /// Index into the pipeline table the batch is emitted with.
    pipe: u8,
    vertex: BufferId,
    index: BufferId,
    bind: BindGroupId,
    base_vertex: i32,
    first: u32,
    count: u32,
    /// The region in [`GxBatch::regions`] and its bind-group slot, to find what the ids name.
    region: u32,
    slot: u32,
}

impl GxDraw {
    /// Everything a draw binds, in sort order. The vertex buffer (a mesh-allocator slab) comes
    /// before the index buffer and bind group, which are each one region's own, so a region's
    /// draws stay together and a slab binds once for all the regions in it.
    fn state(&self) -> (u8, u8, BufferId, BufferId, BindGroupId, i32) {
        (
            self.phase,
            self.pipe,
            self.vertex,
            self.index,
            self.bind,
            self.base_vertex,
        )
    }
}

/// A region a batch draws, with the vertex slice its mesh lives in.
#[derive(Clone, Copy)]
struct GxRegion<'w> {
    gpu: &'w GxCellGpu,
    vertex: &'w Buffer,
    base_vertex: i32,
}

/// A batch's reusable allocations.
#[derive(Default)]
struct GxScratch {
    draws: Vec<GxDraw>,
    args: Vec<DrawIndexedIndirectArgs>,
}

/// The batches' scratch, returned on drop, so a frame's gathers reuse last frame's allocations.
#[derive(Resource, Default)]
pub(super) struct GxDrawPool(Mutex<Vec<GxScratch>>);

/// This frame's indirect-argument buffer. Batches take disjoint ranges of it from `cursor` and
/// write them through the queue, which lands every write before the frame's command buffers run.
/// A batch whose range does not fit draws directly; the next frame's buffer is sized to the cursor.
#[derive(Resource)]
pub(super) struct GxIndirect {
    /// Native multi-draw is available and not switched off (`$WOW_STATIC_GX_MDI=0`).
    native: bool,
    buffer: Option<Buffer>,
    /// `buffer`'s size in draws.
    capacity: u32,
    /// Draws asked for this frame, granted or not.
    cursor: AtomicU32,
}

/// `RenderStartup`, after [`crate::modern::caps::init`]: multi-draw where the device draws one in
/// hardware ([`HwCaps::mdi_count`], never on wasm32) and the lever leaves it on.
pub(super) fn init_indirect(mut commands: Commands, caps: Option<Res<HwCaps>>) {
    commands.insert_resource(GxIndirect {
        native: caps.is_some_and(|c| c.mdi_count)
            && crate::launch_env::var("WOW_STATIC_GX_MDI").as_deref() != Some("0"),
        buffer: None,
        capacity: 0,
        cursor: AtomicU32::new(0),
    });
}

/// Rewind the cursor, and grow the buffer to what last frame asked for if that did not fit.
pub(super) fn prepare_indirect(mut indirect: ResMut<GxIndirect>, device: Res<RenderDevice>) {
    if !indirect.native {
        return;
    }
    let asked = std::mem::take(indirect.cursor.get_mut());
    if asked <= indirect.capacity {
        return;
    }
    let capacity = asked.next_power_of_two().max(MIN_ARGS);
    indirect.buffer = Some(device.create_buffer(&BufferDescriptor {
        label: Some("static_gx_indirect"),
        size: u64::from(capacity) * ARGS_SIZE,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    indirect.capacity = capacity;
}

/// One pass's draws: gathered with [`Self::region`] and the `push_*` calls, ordered by
/// [`Self::finish`], issued by [`Self::emit`] — as often as passes draw the same list, which is
/// how the depth prepass and the colour pass stay identical.
pub(super) struct GxBatch<'w> {
    pool: &'w GxDrawPool,
    indirect: &'w GxIndirect,
    queue: &'w RenderQueue,
    regions: Vec<GxRegion<'w>>,
    scratch: GxScratch,
    /// Runs gathered, before any fused: the meter's measure of what the passes asked for.
    gathered: u32,
    /// This batch's range of the indirect buffer, once [`Self::finish`] got one.
    args_at: Option<(&'w Buffer, u64)>,
}

impl<'w> GxBatch<'w> {
    pub(super) fn new(world: &'w World) -> Self {
        let pool = world.resource::<GxDrawPool>();
        let scratch = pool
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default();
        Self {
            pool,
            indirect: world.resource::<GxIndirect>(),
            queue: world.resource::<RenderQueue>(),
            regions: Vec::new(),
            scratch,
            gathered: 0,
            args_at: None,
        }
    }

    /// Empty the batch for another gather, keeping its allocations.
    pub(super) fn clear(&mut self) {
        self.regions.clear();
        self.scratch.draws.clear();
        self.scratch.args.clear();
        self.gathered = 0;
        self.args_at = None;
    }

    /// Whether any region was added since the last [`Self::clear`].
    pub(super) fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Add a region, drawn from its mesh's vertex slice; `None` while the mesh has none.
    pub(super) fn region(
        &mut self,
        gpu: &'w GxCellGpu,
        draw: &GxCellDraw,
        allocator: &'w MeshAllocator,
    ) -> Option<u32> {
        let vslice = allocator.mesh_vertex_slice(&draw.mesh.id())?;
        self.regions.push(GxRegion {
            gpu,
            vertex: vslice.buffer,
            base_vertex: i32::try_from(vslice.range.start).unwrap_or(0),
        });
        Some(self.regions.len() as u32 - 1)
    }

    /// Every run of region `r` merged across selection keys: the region drawn whole. `pipe` maps
    /// `(cutout, two_sided)` to the pipeline table's index; a run it has none for is not drawn.
    pub(super) fn push_whole(
        &mut self,
        r: u32,
        phase: u8,
        pipe: impl Fn(bool, bool) -> Option<u8>,
    ) {
        let gpu = self.regions[r as usize].gpu;
        for run in &gpu.whole {
            self.push(r, phase, run, &pipe);
        }
    }

    /// Region `r`'s runs whose selection key `sel` admits this frame; a run with no key (a cell
    /// item) always draws.
    pub(super) fn push_admitted(
        &mut self,
        r: u32,
        phase: u8,
        sel: &GxSel,
        pipe: impl Fn(bool, bool) -> Option<u8>,
    ) {
        let gpu = self.regions[r as usize].gpu;
        for run in &gpu.runs {
            if let Some(group) = run.group {
                if !sel.drawn.get(usize::from(group)).copied().unwrap_or(false) {
                    continue;
                }
            }
            self.push(r, phase, run, &pipe);
        }
    }

    fn push(&mut self, r: u32, phase: u8, run: &GxRun, pipe: &impl Fn(bool, bool) -> Option<u8>) {
        let Some(pipe) = pipe(run.cutout, run.two_sided) else {
            return;
        };
        let region = self.regions[r as usize];
        self.gathered += 1;
        self.scratch.draws.push(GxDraw {
            phase,
            pipe,
            vertex: region.vertex.id(),
            index: region.gpu.indices.id(),
            bind: region.gpu.bind_groups[run.slot].1.id(),
            base_vertex: region.base_vertex,
            first: run.index_range.start,
            count: run.index_range.len() as u32,
            region: r,
            slot: run.slot as u32,
        });
    }

    /// Order the gathered draws by state, fuse each draw into the one before it where its indices
    /// continue that one's under the same state, and on native write one indirect draw for every
    /// draw of a state that still has two or more. Opaque and cutout runs only, depth-tested: the
    /// order inside a phase changes no pixel but an exact depth tie.
    pub(super) fn finish(&mut self) {
        let indirect: &'w GxIndirect = self.indirect;
        let GxScratch { draws, args } = &mut self.scratch;
        sort_and_fuse(draws);
        self.args_at = None;
        args.clear();
        if !indirect.native {
            return;
        }
        multi_draw_args(draws, args);
        if args.is_empty() {
            return;
        }
        let n = args.len() as u32;
        let start = indirect.cursor.fetch_add(n, Relaxed);
        let fits = start
            .checked_add(n)
            .is_some_and(|end| end <= indirect.capacity);
        let Some(buffer) = indirect.buffer.as_ref().filter(|_| fits) else {
            return; // drawn directly this frame; `prepare_indirect` grows the buffer for the next
        };
        let offset = u64::from(start) * ARGS_SIZE;
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(args.as_slice()));
        self.args_at = Some((buffer, offset));
    }

    /// Issue the finished draws into `pass` through `pipes`, the pipeline table the classes
    /// index; a class with no pipeline is not drawn. Group 0 (the view) and any pass-wide group
    /// are the caller's to bind; this binds group 1, the vertex and the index buffer.
    /// `counter` is the [`crate::static_gx::GX_DRAWS`] slot the calls count in.
    pub(super) fn emit<'p>(
        &self,
        pass: &mut TrackedRenderPass<'p>,
        pipes: &[Option<&'p RenderPipeline>],
        counter: usize,
    ) where
        'w: 'p,
    {
        crate::static_gx::count_runs(counter, self.gathered);
        // The next group's first draw in this batch's indirect range: `finish` laid the ranges
        // out in this same group order, one per group of two or more.
        let mut arg = 0u64;
        for group in self.scratch.draws.chunk_by(|a, b| a.state() == b.state()) {
            let multi = self.args_at.filter(|_| group.len() > 1);
            let at = arg;
            if multi.is_some() {
                arg += group.len() as u64;
            }
            let head = group[0];
            let Some(pipeline) = pipes.get(usize::from(head.pipe)).copied().flatten() else {
                continue;
            };
            let region = self.regions[head.region as usize];
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(1, &region.gpu.bind_groups[head.slot as usize].1, &[]);
            pass.set_vertex_buffer(0, region.vertex.slice(..));
            pass.set_index_buffer(region.gpu.indices.slice(..), IndexFormat::Uint32);
            if let Some((buffer, offset)) = multi {
                crate::static_gx::count_draw(counter);
                pass.multi_draw_indexed_indirect(
                    buffer,
                    offset + at * ARGS_SIZE,
                    group.len() as u32,
                );
            } else {
                for d in group {
                    crate::static_gx::count_draw(counter);
                    pass.draw_indexed(d.first..d.first + d.count, d.base_vertex, 0..1);
                }
            }
        }
    }
}

/// Order `draws` by state, then index range, and fuse each draw into the one before it where its
/// indices continue that one's under the same state.
fn sort_and_fuse(draws: &mut Vec<GxDraw>) {
    draws.sort_unstable_by_key(|d| (d.state(), d.first));
    draws.dedup_by(|next, prev| {
        let joins = prev.state() == next.state() && prev.first + prev.count == next.first;
        if joins {
            prev.count += next.count;
        }
        joins
    });
}

/// One indirect draw for every draw of a state that has two or more, in [`GxBatch::emit`]'s
/// group order; a state with one draw is drawn directly and takes none.
fn multi_draw_args(draws: &[GxDraw], args: &mut Vec<DrawIndexedIndirectArgs>) {
    for group in draws.chunk_by(|a, b| a.state() == b.state()) {
        if group.len() < 2 {
            continue;
        }
        args.extend(group.iter().map(|d| DrawIndexedIndirectArgs {
            index_count: d.count,
            instance_count: 1,
            first_index: d.first,
            base_vertex: d.base_vertex,
            first_instance: 0,
        }));
    }
}

impl Drop for GxBatch<'_> {
    fn drop(&mut self) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.draws.clear();
        scratch.args.clear();
        self.pool
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depth-only draw of one slab, phase 0.
    fn draw(
        pipe: u8,
        slab: BufferId,
        index: BufferId,
        bind: BindGroupId,
        range: (u32, u32),
    ) -> GxDraw {
        GxDraw {
            phase: 0,
            pipe,
            vertex: slab,
            index,
            bind,
            base_vertex: 0,
            first: range.0,
            count: range.1,
            region: 0,
            slot: 0,
        }
    }

    /// Two regions' runs gathered interleaved come out grouped by pipeline, then region; ranges
    /// that continue each other fuse, and only a state left with two or more draws multi-draws.
    #[test]
    fn draws_group_by_state_and_fuse_contiguous_ranges() {
        let slab = BufferId::new();
        // Ids count up, so region `a` sorts before region `b`.
        let (a, b) = (BufferId::new(), BufferId::new());
        let (bind_a, bind_b) = (BindGroupId::new(), BindGroupId::new());
        let mut draws = vec![
            draw(1, slab, a, bind_a, (30, 6)),
            draw(0, slab, b, bind_b, (0, 3)),
            draw(0, slab, a, bind_a, (6, 6)),
            draw(0, slab, a, bind_a, (0, 6)),
            draw(0, slab, a, bind_a, (18, 3)),
        ];
        sort_and_fuse(&mut draws);
        let shape: Vec<(u8, BufferId, u32, u32)> = draws
            .iter()
            .map(|d| (d.pipe, d.index, d.first, d.count))
            .collect();
        assert_eq!(
            shape,
            vec![(0, a, 0, 12), (0, a, 18, 3), (0, b, 0, 3), (1, a, 30, 6)],
            "0..6 and 6..12 fuse; the cutout bucket follows every opaque draw"
        );
        let mut args = Vec::new();
        multi_draw_args(&draws, &mut args);
        let args: Vec<(u32, u32)> = args
            .iter()
            .map(|a| (a.first_index, a.index_count))
            .collect();
        assert_eq!(
            args,
            vec![(0, 12), (18, 3)],
            "region a's two opaque draws, nothing else"
        );
    }

    /// The colour pass's WMOs stay ahead of its doodads whatever their state.
    #[test]
    fn a_phase_is_never_crossed() {
        let slab = BufferId::new();
        let mut doodad = draw(0, slab, BufferId::new(), BindGroupId::new(), (0, 3));
        doodad.phase = 1;
        let wmo = draw(3, slab, BufferId::new(), BindGroupId::new(), (0, 3));
        let mut draws = vec![doodad, wmo];
        sort_and_fuse(&mut draws);
        assert_eq!((draws[0].phase, draws[1].phase), (0, 1));
    }
}
