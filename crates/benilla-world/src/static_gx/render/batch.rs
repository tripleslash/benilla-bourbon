//! fork: the static pass's draw list. Every pass that draws retained regions — the colour pass and
//! its depth prepass, the water mirror, the shadow cascades, the local-light atlas, the GI
//! voxelizer — gathers its admitted runs into a [`GxBatch`], which sorts them by state, fuses runs
//! that continue each other's index range, and issues them.
//!
//! The sort is the saving on both targets: [`TrackedRenderPass`] drops a `set_*` that repeats the
//! bound state, so consecutive draws that share a pipeline, bind group and buffers cost one bind,
//! and the browser, which validates every call it is sent, gains the most. On native, where the
//! device draws a multi-draw in hardware ([`HwCaps::mdi_count`]: wgpu 29 has no separate
//! `MULTI_DRAW_INDIRECT` feature and emulates the call as a loop without it), a state's draws go
//! as one multi-draw over this frame's indirect buffer.
//!
//! A batch holds no pass, so recording passes on several threads only needs each thread to own
//! its batch: the scratch comes from a locked pool and the indirect ranges from atomic cursors.
//!
//! # Buckets
//!
//! A **bucket** is one state's draws as one multi-draw. Per region (the web, `WOW_GX_BINDLESS=0`)
//! a region's index buffer and bind groups are its own, so a bucket never leaves its region; it
//! is taken by a state with two draws or more and issued as `multi_draw_indexed_indirect`.
//! Bindless (`bindless`), every region binds the one bind group and the one index arena, so a
//! state is `(phase, pipeline, vertex slab)` and a bucket spans every region of the pass in that
//! slab. Every bindless state is a bucket, even of one draw, issued as
//! `multi_draw_indexed_indirect_count` with its count in the frame's count buffer.
//!
//! # The indirect layout (what a GPU cull writes in place)
//!
//! Two buffers per frame, both `INDIRECT | STORAGE | COPY_DST`, carved by [`GxBatch::finish`]:
//!
//! - **args**: [`DrawIndexedIndirectArgs`], 20 bytes. A bucket owns `capacity` contiguous slots
//!   from `args_at`, one per draw in `(row, first index)` order. A draw's slot holds
//!   `index_count`, `instance_count = 1`, `first_index` (the arena index, `index_base` added),
//!   `base_vertex` (the region's slab offset) and `first_instance = row`, the region's first row
//!   in the region table (`bindless`): the shader reads the region from `instance_index`.
//! - **counts**: one `u32` per counted bucket at `count_at`, written as `capacity` by the CPU.
//!
//! A GPU cull keeps both and changes only their contents: zero the bucket's count, test each
//! candidate draw (its region's bound sits in table rows `row + 1` and `row + 2`), and for each
//! survivor `atomicAdd` the count and write its args at `args_at + the old count`. The draw call
//! already reads the count, so nothing on the CPU side moves.

use super::bindless::{GxBindless, GxShared};
use super::{GxCellDraw, GxCellGpu, GxRegionBinds, GxRun, GxSel};
use crate::modern::caps::HwCaps;
use bevy::prelude::*;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{
    BindGroup, BindGroupId, Buffer, BufferDescriptor, BufferId, BufferUsages,
    DrawIndexedIndirectArgs, IndexFormat, RenderPipeline,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};

/// Bytes of one indirect draw's arguments.
const ARGS_SIZE: u64 = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;

/// Bytes of one bucket's draw count.
const COUNT_SIZE: u64 = 4;

/// The indirect buffer's floor, in draws: what a first frame grows it to at least.
const MIN_ARGS: u32 = 4096;

/// The count buffer's floor, in buckets.
const MIN_COUNTS: u32 = 256;

/// The pipeline keys `(cutout, two_sided)` in [`pipe_class`] order: a four-entry pipeline table
/// is built by mapping these.
pub(super) const PIPE_KEYS: [(bool, bool); 4] =
    [(false, false), (false, true), (true, false), (true, true)];

/// The four-entry pipeline table's index of `(cutout, two_sided)`, as a batch's pipeline class
/// (an `Option`, the shape every `push_*` class function takes).
pub(super) fn pipe_class(cutout: bool, two_sided: bool) -> Option<u8> {
    Some(u8::from(cutout) * 2 + u8::from(two_sided))
}

/// Everything a draw binds, in sort order. The vertex buffer (a mesh-allocator slab) comes before
/// the index buffer and bind group, so a slab binds once for all the regions in it. Per region the
/// index buffer and bind group are the region's own, which keeps its draws together; bindless they
/// are the arena and the shared group, and `base_vertex` is 0 (it rides in each draw's args).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct GxState {
    /// Draw-order phase: the colour pass's WMOs before its doodads, 1.12's drain order; 0 elsewhere.
    phase: u8,
    /// Index into the pipeline table the batch is emitted with.
    pipe: u8,
    vertex: BufferId,
    index: BufferId,
    bind: BindGroupId,
    base_vertex: i32,
}

/// One draw of a batch: its state, then what goes in its args.
#[derive(Clone, Copy, Debug)]
struct GxDraw {
    state: GxState,
    /// `first_instance`: the region's table row bindless, 0 per region.
    row: u32,
    /// The region's slab offset, whatever `state.base_vertex` says.
    base_vertex: i32,
    /// The first index in the bound index buffer (`index_base` added, bindless).
    first: u32,
    count: u32,
    /// The region in [`GxBatch::regions`] and its bind-group slot, to find what the ids name.
    region: u32,
    slot: u32,
}

/// A region a batch draws, with the vertex slice its mesh lives in.
#[derive(Clone, Copy)]
struct GxRegion<'w> {
    gpu: &'w GxCellGpu,
    vertex: &'w Buffer,
    base_vertex: i32,
    /// The region's table row and first arena index bindless; `(0, 0)` per region.
    row: u32,
    index_base: u32,
}

/// One state's draws as one multi-draw: `capacity` args from `args_at`, and where counted, the
/// count at `count_at`. Offsets are in units of the batch's own ranges ([`GxBatch::finish`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GxBucket {
    args_at: u32,
    capacity: u32,
    count_at: Option<u32>,
}

/// A batch's reusable allocations.
#[derive(Default)]
struct GxScratch {
    draws: Vec<GxDraw>,
    args: Vec<DrawIndexedIndirectArgs>,
    counts: Vec<u32>,
    /// One per state group, in [`GxBatch::emit`]'s order: its bucket, or `None` to draw directly.
    buckets: Vec<Option<GxBucket>>,
}

impl GxScratch {
    fn clear(&mut self) {
        self.draws.clear();
        self.args.clear();
        self.counts.clear();
        self.buckets.clear();
    }
}

/// The batches' scratch, returned on drop, so a frame's gathers reuse last frame's allocations.
#[derive(Resource, Default)]
pub(super) struct GxDrawPool(Mutex<Vec<GxScratch>>);

/// One of this frame's indirect buffers. Batches take disjoint ranges of it from `cursor` and
/// write them through the queue, which lands every write before the frame's command buffers run.
/// A batch whose range does not fit draws directly; the next frame's buffer is sized to the cursor.
struct GxFrameBuffer {
    label: &'static str,
    /// Bytes per unit.
    unit: u64,
    /// The floor a first growth reaches, in units.
    min: u32,
    buffer: Option<Buffer>,
    /// `buffer`'s size in units.
    capacity: u32,
    /// Units asked for this frame, granted or not.
    cursor: AtomicU32,
}

impl GxFrameBuffer {
    fn new(label: &'static str, unit: u64, min: u32) -> Self {
        Self {
            label,
            unit,
            min,
            buffer: None,
            capacity: 0,
            cursor: AtomicU32::new(0),
        }
    }

    /// `n` units of this frame's buffer: the buffer and the first unit, or `None` if they do not
    /// fit (they are counted anyway, so the next frame's buffer holds them).
    fn take(&self, n: u32) -> Option<(&Buffer, u32)> {
        let start = self.cursor.fetch_add(n, Relaxed);
        let fits = start.checked_add(n).is_some_and(|end| end <= self.capacity);
        self.buffer.as_ref().filter(|_| fits).map(|b| (b, start))
    }

    /// Rewind the cursor, and grow the buffer to what last frame asked for if that did not fit.
    /// The old buffer is dropped, not copied: its contents were last frame's.
    fn grow(&mut self, device: &RenderDevice) {
        let asked = std::mem::take(self.cursor.get_mut());
        if asked <= self.capacity {
            return;
        }
        let capacity = asked.next_power_of_two().max(self.min);
        self.buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some(self.label),
            size: u64::from(capacity) * self.unit,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.capacity = capacity;
    }
}

/// This frame's indirect arguments and bucket counts.
#[derive(Resource)]
pub(super) struct GxIndirect {
    /// Native multi-draw is available and not switched off (`$WOW_STATIC_GX_MDI=0`).
    native: bool,
    args: GxFrameBuffer,
    counts: GxFrameBuffer,
}

/// `RenderStartup`, after [`crate::modern::caps::init`]: multi-draw where the device draws one in
/// hardware ([`HwCaps::mdi_count`], never on wasm32) and the lever leaves it on.
pub(super) fn init_indirect(mut commands: Commands, caps: Option<Res<HwCaps>>) {
    commands.insert_resource(GxIndirect {
        native: caps.is_some_and(|c| c.mdi_count)
            && crate::launch_env::var("WOW_STATIC_GX_MDI").as_deref() != Some("0"),
        args: GxFrameBuffer::new("static_gx_indirect", ARGS_SIZE, MIN_ARGS),
        counts: GxFrameBuffer::new("static_gx_indirect_counts", COUNT_SIZE, MIN_COUNTS),
    });
}

/// Rewind both cursors, and grow a buffer to what last frame asked of it if that did not fit.
pub(super) fn prepare_indirect(mut indirect: ResMut<GxIndirect>, device: Res<RenderDevice>) {
    if !indirect.native {
        return;
    }
    indirect.args.grow(&device);
    indirect.counts.grow(&device);
}

/// One pass's draws: gathered with [`Self::region`] and the `push_*` calls, ordered by
/// [`Self::finish`], issued by [`Self::emit`] — as often as passes draw the same list, which is
/// how the depth prepass and the colour pass stay identical.
pub(super) struct GxBatch<'w> {
    pool: &'w GxDrawPool,
    indirect: &'w GxIndirect,
    queue: &'w RenderQueue,
    /// The bindless bind group and arena, once built; `None` per region, and bindless before the
    /// first region is placed.
    shared: Option<GxShared<'w>>,
    regions: Vec<GxRegion<'w>>,
    scratch: GxScratch,
    /// Runs gathered, before any fused: the meter's measure of what the passes asked for.
    gathered: u32,
    /// This batch's ranges of the args and count buffers, once [`Self::finish`] got them.
    args_at: Option<(&'w Buffer, u32)>,
    counts_at: Option<(&'w Buffer, u32)>,
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
            shared: world
                .get_resource::<GxBindless>()
                .and_then(GxBindless::shared),
            regions: Vec::new(),
            scratch,
            gathered: 0,
            args_at: None,
            counts_at: None,
        }
    }

    /// Empty the batch for another gather, keeping its allocations.
    pub(super) fn clear(&mut self) {
        self.regions.clear();
        self.scratch.clear();
        self.gathered = 0;
        self.args_at = None;
        self.counts_at = None;
    }

    /// Whether any region was added since the last [`Self::clear`].
    pub(super) fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Add a region, drawn from its mesh's vertex slice; `None` while the mesh has none, or while
    /// a bindless region has no shared bind group to draw with.
    pub(super) fn region(
        &mut self,
        gpu: &'w GxCellGpu,
        draw: &GxCellDraw,
        allocator: &'w MeshAllocator,
    ) -> Option<u32> {
        let (row, index_base) = match &gpu.binds {
            GxRegionBinds::Own { .. } => (0, 0),
            GxRegionBinds::Shared(slots) => {
                self.shared?;
                (slots.row, slots.index_base)
            }
        };
        let vslice = allocator.mesh_vertex_slice(&draw.mesh.id())?;
        self.regions.push(GxRegion {
            gpu,
            vertex: vslice.buffer,
            base_vertex: i32::try_from(vslice.range.start).unwrap_or(0),
            row,
            index_base,
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
        let (bind, indices) = self.binds(&region, run.slot);
        let state_base = match region.gpu.binds {
            GxRegionBinds::Own { .. } => region.base_vertex,
            GxRegionBinds::Shared(_) => 0,
        };
        self.gathered += 1;
        self.scratch.draws.push(GxDraw {
            state: GxState {
                phase,
                pipe,
                vertex: region.vertex.id(),
                index: indices.id(),
                bind: bind.id(),
                base_vertex: state_base,
            },
            row: region.row,
            base_vertex: region.base_vertex,
            first: region.index_base + run.index_range.start,
            count: run.index_range.len() as u32,
            region: r,
            slot: run.slot as u32,
        });
    }

    /// The bind group and index buffer a draw of `region`'s run slot `slot` binds.
    fn binds(&self, region: &GxRegion<'w>, slot: usize) -> (&'w BindGroup, &'w Buffer) {
        let gpu: &'w GxCellGpu = region.gpu;
        match &gpu.binds {
            GxRegionBinds::Own {
                bind_groups,
                indices,
                ..
            } => (&bind_groups[slot].1, indices),
            GxRegionBinds::Shared(_) => {
                let shared = self
                    .shared
                    .expect("`region` admits a shared region only with the shared bind group");
                (shared.bind, shared.indices)
            }
        }
    }

    /// Order the gathered draws by state, fuse each draw into the one before it where its indices
    /// continue that one's under the same state and row, and on native lay out the buckets and
    /// write their args and counts. Opaque and cutout runs only, depth-tested: the order inside a
    /// phase changes no pixel but an exact depth tie.
    pub(super) fn finish(&mut self) {
        let indirect: &'w GxIndirect = self.indirect;
        self.args_at = None;
        self.counts_at = None;
        let regions = &self.regions;
        let GxScratch {
            draws,
            args,
            counts,
            buckets,
        } = &mut self.scratch;
        sort_and_fuse(draws);
        args.clear();
        counts.clear();
        buckets.clear();
        if !indirect.native {
            return;
        }
        plan_buckets(
            draws,
            |d| {
                matches!(
                    regions[d.region as usize].gpu.binds,
                    GxRegionBinds::Shared(_)
                )
            },
            args,
            counts,
            buckets,
        );
        if args.is_empty() {
            buckets.clear();
            return;
        }
        // Both ranges or neither: a batch short of either draws directly this frame, and
        // `prepare_indirect` grows the buffers for the next. `counts_at` is `Some(None)` where no
        // bucket is counted.
        let args_at = indirect.args.take(args.len() as u32);
        let counts_at = if counts.is_empty() {
            Some(None)
        } else {
            indirect.counts.take(counts.len() as u32).map(Some)
        };
        let (Some((args_buffer, args_start)), Some(counts_at)) = (args_at, counts_at) else {
            buckets.clear();
            return;
        };
        self.queue.write_buffer(
            args_buffer,
            u64::from(args_start) * ARGS_SIZE,
            bytemuck::cast_slice(args.as_slice()),
        );
        if let Some((counts_buffer, counts_start)) = counts_at {
            self.queue.write_buffer(
                counts_buffer,
                u64::from(counts_start) * COUNT_SIZE,
                bytemuck::cast_slice(counts.as_slice()),
            );
        }
        self.args_at = args_at;
        self.counts_at = counts_at;
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
        // One entry per state group, in this same group order; empty when nothing goes indirect.
        let mut buckets = self.scratch.buckets.iter().copied();
        for group in self.scratch.draws.chunk_by(|a, b| a.state == b.state) {
            let bucket = buckets.next().flatten();
            let head = group[0];
            let Some(pipeline) = pipes.get(usize::from(head.state.pipe)).copied().flatten() else {
                continue;
            };
            let region = self.regions[head.region as usize];
            let (bind, indices) = self.binds(&region, head.slot as usize);
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(1, bind, &[]);
            pass.set_vertex_buffer(0, region.vertex.slice(..));
            pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
            match (bucket, self.args_at) {
                (Some(b), Some((args, args_start))) => {
                    crate::static_gx::count_draw(counter);
                    let args_offset = u64::from(args_start + b.args_at) * ARGS_SIZE;
                    match (b.count_at, self.counts_at) {
                        (Some(c), Some((counts, counts_start))) => pass
                            .multi_draw_indexed_indirect_count(
                                args,
                                args_offset,
                                counts,
                                u64::from(counts_start + c) * COUNT_SIZE,
                                b.capacity,
                            ),
                        _ => pass.multi_draw_indexed_indirect(args, args_offset, b.capacity),
                    }
                }
                _ => {
                    for d in group {
                        crate::static_gx::count_draw(counter);
                        pass.draw_indexed(
                            d.first..d.first + d.count,
                            d.base_vertex,
                            d.row..d.row + 1,
                        );
                    }
                }
            }
        }
    }
}

/// Order `draws` by state, row, then index range, and fuse each draw into the one before it where
/// its indices continue that one's under the same state and row (the same region, bindless).
fn sort_and_fuse(draws: &mut Vec<GxDraw>) {
    draws.sort_unstable_by_key(|d| (d.state, d.row, d.first));
    draws.dedup_by(|next, prev| {
        let joins = prev.state == next.state
            && prev.row == next.row
            && prev.first + prev.count == next.first;
        if joins {
            prev.count += next.count;
        }
        joins
    });
}

/// Lay out one bucket per state group that takes one, in [`GxBatch::emit`]'s group order, with
/// its args (and count) appended: a counted group (`counted` of its first draw: bindless) always
/// takes one, with its count; any other only with two draws or more, uncounted. `buckets` gets one
/// entry per group.
fn plan_buckets(
    draws: &[GxDraw],
    counted: impl Fn(&GxDraw) -> bool,
    args: &mut Vec<DrawIndexedIndirectArgs>,
    counts: &mut Vec<u32>,
    buckets: &mut Vec<Option<GxBucket>>,
) {
    for group in draws.chunk_by(|a, b| a.state == b.state) {
        let counted = counted(&group[0]);
        if !counted && group.len() < 2 {
            buckets.push(None);
            continue;
        }
        let capacity = group.len() as u32;
        let count_at = counted.then(|| {
            counts.push(capacity);
            counts.len() as u32 - 1
        });
        buckets.push(Some(GxBucket {
            args_at: args.len() as u32,
            capacity,
            count_at,
        }));
        args.extend(group.iter().map(|d| DrawIndexedIndirectArgs {
            index_count: d.count,
            instance_count: 1,
            first_index: d.first,
            base_vertex: d.base_vertex,
            first_instance: d.row,
        }));
    }
}

impl Drop for GxBatch<'_> {
    fn drop(&mut self) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
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

    /// A per-region draw of one slab, phase 0: row 0, the slab offset in the state.
    fn draw(
        pipe: u8,
        slab: BufferId,
        index: BufferId,
        bind: BindGroupId,
        range: (u32, u32),
    ) -> GxDraw {
        GxDraw {
            state: GxState {
                phase: 0,
                pipe,
                vertex: slab,
                index,
                bind,
                base_vertex: 0,
            },
            row: 0,
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
            .map(|d| (d.state.pipe, d.state.index, d.first, d.count))
            .collect();
        assert_eq!(
            shape,
            vec![(0, a, 0, 12), (0, a, 18, 3), (0, b, 0, 3), (1, a, 30, 6)],
            "0..6 and 6..12 fuse; the cutout bucket follows every opaque draw"
        );
        let (mut args, mut counts, mut buckets) = (Vec::new(), Vec::new(), Vec::new());
        plan_buckets(&draws, |_| false, &mut args, &mut counts, &mut buckets);
        let args: Vec<(u32, u32)> = args
            .iter()
            .map(|a| (a.first_index, a.index_count))
            .collect();
        assert_eq!(
            args,
            vec![(0, 12), (18, 3)],
            "region a's two opaque draws, nothing else"
        );
        assert!(counts.is_empty(), "per region, no bucket is counted");
        assert_eq!(
            buckets,
            vec![
                Some(GxBucket {
                    args_at: 0,
                    capacity: 2,
                    count_at: None
                }),
                None,
                None
            ],
            "one entry per state group"
        );
    }

    /// The colour pass's WMOs stay ahead of its doodads whatever their state.
    #[test]
    fn a_phase_is_never_crossed() {
        let slab = BufferId::new();
        let mut doodad = draw(0, slab, BufferId::new(), BindGroupId::new(), (0, 3));
        doodad.state.phase = 1;
        let wmo = draw(3, slab, BufferId::new(), BindGroupId::new(), (0, 3));
        let mut draws = vec![doodad, wmo];
        sort_and_fuse(&mut draws);
        assert_eq!((draws[0].state.phase, draws[1].state.phase), (0, 1));
    }

    /// Bindless, regions in one slab share a state: their draws go as one counted bucket, each
    /// with its region's row as `first_instance` and slab offset as `base_vertex`. Ranges that
    /// touch in the arena fuse only inside a region, and a lone draw in another slab is still
    /// a bucket of its own.
    #[test]
    fn bindless_buckets_span_regions() {
        let (slab, other) = (BufferId::new(), BufferId::new());
        let (arena, bind) = (BufferId::new(), BindGroupId::new());
        let shared = |vertex, row, base_vertex, range: (u32, u32)| {
            let mut d = draw(0, vertex, arena, bind, range);
            d.row = row;
            d.base_vertex = base_vertex;
            d
        };
        // Region at row 0 owns arena 0..12, region at row 7 owns 12..20; a third, in another slab,
        // sits at row 20.
        let mut draws = vec![
            shared(slab, 7, 500, (12, 8)),
            shared(slab, 0, 100, (6, 6)),
            shared(other, 20, 0, (40, 3)),
            shared(slab, 0, 100, (0, 6)),
        ];
        sort_and_fuse(&mut draws);
        let shape: Vec<(u32, u32, u32)> = draws.iter().map(|d| (d.row, d.first, d.count)).collect();
        assert_eq!(
            shape,
            vec![(0, 0, 12), (7, 12, 8), (20, 40, 3)],
            "row 0's halves fuse; 0..12 and 12..20 touch but are two regions"
        );
        let (mut args, mut counts, mut buckets) = (Vec::new(), Vec::new(), Vec::new());
        plan_buckets(&draws, |_| true, &mut args, &mut counts, &mut buckets);
        let args: Vec<(u32, u32, i32, u32)> = args
            .iter()
            .map(|a| {
                (
                    a.first_index,
                    a.index_count,
                    a.base_vertex,
                    a.first_instance,
                )
            })
            .collect();
        assert_eq!(args, vec![(0, 12, 100, 0), (12, 8, 500, 7), (40, 3, 0, 20)]);
        assert_eq!(
            counts,
            vec![2, 1],
            "each count starts at its bucket's capacity"
        );
        assert_eq!(
            buckets,
            vec![
                Some(GxBucket {
                    args_at: 0,
                    capacity: 2,
                    count_at: Some(0)
                }),
                Some(GxBucket {
                    args_at: 2,
                    capacity: 1,
                    count_at: Some(1)
                }),
            ],
            "one bucket per slab, the lone draw included"
        );
    }
}
