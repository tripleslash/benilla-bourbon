//! fork: the bindless static_gx layout — one bind group, one region table and one index buffer
//! for every retained region, so a pipeline bucket draws across regions in one multi-draw.
//!
//! On where [`HwCaps::binding_arrays`] and [`HwCaps::mdi_count`] both hold, unless
//! `WOW_GX_BINDLESS=0`; the decision is made once at render startup ([`enabled`]) and every
//! static_gx pipeline (the colour pass and its depth prepass, the shadow cascades, the lamp
//! atlas, the GI voxelizer) is built for the one layout it picked, with the `GX_BINDLESS` shader
//! def. The web and the lever keep the per-region layout (`render.rs`, `cell_layout`): a uniform
//! origin, a record table and one bind group per pool class for each region.
//!
//! # Group 1, bindless (`benilla_modern::gx_bindless`)
//!
//! | binding | resource | stages |
//! |---|---|---|
//! | 0 | the region table, `array<vec4<u32>>` ([`GxBindless`]'s table) | vertex, fragment |
//! | 1 | `binding_array<texture_2d_array<f32>, 1024>`: pool class `c` at element `c` ([`GX_CLASS_SLOTS`]; elements past the pool's last class bind its white class) | fragment |
//! | 2 | the repeat sampler | vertex, fragment |
//! | 3 | the clamp sampler | vertex, fragment |
//!
//! The layout has no uniform and no dynamic offset: wgpu refuses either beside a binding array.
//!
//! # The region table
//!
//! One storage buffer of 16-byte rows. A region owns one contiguous block of it: [`HEADER_ROWS`]
//! header rows, then one record row per baked item, in bake order (the vertex word's low 16 bits
//! index it). The block's first row is the region's **row**: the draw passes it as
//! `first_instance`, so `instance_index` in the vertex stage names the region, and the vertex
//! hands it to the fragment as a flat varying.
//!
//! | row | x | y | z | w |
//! |---|---|---|---|---|
//! | `row + 0` | origin.x (f32 bits) | origin.y | origin.z | item count |
//! | `row + 1` | mesh-local bound centre.x (f32 bits) | centre.y | centre.z | first index in the arena |
//! | `row + 2` | bound half-extent.x (f32 bits) | half.y | half.z | index count |
//! | `row + 3 + i` | item `i`'s record | | | |
//!
//! A record is the per-region layout's record with the pool class in x's high half:
//! x = layer (bits 0..16) | class (bits 16..32), y = the MOBA batch-order nudge, z = SIDN
//! r | g << 8 | b << 16 in gamma bytes, w = flags (bit 0 exile kill, bits 1..=13 the
//! interior-prop probe slot, bit 14 interior fog). The kill and fog syncs rewrite a region's
//! record rows in place ([`GxBindless::write_records`]).
//!
//! # The index arena
//!
//! One `u32` index buffer (`INDEX | STORAGE | COPY_DST | COPY_SRC`, plus `BLAS_INPUT` where
//! [`HwCaps::ray_query`]). A region's indices are one block of it, `index_base..index_base +
//! index count` (header row 1.w and 2.w), copied from the region's mesh slice at assembly in the
//! region's draw order; its runs' index ranges are relative to `index_base`, and a draw adds it.
//! The vertices stay in the mesh allocator's slab: a draw's `base_vertex` is the region's slab
//! offset, so a whole-region block is `indices[index_base..][..count] + base_vertex` into that slab.
//!
//! Both buffers are suballocated first-fit ([`RangeAlloc`]): a region frees its blocks when it
//! leaves the cache or re-assembles, and the map clear frees everything. A buffer that runs out
//! grows to the next power of two by a GPU copy, submitted at once behind every write and copy
//! already aimed at the old buffer, so the blocks keep their offsets; the bind group is rebuilt
//! when the table or the pool's class count changes.

use super::super::pool::GxTexturePool;
use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d_array,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::ops::Range;

use crate::modern::caps::HwCaps;

/// Elements of the texture binding array: the most pool classes the bindless layout binds. A
/// class past it is held back like an unready texture (warned once). Keep in sync with
/// `shaders/modern/gx_bindless.wgsl`.
pub(super) const GX_CLASS_SLOTS: u32 = 1024;

/// A region block's header rows, ahead of its records. Keep in sync with `gx_bindless.wgsl`.
pub(super) const HEADER_ROWS: u32 = 3;

/// Bytes of a table row.
const ROW: u64 = 16;

/// The table's floor, in rows (256 KiB).
const MIN_ROWS: u32 = 1 << 14;

/// The arena's floor, in indices (4 MiB).
const MIN_INDICES: u32 = 1 << 20;

/// Whether this session draws static_gx bindless: the device can, and `WOW_GX_BINDLESS=0` did not
/// turn it off. Never on wasm32 (every [`HwCaps`] field is false there).
pub(super) fn enabled(caps: Option<&HwCaps>) -> bool {
    caps.is_some_and(|c| c.binding_arrays && c.mdi_count)
        && crate::launch_env::var("WOW_GX_BINDLESS").as_deref() != Some("0")
}

/// The bindless group 1 layout (see the module header).
pub(super) fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "static_gx_bindless_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Float { filterable: true })
                    .count(std::num::NonZero::new(GX_CLASS_SLOTS).unwrap())
                    .visibility(ShaderStages::FRAGMENT),
                sampler(SamplerBindingType::Filtering),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}

/// A region's blocks: its table row and its index range in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GxSlots {
    /// The header's first row; the region's `first_instance`.
    pub row: u32,
    /// Rows held: [`HEADER_ROWS`] plus one per item.
    pub rows: u32,
    /// The region's first index in the arena; its run ranges are relative to it.
    pub index_base: u32,
    pub index_len: u32,
}

/// First-fit suballocation of a `u32` range: free blocks kept sorted and coalesced, and the
/// high-water mark the buffer behind it has to hold.
#[derive(Default)]
pub(super) struct RangeAlloc {
    free: Vec<Range<u32>>,
    end: u32,
}

impl RangeAlloc {
    /// A block of `n` (at least 1), the lowest free one that fits, else at the end.
    pub(super) fn alloc(&mut self, n: u32) -> u32 {
        let n = n.max(1);
        if let Some(i) = self.free.iter().position(|f| f.len() as u32 >= n) {
            let at = self.free[i].start;
            self.free[i].start += n;
            if self.free[i].is_empty() {
                self.free.remove(i);
            }
            return at;
        }
        let at = self.end;
        self.end += n;
        at
    }

    /// Return the block `at..at + n` (`n` as passed to [`Self::alloc`]).
    pub(super) fn free(&mut self, at: u32, n: u32) {
        let block = at..at + n.max(1);
        let i = self.free.partition_point(|f| f.start < block.start);
        self.free.insert(i, block);
        // Coalesce with the next, then with the previous.
        if i + 1 < self.free.len() && self.free[i].end == self.free[i + 1].start {
            self.free[i].end = self.free[i + 1].end;
            self.free.remove(i + 1);
        }
        let i = if i > 0 && self.free[i - 1].end == self.free[i].start {
            self.free[i - 1].end = self.free[i].end;
            self.free.remove(i);
            i - 1
        } else {
            i
        };
        // A free block at the top lowers the high-water mark.
        if self.free[i].end == self.end {
            self.end = self.free[i].start;
            self.free.remove(i);
        }
    }

    /// The high-water mark: every block lies below it.
    pub(super) fn end(&self) -> u32 {
        self.end
    }
}

/// A buffer that grows by copy, so the blocks in it keep their offsets. Also the ray-traced
/// scene's arenas (`rt_scene`).
pub(super) struct GrowBuffer {
    label: &'static str,
    usage: BufferUsages,
    /// Bytes per unit the allocator counts in.
    unit: u64,
    min: u32,
    buffer: Option<Buffer>,
    /// `buffer`'s size in units.
    capacity: u32,
}

impl GrowBuffer {
    pub(super) fn new(label: &'static str, usage: BufferUsages, unit: u64, min: u32) -> Self {
        Self {
            label,
            usage: usage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            unit,
            min,
            buffer: None,
            capacity: 0,
        }
    }

    /// Hold at least `end` units. A grown buffer takes the old one's contents by a copy appended
    /// to `copies`, which is then submitted at once and replaced by a fresh encoder: every copy
    /// recorded into `copies` and every queue write aimed at the old buffer lands before the
    /// growth copy, and everything after aims at the new one.
    pub(super) fn reserve(
        &mut self,
        end: u32,
        device: &RenderDevice,
        queue: &RenderQueue,
        copies: &mut CommandEncoder,
    ) {
        if self.buffer.is_some() && end <= self.capacity {
            return;
        }
        let capacity = end.next_power_of_two().max(self.min);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(self.label),
            size: u64::from(capacity) * self.unit,
            usage: self.usage,
            mapped_at_creation: false,
        });
        if let Some(old) = self.buffer.take() {
            let fresh = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("static_gx_index_copies"),
            });
            let mut encoder = std::mem::replace(copies, fresh);
            encoder.copy_buffer_to_buffer(
                &old,
                0,
                &buffer,
                0,
                u64::from(self.capacity) * self.unit,
            );
            queue.submit([encoder.finish()]);
        }
        self.buffer = Some(buffer);
        self.capacity = capacity;
    }

    /// The buffer, once anything was reserved.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))] // read by `rt_scene`, native only
    pub(super) fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Bytes held.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))] // read by `rt_scene`, native only
    pub(super) fn bytes(&self) -> u64 {
        u64::from(self.capacity) * self.unit
    }
}

/// What a bindless draw binds: the shared bind group and the index arena.
#[derive(Clone, Copy)]
pub(super) struct GxShared<'w> {
    pub bind: &'w BindGroup,
    pub indices: &'w Buffer,
}

/// The shared table, the index arena and the one bind group over them and the pool. Present in
/// the render world only while the session draws bindless ([`enabled`]).
#[derive(Resource)]
pub(super) struct GxBindless {
    table: GrowBuffer,
    rows: RangeAlloc,
    indices: GrowBuffer,
    index_alloc: RangeAlloc,
    bind: Option<BindGroup>,
    /// `(pool classes, table buffer)` the bind group was built over.
    bound: Option<(u32, BufferId)>,
    warned_overflow: bool,
}

impl GxBindless {
    /// `ray_query`: the arena is also a BLAS input (`BLAS_INPUT`), for the ray-traced lanes.
    pub(super) fn new(ray_query: bool) -> Self {
        let mut index_usage = BufferUsages::INDEX | BufferUsages::STORAGE;
        if ray_query {
            index_usage |= BufferUsages::BLAS_INPUT;
        }
        Self {
            table: GrowBuffer::new(
                "static_gx_region_table",
                BufferUsages::STORAGE,
                ROW,
                MIN_ROWS,
            ),
            rows: RangeAlloc::default(),
            indices: GrowBuffer::new("static_gx_index_arena", index_usage, 4, MIN_INDICES),
            index_alloc: RangeAlloc::default(),
            bind: None,
            bound: None,
            warned_overflow: false,
        }
    }

    /// Place a region: allocate its table block and index range, write its header and `records`.
    /// The caller copies its indices into [`Self::index_buffer`] at `index_base` through `copies`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place(
        &mut self,
        origin: Vec3,
        aabb: &Aabb,
        records: &[[u32; 4]],
        index_len: u32,
        device: &RenderDevice,
        queue: &RenderQueue,
        copies: &mut CommandEncoder,
    ) -> GxSlots {
        let rows = HEADER_ROWS + records.len() as u32;
        let row = self.rows.alloc(rows);
        self.table.reserve(self.rows.end(), device, queue, copies);
        let index_base = self.index_alloc.alloc(index_len);
        self.indices
            .reserve(self.index_alloc.end(), device, queue, copies);
        let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
        let header: [[u32; 4]; HEADER_ROWS as usize] = [
            [
                origin.x.to_bits(),
                origin.y.to_bits(),
                origin.z.to_bits(),
                records.len() as u32,
            ],
            [c.x.to_bits(), c.y.to_bits(), c.z.to_bits(), index_base],
            [h.x.to_bits(), h.y.to_bits(), h.z.to_bits(), index_len],
        ];
        let table = self.table.buffer.as_ref().expect("reserved above");
        queue.write_buffer(table, u64::from(row) * ROW, bytemuck::cast_slice(&header));
        let slots = GxSlots {
            row,
            rows,
            index_base,
            index_len,
        };
        self.write_records(queue, slots, records);
        slots
    }

    /// Rewrite a placed region's record rows.
    pub(super) fn write_records(&self, queue: &RenderQueue, slots: GxSlots, records: &[[u32; 4]]) {
        if records.is_empty() {
            return;
        }
        if let Some(table) = &self.table.buffer {
            let at = u64::from(slots.row + HEADER_ROWS) * ROW;
            queue.write_buffer(table, at, bytemuck::cast_slice(records));
        }
    }

    /// Free a region's blocks.
    pub(super) fn release(&mut self, slots: GxSlots) {
        self.rows.free(slots.row, slots.rows);
        self.index_alloc.free(slots.index_base, slots.index_len);
    }

    /// The map clear: every block free, the buffers kept, the bind group stale (the pool resets
    /// with the map, and its new classes may number as many as the old).
    pub(super) fn reset(&mut self) {
        self.rows = RangeAlloc::default();
        self.index_alloc = RangeAlloc::default();
        self.bound = None;
    }

    /// The arena the assembly copies a region's indices into; present once a region is placed.
    pub(super) fn index_buffer(&self) -> Option<&Buffer> {
        self.indices.buffer.as_ref()
    }

    /// The region table; present once a region is placed. The ray-traced scene binds it.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))] // read by `rt_scene`, native only
    pub(super) fn table_buffer(&self) -> Option<&Buffer> {
        self.table.buffer.as_ref()
    }

    /// Whether pool class `class` has an element in the binding array; warns the first time one
    /// does not.
    pub(super) fn class_fits(&mut self, class: u16) -> bool {
        let fits = u32::from(class) < GX_CLASS_SLOTS;
        if !fits && !std::mem::replace(&mut self.warned_overflow, true) {
            warn!(
                "static_gx: the texture pool passed {GX_CLASS_SLOTS} classes; textures in the \
                 classes past it are not drawn (WOW_GX_BINDLESS=0 draws them)"
            );
        }
        fits
    }

    /// Rebuild the bind group if the table buffer or the pool's class count moved since it was
    /// built. Element `c` of the array is class `c`; the rest bind the white class.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn refresh_bind(
        &mut self,
        pool: &mut GxTexturePool,
        layout: &BindGroupLayout,
        repeat: &Sampler,
        clamp: &Sampler,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        let Some(table) = self.table.buffer.as_ref() else {
            return;
        };
        let white = pool.white(device, queue);
        let classes = pool.class_count().min(GX_CLASS_SLOTS as usize) as u32;
        let key = (classes, table.id());
        if self.bound == Some(key) && self.bind.is_some() {
            return;
        }
        let views: Vec<&WgpuTextureView> = (0..GX_CLASS_SLOTS)
            .map(|c| {
                let class = if c < classes { c as u16 } else { white };
                &**pool.view(class)
            })
            .collect();
        self.bind = Some(device.create_bind_group(
            "static_gx_bindless",
            layout,
            &BindGroupEntries::sequential((table.as_entire_binding(), &views[..], repeat, clamp)),
        ));
        self.bound = Some(key);
    }

    /// What a draw binds, once a region is placed and the bind group built.
    pub(super) fn shared(&self) -> Option<GxShared<'_>> {
        Some(GxShared {
            bind: self.bind.as_ref()?,
            indices: self.indices.buffer.as_ref()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks come back lowest-first, freed neighbours coalesce, and a free block at the top
    /// lowers the high-water mark.
    #[test]
    fn freed_blocks_coalesce_and_are_reused_first_fit() {
        let mut a = RangeAlloc::default();
        let (x, y, z) = (a.alloc(4), a.alloc(4), a.alloc(4));
        assert_eq!((x, y, z, a.end()), (0, 4, 8, 12));
        a.free(x, 4);
        a.free(y, 4);
        assert_eq!(a.free, vec![0..8], "neighbours coalesce");
        assert_eq!(a.alloc(6), 0, "first fit");
        assert_eq!(a.alloc(3), 12, "the 2 left at 6..8 are too few");
        a.free(12, 3);
        assert_eq!(a.end(), 12, "the top block lowers the mark");
        a.free(z, 4);
        assert_eq!(
            (a.end(), a.free.len()),
            (6, 0),
            "6..12 joins and lowers the mark"
        );
        assert_eq!(a.alloc(0), 6, "an empty block still takes one");
    }

    /// The WGSL library declares the same array size and header length as the layout here.
    #[test]
    fn the_shader_library_agrees_with_the_layout() {
        let src = include_str!("../../shaders/modern/gx_bindless.wgsl");
        assert!(src.contains(&format!(
            "binding_array<texture_2d_array<f32>, {GX_CLASS_SLOTS}>"
        )));
        assert!(src.contains(&format!("const HEADER_ROWS: u32 = {HEADER_ROWS}u;")));
    }
}
