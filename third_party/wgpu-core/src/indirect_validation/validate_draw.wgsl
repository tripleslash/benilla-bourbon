override supports_indirect_first_instance: bool;
override write_d3d12_special_constants: bool;

struct MetadataEntry {
    // bits 0..30 are an offset into `src`
    // bit 31 signifies that we are validating an indexed draw
    src_offset: u32,
    // bits 0..30 are an offset into `dst`
    // bit 30 is the most significant bit of `vertex_or_index_limit`
    // bit 31 is the most significant bit of `instance_limit`
    dst_offset: u32,
    vertex_or_index_limit: u32,
    instance_limit: u32,
    // BENILLA: the draw's index in its call.
    draw_index: u32,
    // BENILLA: for a draw of a multi-draw-indirect-count, the offset into `counts` of the call's
    // count; `NO_COUNT` for any other draw.
    count_offset: u32,
}
const OFFSET_MASK: u32 = 0x3FFFFFFF;
// BENILLA: `count_offset` of a draw that has no count.
const NO_COUNT: u32 = 0xFFFFFFFFu;

struct MetadataRange {
    start: u32,
    count: u32,
}
var<immediate> metadata_range: MetadataRange;

@group(0) @binding(0)
var<storage, read> metadata: array<MetadataEntry>;
@group(1) @binding(0)
var<storage, read> src: array<u32>;
@group(2) @binding(0)
var<storage, read_write> dst: array<u32>;
// BENILLA: the count buffer of the batch's multi-draw-indirect-count draws; another binding of
// the source layout (the source buffer itself) where the batch has none, never read then.
@group(3) @binding(0)
var<storage, read> counts: array<u32>;

fn is_bit_set(data: u32, index: u32) -> bool {
    return ((data >> index) & 1u) == 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3u) {
    if global_invocation_id.x >= metadata_range.count { return; }

    let metadata = metadata[metadata_range.start + global_invocation_id.x];

    // BENILLA: a draw past its call's count is never executed (`ExecuteIndirect` stops at the
    // count), so its arguments are not rewritten.
    if metadata.count_offset != NO_COUNT && metadata.draw_index >= counts[metadata.count_offset] {
        return;
    }

    var failed = false;

    let is_indexed = is_bit_set(metadata.src_offset, 31);
    let src_base_offset = metadata.src_offset & OFFSET_MASK;
    let dst_base_offset = metadata.dst_offset & OFFSET_MASK;

    let first_vertex_or_index = src[src_base_offset + 2];
    let vertex_or_index_count = src[src_base_offset + 0];

    {
        let overflow_permitted = is_bit_set(metadata.dst_offset, 30);
        let sub_overflows = metadata.vertex_or_index_limit < first_vertex_or_index;
        failed |= sub_overflows && !overflow_permitted;
        let vertex_or_index_limit = metadata.vertex_or_index_limit - first_vertex_or_index;
        failed |= vertex_or_index_limit < vertex_or_index_count;
    }

    let first_instance = src[src_base_offset + 3 + u32(is_indexed)];
    let instance_count = src[src_base_offset + 1];

    {
        let overflow_permitted = is_bit_set(metadata.dst_offset, 31);
        let sub_overflows = metadata.instance_limit < first_instance;
        failed |= sub_overflows && !overflow_permitted;
        let instance_limit = metadata.instance_limit - first_instance;
        failed |= instance_limit < instance_count;
    }

    if !supports_indirect_first_instance {
        failed |= first_instance != 0u;
    }

    let dst_offset = select(0u, 3u, write_d3d12_special_constants);
    if failed {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = 0u;
            dst[dst_base_offset + 1] = 0u;
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = 0u;
        dst[dst_base_offset + dst_offset + 1] = 0u;
        dst[dst_base_offset + dst_offset + 2] = 0u;
        dst[dst_base_offset + dst_offset + 3] = 0u;
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = 0u;
        }
    } else {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = src[src_base_offset + 2 + u32(is_indexed)];
            dst[dst_base_offset + 1] = src[src_base_offset + 3 + u32(is_indexed)];
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = src[src_base_offset + 0];
        dst[dst_base_offset + dst_offset + 1] = src[src_base_offset + 1];
        dst[dst_base_offset + dst_offset + 2] = src[src_base_offset + 2];
        dst[dst_base_offset + dst_offset + 3] = src[src_base_offset + 3];
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = src[src_base_offset + 4];
        }
    }
}
