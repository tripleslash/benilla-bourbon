// fork: the bindless static_gx group 1 (`static_gx/render/bindless.rs` has the contract): one
// region table and one texture-class array for every retained region. Imported by static_gx.wesl,
// static_gx_shadow.wesl and static_gx_voxel.wesl under `GX_BINDLESS` only.
//
// A draw's `first_instance` is its region's row in the table. Each entry point calls `enter` with
// it first (the vertex stage from `instance_index`, the fragment from the flat varying the vertex
// passed on); everything after reads the region through the helpers below.

// naga 30 refuses a `binding_array` without it; it reaches every importer's linked output.
enable wgpu_binding_array;

// The region table: per region `HEADER_ROWS` header rows, then one record per item.
//   row + 0: origin xyz (f32 bits), item count
//   row + 1: mesh-local bound centre xyz (f32 bits), first index in the arena
//   row + 2: bound half-extents xyz (f32 bits), index count
//   row + 3 + i: item i's record — x layer | class << 16, y batch order, z SIDN, w flags
@group(1) @binding(0) var<storage, read> gx_table: array<vec4<u32>>;
// Pool class c at element c (`bindless::GX_CLASS_SLOTS`).
@group(1) @binding(1) var gx_textures: binding_array<texture_2d_array<f32>, 1024>;
@group(1) @binding(2) var gx_repeat: sampler;
@group(1) @binding(3) var gx_clamp: sampler;

const HEADER_ROWS: u32 = 3u;

// This invocation's region row.
var<private> region_row: u32;

fn enter(row: u32) {
    region_row = row;
}

// The region's recentring origin: shader world = vertex + origin.
fn origin() -> vec3<f32> {
    return bitcast<vec3<f32>>(gx_table[region_row].xyz);
}

// The record of the item the vertex word names (its low 16 bits), x's class half included.
fn rec(word: u32) -> vec4<u32> {
    return gx_table[region_row + HEADER_ROWS + (word & 0xffffu)];
}

// The item's layer in its class's array.
fn layer_of(word: u32) -> i32 {
    return i32(rec(word).x & 0xffffu);
}

fn class_of(word: u32) -> u32 {
    return rec(word).x >> 16u;
}

// The item's texel through the repeat (`clamped == false`) or the clamp sampler, explicit gradients.
fn sample_grad(clamped: bool, uv: vec2<f32>, word: u32, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let r = rec(word);
    let c = r.x >> 16u;
    let l = i32(r.x & 0xffffu);
    if (clamped) {
        return textureSampleGrad(gx_textures[c], gx_clamp, uv, l, dx, dy);
    }
    return textureSampleGrad(gx_textures[c], gx_repeat, uv, l, dx, dy);
}

// The item's texel through the repeat sampler at an explicit level.
fn sample_level(uv: vec2<f32>, word: u32, level: f32) -> vec4<f32> {
    let r = rec(word);
    return textureSampleLevel(gx_textures[r.x >> 16u], gx_repeat, uv, i32(r.x & 0xffffu), level);
}

// The item's class array's base size in texels.
fn dims(word: u32) -> vec2<f32> {
    return vec2<f32>(textureDimensions(gx_textures[class_of(word)]).xy);
}
