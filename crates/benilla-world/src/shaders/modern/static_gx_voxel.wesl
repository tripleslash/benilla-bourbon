// Modern lighting (FORK.md, P5): the retained static world rasterized into the GI
// voxel clipmap — the fork's stand-in for the plan's per-model proxy bake.
//
// The pass draws the same cells, WMO groups and props the modern shadow pass draws, but through an
// orthographic box camera whose fragments write the voxel they land in (albedo and the surface
// normal) instead of a colour. It runs once per axis, so a triangle edge-on to one sweep is caught
// by another; the raster is supersampled against the voxel grid, so a wall thinner than a voxel
// still fills one.
//
// Under `VOXEL_CONSERVATIVE` (native, conservative rasterization) the raster is one pixel a voxel
// and every pixel a triangle touches runs: each triangle writes only in the sweep down its plane's
// dominant axis, the one or more voxels its plane crosses over that pixel.
//
// Group 1 is static_gx's own cell layout, bound unchanged (see `static_gx.wesl`); under
// `GX_BINDLESS` the shared region table (`benilla_modern::gx_bindless`).
@if(GX_BINDLESS) import benilla_world::shaders::modern::gx_bindless;

struct VoxelView {
    clip_from_world: mat4x4<f32>,
    // xyz = the dirty box's low voxel coord, w = this level's voxel size (yd).
    box_lo: vec4<f32>,
    // xyz = the box's high voxel coord (exclusive), w = voxels per side.
    box_hi: vec4<f32>,
    // x = the level's slice offset in the texture, y = the alpha key, z = the albedo mip,
    // w = the sweep's axis (0 x, 1 y, 2 z) under `VOXEL_CONSERVATIVE`.
    info: vec4<f32>,
}
@group(0) @binding(0) var<uniform> vox: VoxelView;

@if(GX_BINDLESS) {
fn gx_rec(word: u32) -> vec4<u32> {
    return gx_bindless::rec(word);
}
fn gx_origin() -> vec3<f32> {
    return gx_bindless::origin();
}
} @else {
struct GxCell {
    origin: vec4<f32>,
}
@group(1) @binding(0) var<uniform> cell: GxCell;
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;
fn gx_rec(word: u32) -> vec4<u32> {
    return recs[word & 0xffffu];
}
fn gx_origin() -> vec3<f32> {
    return cell.origin.xyz;
}
}

@group(2) @binding(0) var vox_albedo: texture_storage_3d<rgba8unorm, write>;
@group(2) @binding(1) var vox_normal: texture_storage_3d<rgba8unorm, write>;

const WORD_TEXTURED: u32 = 2097152u;

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
    // The draw's `first_instance`: its region's table row.
    @if(GX_BINDLESS) @builtin(instance_index) instance: u32,
}

struct VoxelVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) word: u32,
    @if(GX_BINDLESS) @location(4) @interpolate(flat) region: u32,
}

@vertex
fn vertex(v: GxVertex) -> VoxelVsOut {
    var out: VoxelVsOut;
@if(GX_BINDLESS) {
    gx_bindless::enter(v.instance);
    out.region = v.instance;
}
    // A killed item (record bit 0) collapses outside the clip volume, as in every other lane.
    if ((gx_rec(v.word).w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let world = v.position + gx_origin();
    out.world = world;
    out.normal = v.normal;
    out.uv = v.uv;
    out.word = v.word;
    out.position = vox.clip_from_world * vec4<f32>(world, 1.0);
    return out;
}

fn wrap_texel(v: i32, n: i32) -> i32 {
    return ((v % n) + n) % n;
}

@if(VOXEL_CONSERVATIVE) {
// `v`'s component along world axis `axis`.
fn along_axis(v: vec3<f32>, axis: u32) -> f32 {
    return select(select(v.z, v.y, axis == 1u), v.x, axis == 0u);
}
}

@fragment
fn fragment(in: VoxelVsOut) {
@if(VOXEL_CONSERVATIVE) {
    // The triangle's plane per pixel step, taken first, in uniform control flow. The plane is exact
    // under conservative rasterization: a pixel the triangle only grazes extrapolates it.
    let step_x = dpdx(in.world);
    let step_y = dpdy(in.world);
}
@if(GX_BINDLESS) {
    gx_bindless::enter(in.region);
}
    var albedo = vec3<f32>(0.5);
    if ((in.word & WORD_TEXTURED) != 0u) {
@if(GX_BINDLESS) {
        let texel = gx_bindless::sample_level(in.uv, in.word, vox.info.z);
} @else {
        let layer = i32(gx_rec(in.word).x);
        let texel = textureSampleLevel(tex_array, samp_repeat, in.uv, layer, vox.info.z);
}
@if(CUTOUT) {
        // A leaf card only occupies the voxels its opaque part covers — but at a lower key than the
        // draw lane's, so a canopy voxelizes as the canopy rather than as a handful of stems.
        if (texel.a < vox.info.y) {
            discard;
        }
}
        // The pool keeps the images' unorm formats, so the sample is the authored value the modern
        // lanes light in, and the probes match them.
        albedo = texel.rgb;
    }

@if(VOXEL_CONSERVATIVE) {
@if(TWO_SIDED) {
    let packed = vec4<f32>(0.5, 0.5, 0.5, 0.0);
} @else {
    let packed = vec4<f32>(normalize(in.normal) * 0.5 + vec3<f32>(0.5), 1.0);
}
    // Only the sweep down the plane's dominant axis writes: there the plane climbs at most one voxel
    // a pixel, where in another sweep a grazed pixel's extrapolated point can lie far off the
    // triangle. A tie (a 45-degree plane) writes in each tied sweep; a NaN plane writes nowhere.
    let axis = u32(vox.info.w);
    let face = abs(cross(step_x, step_y));
    let along = along_axis(face, axis);
    let across = max(max(face.x, face.y), face.z);
    if (!(along > 0.0) || !(along * 1.001 >= across)) {
        return;
    }
    // The voxels the plane crosses along the sweep over this pixel's square: three at most.
    let size = vox.box_lo.w;
    let depth = along_axis(in.world, axis);
    let reach = 0.5 * (abs(along_axis(step_x, axis)) + abs(along_axis(step_y, axis)));
    let first = i32(floor((depth - reach) / size));
    let last = min(i32(floor((depth + reach) / size)), first + 2);
    let n = i32(vox.box_hi.w);
    var w = vec3<i32>(floor(in.world / size));
    for (var k = first; k <= last; k += 1) {
        if (axis == 0u) {
            w.x = k;
        } else if (axis == 1u) {
            w.y = k;
        } else {
            w.z = k;
        }
        if (any(w < vec3<i32>(vox.box_lo.xyz)) || any(w >= vec3<i32>(vox.box_hi.xyz))) {
            continue;
        }
        let slot = vec3<i32>(
            wrap_texel(w.x, n),
            wrap_texel(w.y, n),
            wrap_texel(w.z, n) + i32(vox.info.x),
        );
        textureStore(vox_albedo, slot, vec4<f32>(albedo, 1.0));
        textureStore(vox_normal, slot, packed);
    }
} @else {
    let w = vec3<i32>(floor(in.world / vox.box_lo.w));
    if (any(w < vec3<i32>(vox.box_lo.xyz)) || any(w >= vec3<i32>(vox.box_hi.xyz))) {
        return;
    }
    let n = i32(vox.box_hi.w);
    let slot = vec3<i32>(
        wrap_texel(w.x, n),
        wrap_texel(w.y, n),
        wrap_texel(w.z, n) + i32(vox.info.x),
    );
    textureStore(vox_albedo, slot, vec4<f32>(albedo, 1.0));
@if(TWO_SIDED) {
    // Two-sided geometry (foliage, thin banners) has no meaningful facing: the march lights it from
    // whichever side the ray arrives on.
    textureStore(vox_normal, slot, vec4<f32>(0.5, 0.5, 0.5, 0.0));
} @else {
    let nrm = normalize(in.normal);
    textureStore(vox_normal, slot, vec4<f32>(nrm * 0.5 + vec3<f32>(0.5), 1.0));
}
}
}
