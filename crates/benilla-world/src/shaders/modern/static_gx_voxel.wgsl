// Modern lighting (FORK.md, P5): the retained static world rasterized into the GI
// voxel clipmap — the fork's stand-in for the plan's per-model proxy bake.
//
// The pass draws the same cells, WMO groups and props the modern shadow pass draws, but through an
// orthographic box camera whose fragments write the voxel they land in (albedo and the surface
// normal) instead of a colour. It runs once per axis, so a triangle edge-on to one sweep is caught
// by another; the raster is supersampled against the voxel grid, so a wall thinner than a voxel
// still fills one.
//
// Group 1 is static_gx's own cell layout, bound unchanged (see `static_gx.wgsl`).
#import benilla_modern::colorspace

struct VoxelView {
    clip_from_world: mat4x4<f32>,
    // xyz = the dirty box's low voxel coord, w = this level's voxel size (yd).
    box_lo: vec4<f32>,
    // xyz = the box's high voxel coord (exclusive), w = voxels per side.
    box_hi: vec4<f32>,
    // x = the level's slice offset in the texture, y = the alpha key, z = the albedo mip.
    info: vec4<f32>,
}
@group(0) @binding(0) var<uniform> vox: VoxelView;

struct GxCell {
    origin: vec4<f32>,
}
@group(1) @binding(0) var<uniform> cell: GxCell;
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;

@group(2) @binding(0) var vox_albedo: texture_storage_3d<rgba8unorm, write>;
@group(2) @binding(1) var vox_normal: texture_storage_3d<rgba8unorm, write>;

const WORD_TEXTURED: u32 = 2097152u;

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
}

struct VoxelVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) word: u32,
}

@vertex
fn vertex(v: GxVertex) -> VoxelVsOut {
    var out: VoxelVsOut;
    // A killed item (record bit 0) collapses outside the clip volume, as in every other lane.
    if ((recs[v.word & 0xffffu].w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let world = v.position + cell.origin.xyz;
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

@fragment
fn fragment(in: VoxelVsOut) {
    var albedo = vec3<f32>(0.5);
    if ((in.word & WORD_TEXTURED) != 0u) {
        let layer = i32(recs[in.word & 0xffffu].x);
        let texel = textureSampleLevel(tex_array, samp_repeat, in.uv, layer, vox.info.z);
#ifdef CUTOUT
        // A leaf card only occupies the voxels its opaque part covers — but at a lower key than the
        // draw lane's, so a canopy voxelizes as the canopy rather than as a handful of stems.
        if (texel.a < vox.info.y) {
            discard;
        }
#endif
        albedo = texel.rgb;
    }
    // The pool's arrays are sRGB views, so the sample arrives linear; the modern lanes light in the
    // authored value space, and the probes have to match them.
    albedo = colorspace::linear_to_srgb(albedo);

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
#ifdef TWO_SIDED
    // Two-sided geometry (foliage, thin banners) has no meaningful facing: the march lights it from
    // whichever side the ray arrives on.
    textureStore(vox_normal, slot, vec4<f32>(0.5, 0.5, 0.5, 0.0));
#else
    let nrm = normalize(in.normal);
    textureStore(vox_normal, slot, vec4<f32>(nrm * 0.5 + vec3<f32>(0.5), 1.0));
#endif
}
