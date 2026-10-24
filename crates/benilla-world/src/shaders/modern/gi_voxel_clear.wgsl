// Modern lighting (FORK.md, P5): empty one box of the GI voxel clipmap before it
// is rasterized again — a slab the toroidal grid just scrolled in, or the box around content that
// streamed out. Toroidal, so the box's texels are wrapped exactly as the raster wraps them.

struct VoxelView {
    clip_from_world: mat4x4<f32>,
    box_lo: vec4<f32>,
    box_hi: vec4<f32>,
    info: vec4<f32>,
}
@group(0) @binding(0) var<uniform> vox: VoxelView;
@group(1) @binding(0) var vox_albedo: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(1) var vox_normal: texture_storage_3d<rgba8unorm, write>;

fn wrap_texel(v: i32, n: i32) -> i32 {
    return ((v % n) + n) % n;
}

@compute @workgroup_size(4, 4, 4)
fn clear_box(@builtin(global_invocation_id) id: vec3<u32>) {
    let lo = vec3<i32>(vox.box_lo.xyz);
    let hi = vec3<i32>(vox.box_hi.xyz);
    let w = lo + vec3<i32>(id);
    if (any(w >= hi)) {
        return;
    }
    let n = i32(vox.box_hi.w);
    let slot = vec3<i32>(
        wrap_texel(w.x, n),
        wrap_texel(w.y, n),
        wrap_texel(w.z, n) + i32(vox.info.x),
    );
    textureStore(vox_albedo, slot, vec4<f32>(0.0));
    textureStore(vox_normal, slot, vec4<f32>(0.0));
}
