// FSR 3.1.5 frame generation — one level of an inpainting pyramid.
// FidelityFX SDK 3.1.5 (MIT), via physics-rs's WGSL port; see `fsr/LICENSE-FIDELITYFX.txt`.
//
// Stands in for the SDK's single-pass downsampler. SPD's whole reason to exist is
// its cross-workgroup atomic counter, which lets one dispatch reach mip levels
// below what a single tile can cover; both pyramids here are read all the way to
// the top, which SPD would need that counter — and a `globallycoherent` buffer —
// to produce. WebGPU has neither the coherence guarantee nor a portable way to
// elect a last workgroup, so the chain is built one level per dispatch instead.
// Each level is a quarter of the one before, so the whole chain costs about a
// third more than its base level and the extra dispatches are noise.
//
// The source is bound as a single-mip view rather than the whole pyramid: WebGPU
// forbids a texture being readable and writable through overlapping views in the
// same bind group, and level `n-1` and level `n` are distinct subresources.
//
// `FG_PYRAMID_REDUCE_MV` is resolved in `fsr/framegen.rs` and selects which of the
// two weighted means the pyramid being built calls for: `true` or `false`.

@group(0) @binding(1) var mip_in: texture_2d<f32>;
@group(0) @binding(2) var mip_out: texture_storage_2d<rgba16float, write>;

/// Which pyramid this pipeline builds: the motion vector field's, or the
/// interpolated image's.
const REDUCE_MV: bool = FG_PYRAMID_REDUCE_MV;

fn load_source(p: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
    if (!is_on_screen(p, size)) {
        return vec4<f32>(0.0);
    }
    return textureLoad(mip_in, p, 0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = vec2<i32>(textureDimensions(mip_out));
    let p = vec2<i32>(gid.xy);
    if (any(p >= dst_size)) {
        return;
    }

    let src_size = vec2<i32>(textureDimensions(mip_in));
    let base = p * 2;
    let v0 = load_source(base, src_size);
    let v1 = load_source(base + vec2<i32>(1, 0), src_size);
    let v2 = load_source(base + vec2<i32>(0, 1), src_size);
    let v3 = load_source(base + vec2<i32>(1, 1), src_size);

    var reduced: vec4<f32>;
    if (REDUCE_MV) {
        reduced = reduce4_mv(v0, v1, v2, v3);
    } else {
        reduced = reduce4_color(v0, v1, v2, v3);
    }

    textureStore(mip_out, p, reduced);
}
