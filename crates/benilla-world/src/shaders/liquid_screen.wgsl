// **The cata liquid path's screen-space inputs** — the three textures `liquid_cata.wgsl` reads that
// the opaque world has to be finished before anything can produce.
//
// The reference calls these `"LiquidProc BackBufferTex"`, `"blurredBack"` and the depth
// shader-resource view (`DepthCache::GetDepthSR`), bound on stages 0, 3 and 6 of
// `ProcWaterAbove`/`ProcWaterBelow` (`Liquid::CMaterialWater::SetupRender`). This file is the three
// passes that fill them, and it is deliberately dumb: a blit, a four-tap box, and a depth
// linearisation. Everything interesting happens in the water shader that reads them.
//
// **Why a copy at all, rather than reading the view target in place.** The water draws in the
// transparent pass, into the same attachment. A texture cannot be bound for sampling and as a
// colour attachment in the same pass, so the refraction source has to be a snapshot taken before
// the pass begins. That snapshot is also *why* water refracts only what was already opaque —
// the same limitation the reference has, and the reason its own water never refracts other water.
//
// Everything reads with `textureLoad` where the source is the view's own size, and with a sampler
// only where the blur genuinely resamples across a scale change.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::{View, position_ndc_to_view, uv_to_ndc}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var src_samp: sampler;

/// What a fragment that never wrote depth reports as its view distance.
///
/// Reverse-Z puts the infinite far plane at zero, so the sky, and anything else the opaque pass
/// left untouched, has no position to reconstruct. Handing the water a very large number is the
/// answer that makes its guards behave: the sky is never "in front of" a water surface, so the
/// refraction offset is never rejected against it and the depth-tint never saturates on it.
const NO_DEPTH: f32 = 1.0e9;

/// **Pass 1 — the scene colour snapshot.** A copy of the opaque frame; the refraction reads it with
/// a bump-driven offset, so it wants the sharpness and takes the viewport's own resolution.
///
/// Sampled by UV rather than loaded by pixel: the target is allocated against the *window*, while
/// the source is the camera's viewport, and render scale (`world_backdrop::RenderScale`) is free to
/// put a factor between them. Addressing by UV is correct at any ratio; addressing by pixel would
/// silently crop or tile the frame the moment a player moved that slider.
@fragment
fn fs_copy(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(src_tex, src_samp, in.uv, 0.0);
}

/// **Pass 2 — the blurred copy.** The reference's `FourTapBlur`, run once while downsampling: four
/// bilinear taps on a half-texel diagonal, which on a 4× reduction covers the source footprint
/// exactly and costs one pass instead of a separable pair.
///
/// This is the *reflection* source. A screen-space reflection off a bumped normal lands somewhere
/// slightly wrong by construction, and blurring is what turns "wrong" into "shimmering" — the
/// reference blurs for the same reason, and it is why the quarter-resolution target below is a
/// quality choice rather than only a cost one.
@fragment
fn fs_blur(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Taps three source texels out, not one: the target is a quarter of the source's linear size,
    // so a one-texel kernel would leave three quarters of the footprint unsampled and the
    // "blurred" copy would be an aliased downsample wearing the name.
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let o = texel * 3.0;
    var sum = textureSample(src_tex, src_samp, in.uv + vec2<f32>(-o.x, -o.y));
    sum += textureSample(src_tex, src_samp, in.uv + vec2<f32>(o.x, -o.y));
    sum += textureSample(src_tex, src_samp, in.uv + vec2<f32>(-o.x, o.y));
    sum += textureSample(src_tex, src_samp, in.uv + vec2<f32>(o.x, o.y));
    return sum * 0.25;
}

/// **Pass 3 — linear view depth.** The main depth buffer, reconstructed into yards along the view
/// axis and written to `R32Float`.
///
/// The reference binds its raw depth surface and carries a scale in `c22.y` to undo whatever the
/// hardware format meant. Resolving that here instead collapses the scale to 1.0 and leaves the
/// water shader comparing two quantities that are already in the same space as its own fragment —
/// which is the only reason its refraction guard is a plain subtract.
@fragment
fn fs_linear_depth(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Pixel coordinates in the DEPTH texture's own grid, not this pass's — see `fs_copy` for why
    // the two can differ. A depth texture takes no filtering sampler, so this is a nearest fetch
    // by construction, which is what a depth comparison wants anyway.
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let px = vec2<i32>(clamp(in.uv * dims, vec2<f32>(0.0), dims - 1.0));
    let d = textureLoad(depth_tex, px, 0);
    if (d <= 0.0) {
        return vec4<f32>(NO_DEPTH, 0.0, 0.0, 1.0);
    }
    let p = position_ndc_to_view(vec3<f32>(uv_to_ndc(in.uv), d), view.view_from_clip);
    return vec4<f32>(-p.z, 0.0, 0.0, 1.0);
}
