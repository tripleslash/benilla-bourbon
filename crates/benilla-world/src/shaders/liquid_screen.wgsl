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
// ## Multisampling
//
// Only the DEPTH binding changes. Bevy resolves the colour attachment at the end of the opaque
// pass — `ViewTarget::main_texture_view()` is the resolved, single-sampled image on every path — so
// the scene copy reads the same texture whatever `gxMultisample` says, and there is no resolve for
// this file to add. The depth buffer has no such resolve (there is no meaningful way to average
// one), so it stays multisampled and `LIQUID_SCREEN_MSAA` gives it the binding type and the fetch
// that matches. `0` as the third argument to `textureLoad` is the only mip on the plain build and
// the deliberately-chosen sample on the multisampled one: a depth is a set of alternatives, not a
// quantity to mix, and averaging across a silhouette invents a surface that is in neither place —
// which lands in the water's refraction guard as a halo.
//
// So the non-MSAA path pays nothing (same bindings, same fetches) and the MSAA path pays one
// binding type, no extra pass and no extra texture.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::{View, position_ndc_to_view, uv_to_ndc}

@group(0) @binding(0) var<uniform> view: View;

#ifdef LIQUID_SCREEN_MSAA
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth_tex: texture_depth_2d;
#endif

/// The view's colour, already resolved by the opaque pass — single-sampled in every build.
@group(0) @binding(2) var src_tex: texture_2d<f32>;

@group(0) @binding(3) var src_samp: sampler;

/// The blur's source: the copy this file just made, one pass earlier. A separate binding from
/// `src_tex` so all three passes can share one bind group — the blur wants the copy, the other two
/// want the view.
@group(0) @binding(4) var blur_src: texture_2d<f32>;

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
/// Addressed by UV, not by pixel: the target is allocated against the *window*, while the source is
/// the camera's viewport, and render scale (`world_backdrop::RenderScale`) is free to put a factor
/// between them. Addressing by UV is correct at any ratio; addressing by pixel would silently crop
/// or tile the frame the moment a player moved that slider.
@fragment
fn fs_copy(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(src_tex, src_samp, in.uv, 0.0);
}

/// **Pass 2 — the blurred copy.** The reference's `FourTapBlur`, run once while downsampling: four
/// bilinear taps, which on a 4x reduction covers the source footprint in one pass instead of a
/// separable pair.
///
/// This is the *reflection* source. A screen-space reflection off a bumped normal lands somewhere
/// slightly wrong by construction, and blurring is what turns "wrong" into "shimmering" — the
/// reference blurs for the same reason, and it is why the quarter-resolution target is a quality
/// choice rather than only a cost one.
///
/// Reads `blur_src` (the copy above), so multisampling never reaches it.
@fragment
fn fs_blur(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Taps three source texels out, not one: the target is a quarter of the source's linear size,
    // so a one-texel kernel would leave three quarters of the footprint unsampled and the
    // "blurred" copy would be an aliased downsample wearing the name.
    let texel = 1.0 / vec2<f32>(textureDimensions(blur_src));
    let o = texel * 3.0;
    var sum = textureSample(blur_src, src_samp, in.uv + vec2<f32>(-o.x, -o.y));
    sum += textureSample(blur_src, src_samp, in.uv + vec2<f32>(o.x, -o.y));
    sum += textureSample(blur_src, src_samp, in.uv + vec2<f32>(-o.x, o.y));
    sum += textureSample(blur_src, src_samp, in.uv + vec2<f32>(o.x, o.y));
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
    // the two can differ. Nearest by construction: a depth texture takes no filtering sampler, and
    // a multisampled one takes no sampler at all.
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let px = vec2<i32>(clamp(in.uv * dims, vec2<f32>(0.0), dims - 1.0));
    // One spelling covers both builds: the third argument is a mip level on `texture_depth_2d` and
    // a SAMPLE INDEX on `texture_depth_multisampled_2d`, and zero is the answer either way — the
    // only mip there is, and the sample this pass deliberately picks.
    let d = textureLoad(depth_tex, px, 0);
    if (d <= 0.0) {
        return vec4<f32>(NO_DEPTH, 0.0, 0.0, 1.0);
    }
    let p = position_ndc_to_view(vec3<f32>(uv_to_ndc(in.uv), d), view.view_from_clip);
    return vec4<f32>(-p.z, 0.0, 0.0, 1.0);
}
