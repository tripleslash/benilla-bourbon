// SSILVB — screen-space indirect lighting with a **visibility bitmask**.
//
// One gather produces both halves of the effect, which is the whole reason to use this method
// rather than bolting a GTAO pass and an SSGI pass together: while a slice is swept, each sample's
// angular span is OR-ed into a 32-bit mask of hemisphere sectors, so
//
//   · ambient occlusion  = 1 − popcount(mask) / 32           — how much of the hemisphere is shut
//   · indirect colour    = Σ radiance · popcount(NEWLY shut) — what shut it, and in what colour
//
// come out of the same loop, off the same horizon. The bitmask is what makes both terms honest: a
// wall standing behind another wall sets no bits its neighbour has not already set, so it neither
// over-occludes (GTAO's failure) nor bounces a second time, and a sample's *thickness* is expressed
// as a real angular span instead of being assumed infinite.
//
// **Its inputs are the frame's own depth and colour.** There is no prepass: the custom WoW
// materials cannot enter one (that is what broke the first shadow experiment — see
// benilla-app's player_shadow.rs) and the retained static_gx pass would not populate one at all,
// which would leave every Stormwind building missing from the AO. So this reads the MAIN depth
// buffer — which terrain, the opaque and alpha-masked model lanes and the retained cell pass all
// write into — and reconstructs view-space normals from it.
//
// **Gamma, and why the composite is a BLEND.** The pass runs before FFXGlow, so a classic frame's
// buffer holds GAMMA bytes (decision 0161; ffx_glow.wgsl owns the frame's single decode) and a
// modern frame's holds linear HDR radiance (`frame_is_linear`) — and it cannot use the ping-pong
// every other post pass uses, because it runs mid-graph: flipping the view target here
// would hand MainTransparentPass a texture it has not drawn into yet, and Bevy clears a colour
// attachment on its first use per texture, wiping the whole opaque frame.
//
// So the composite writes no colour of its own. It emits a per-pixel FACTOR and multiplies it into
// the main texture with a Dst/Zero blend, leaving the target and its load flags exactly where the
// opaque pass left them. In a classic frame the factor is raised to 1/2.2 first: a multiply
// belongs in linear light, and for a power-law transfer `encode(decode(g)·f) == g · f^(1/2.2)`, so
// pre-warping the factor is the same operation with no decode of the buffer at all. It is an
// approximation only against sRGB's linear toe, where it errs toward LESS darkening — the safe
// direction. A modern frame is linear already and takes the factor as it is.
//
// It also makes the off state exact: at both sliders zero the factor is `pow(1, 1/2.2) == 1` and
// `dst · 1.0` is bit-exact in float, so the frame is the frame a build without this pass renders.
//
// Everything reads with textureLoad: every texture here is the view's own size, so there is nothing
// to filter, and a texture_depth_2d cannot legally take a filtering sampler anyway.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::{View, position_ndc_to_view, uv_to_ndc}

@group(0) @binding(0) var<uniform> view: View;

// **Multisampling touches the depth binding and nothing else.** Bevy resolves the colour attachment
// at the end of the opaque pass, so `ViewTarget::main_texture_view()` is single-sampled on every
// path; the depth buffer has no such resolve, and there is no meaningful way to average one. So
// `SSILVB_MSAA` swaps the depth binding's type and leaves the rest of this file alone — and the
// single-sampled build pays nothing for the option existing.
//
// `textureLoad`'s third argument reads the same in both builds: the only mip on a plain depth
// texture, and the deliberately-chosen sample on a multisampled one. Sample 0 rather than an
// average, because a depth is a set of alternatives rather than a quantity to mix — averaging it
// across a silhouette invents a surface that is in neither place, which this pass would then
// occlude against.
#ifdef SSILVB_MSAA
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth_tex: texture_depth_2d;
#endif

// The pass input: the scene colour for the gather, and this pass's own ping-pong half for each blur
// and for the composite. Single-sampled throughout.
@group(0) @binding(2) var src_tex: texture_2d<f32>;
// strength: x = AO strength, y = indirect strength (both zero and the node never runs at all), z =
// the world sample radius in yards, w = the assumed occluder thickness in yards. frame: x = 1 when
// the buffer holds linear radiance (the modern lighting model), 0 when it holds gamma bytes.
struct SsilvbParams {
    strength: vec4<f32>,
    frame: vec4<f32>,
}
@group(0) @binding(3) var<uniform> params: SsilvbParams;

fn frame_is_linear() -> bool {
    return params.frame.x > 0.5;
}

const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;

/// The bitmask's width. 32 sectors over a half turn is ~5.6° of angular resolution, and it is not
/// a free parameter: it is u32, the widest lane countOneBits resolves in one instruction.
const SECTORS: u32 = 32u;

// `SSILVB_AO_ONLY` is the variant built while the indirect slider is zero: no bounce is gathered, so
// the emitter-colour path is compiled out and the sweep drops to the tap budget an occlusion-only
// term needs (the bitmask stays exact at low counts; the blur hides the extra noise).

/// Directions swept per pixel. Each slice is an independent horizon sweep, so this is the dominant
/// cost multiplier; 3, with a per-pixel rotation, is where the blur below can still hide what noise
/// is left. The AO-only variant takes 2.
#ifdef SSILVB_AO_ONLY
const SLICES: i32 = 2;
#else
const SLICES: i32 = 3;
#endif

/// Samples per side, per slice. Spaced quadratically, so the near field — where contact shadows
/// live and where the eye actually reads occlusion — takes most of them. The AO-only variant
/// takes 4.
#ifdef SSILVB_AO_ONLY
const STEPS: i32 = 4;
#else
const STEPS: i32 = 8;
#endif

/// The closest a sample may be taken, in pixels. Below about a texel the reconstructed normal and
/// the sample position come out of the same depth quad and the pixel shadows itself.
const MIN_STEP_PX: f32 = 1.5;

/// The furthest the sweep may reach, in pixels. A surface hard against the near plane projects an
/// unbounded world radius, and without this the loop degenerates into a full-screen gather on
/// exactly the frames (a wall in the face) where the budget is already worst.
const MAX_RADIUS_PX: f32 = 192.0;

/// The projected sweep radius, in pixels, below which the gather is fading out entirely, and the
/// width of that fade. Under about a dozen pixels there is nothing left to resolve: everything the
/// sweep can reach fits inside a couple of texels, so what comes back is depth quantisation rather
/// than geometry. Distant terrain is where that bites — a hillside most of a mile out, fogged to
/// very nearly the colour of the sky, otherwise comes back with its own triangle edges drawn onto
/// it in occlusion, which reads as a faint web hanging in the sky. Fading across a range rather
/// than switching off at a threshold keeps there from being a line in the world where AO stops.
const FADE_FULL_PX: f32 = 24.0;

/// How far past the sample radius, in world units, a sample may still be admitted as an occluder.
/// Beyond it the far background stops shutting sectors — the bitmask on its own would happily let
/// the horizon occlude a doorway.
const RANGE_SCALE: f32 = 1.5;

// A cheap acos — max error ~1e-3 rad over [-1, 1], and this runs up to SLICES · STEPS · 4 times per
// pixel. (Eberly's approximation, the one every screen-space AO reaches for.)
fn fast_acos(x: f32) -> f32 {
    let a = abs(x);
    let r = (-0.156583 * a + HALF_PI) * sqrt(max(0.0, 1.0 - a));
    return select(PI - r, r, x >= 0.0);
}

// The interleaved gradient noise every screen-space sampler rotates by. A function of the PIXEL
// alone — no frame counter: there is no temporal accumulator in this frame to average a per-frame
// jitter back out, and a $WOW_CAPTURE run has to be able to render the same frame twice.
fn gradient_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn depth_at(px: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, px, 0);
}

// The view-space position behind a pixel. Reverse-Z, so the caller must already have established
// that the depth is non-zero — at the infinite far plane this divides by zero.
fn view_pos_at(px: vec2<i32>, dims: vec2<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / dims;
    return position_ndc_to_view(vec3<f32>(uv_to_ndc(uv), depth_at(px)), view.view_from_clip);
}

// One axis of the depth gradient, pointing along the POSITIVE screen axis. Both neighbours are
// tried and the one nearer in depth wins: taking a fixed side fabricates a normal across every
// silhouette, which is what draws the dark outline around every object in a naive
// reconstruct-from-depth AO. A neighbour at the far plane (sky) is not a neighbour.
fn axis_delta(p: vec3<f32>, lo: vec2<i32>, hi: vec2<i32>, dims: vec2<f32>) -> vec3<f32> {
    let lo_ok = depth_at(lo) > 0.0;
    let hi_ok = depth_at(hi) > 0.0;
    if (!lo_ok && !hi_ok) {
        return vec3<f32>(0.0);
    }
    if (!lo_ok) {
        return view_pos_at(hi, dims) - p;
    }
    if (!hi_ok) {
        return p - view_pos_at(lo, dims);
    }
    let a = view_pos_at(lo, dims);
    let b = view_pos_at(hi, dims);
    return select(b - p, p - a, abs(a.z - p.z) < abs(b.z - p.z));
}

// The receiver normal, in view space. +u is view +x and +v is view −y, so the cross product is
// taken vertical-first to come out facing the camera.
fn view_normal_at(px: vec2<i32>, p: vec3<f32>, dims: vec2<f32>, bounds: vec2<i32>) -> vec3<f32> {
    let zero = vec2<i32>(0);
    let du = axis_delta(
        p,
        clamp(px - vec2<i32>(1, 0), zero, bounds),
        clamp(px + vec2<i32>(1, 0), zero, bounds),
        dims,
    );
    let dv = axis_delta(
        p,
        clamp(px - vec2<i32>(0, 1), zero, bounds),
        clamp(px + vec2<i32>(0, 1), zero, bounds),
        dims,
    );
    let n = cross(dv, du);
    if (dot(n, n) < 1e-12) {
        return normalize(-p); // a degenerate quad — face the camera and shut nothing
    }
    let unit = normalize(n);
    return select(unit, -unit, dot(unit, -p) < 0.0);
}

/// Spans thinner than this fraction of ONE sector are dropped instead of rounded up. A sample lying
/// in the receiver's own plane produces a span a few float epsilons wide, and `ceil` would promote
/// every one of them to a whole sector — open ground would sit at 31/32 visible and never reach 1.
const SPAN_EPS: f32 = 1.0 / 64.0;

// The bitmask update: set every sector between two normalised horizon angles. lo/hi are the
// sample's front and back faces mapped into [0, 1] across the hemisphere around the normal, so the
// span between them is exactly the solid angle this occluder shuts.
fn sector_bits(lo: f32, hi: f32) -> u32 {
    let span = i32(ceil((hi - lo) * f32(SECTORS) - SPAN_EPS));
    if (span <= 0) {
        return 0u;
    }
    let width = min(u32(span), SECTORS);
    let start = min(u32(lo * f32(SECTORS)), SECTORS - 1u);
    return (0xFFFFFFFFu >> (SECTORS - width)) << start;
}

/// The darkest a receiver is taken to be, in linear light, when the bounce is normalised by it. A
/// near-black texel would otherwise turn a faint bounce into an unbounded ratio — and since the
/// composite multiplies that ratio back into the same near-black texel, all it could ever buy is
/// noise from the blur carrying it onto a brighter neighbour.
const RECEIVER_FLOOR: f32 = 0.005;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// The scene colour at a pixel, in linear light. A modern frame is linear HDR already: decoding it a
// second time crushed every bounce toward black, and saturating it clipped the sun-lit emitters
// that should bleed the most.
fn scene_linear(px: vec2<i32>) -> vec3<f32> {
    let c = textureLoad(src_tex, px, 0).rgb;
    if (frame_is_linear()) {
        return max(c, vec3<f32>(0.0));
    }
    return srgb_to_linear(saturate(c));
}


// ───────────────────────────────────────────────────────────────────────────── the gather

@fragment
fn fs_gather(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(depth_tex);
    let dims = vec2<f32>(dims_u);
    let bounds = vec2<i32>(dims_u) - vec2<i32>(1);
    let px = vec2<i32>(in.position.xy);

    // Reverse-Z: zero IS the infinite far plane. The sky, and anything else that never wrote depth,
    // comes back fully visible with no bounce — which the composite reads as "leave this alone".
    if (depth_at(px) <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let p = view_pos_at(px, dims);
    let n = view_normal_at(px, p, dims, bounds);
    let v = normalize(-p);

    let radius_world = params.strength.z;
    let thickness = params.strength.w;
    let range = radius_world * RANGE_SCALE;
    let range_sq = range * range;

    // The world radius, projected. clip_from_view[1][1] is 1/tan(fovy/2), so a world offset r at
    // view depth −p.z spans 0.5 · f · r / −p.z of NDC — that fraction of the viewport HEIGHT.
    // Marching in pixels off the height alone is what keeps the sweep circular on any aspect.
    let radius_px_raw = 0.5 * view.clip_from_view[1][1] * radius_world / max(-p.z, 1e-3) * dims.y;
    // Smoothstep, not a ramp: the projected radius falls as 1/z, so a linear ramp in pixels is a
    // very long tail in yards and leaves the far field at a third strength for most of the draw
    // distance — which is exactly where the artefact lives.
    let fade = smoothstep(MIN_STEP_PX * 2.0, FADE_FULL_PX, radius_px_raw);
    if (fade <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // too far to resolve anything: fully visible
    }
    let radius_px = clamp(radius_px_raw, MIN_STEP_PX * 2.0, MAX_RADIUS_PX);

    let noise = gradient_noise(in.position.xy);
    // A second, decorrelated jitter for the step offset — the golden-ratio turn of the first, the
    // cheapest way to get two independent-looking sequences out of one hash.
    let noise_step = fract(noise + 0.6180339887);

#ifndef SSILVB_AO_ONLY
    let want_gi = params.strength.y > 0.0;
#endif
    var visibility = 0.0;
    var radiance = vec3<f32>(0.0);

    for (var slice = 0; slice < SLICES; slice += 1) {
        let phi = (f32(slice) + noise) * PI / f32(SLICES);
        let omega = vec2<f32>(cos(phi), sin(phi));

        // The slice plane contains the view vector and this screen direction — and the screen
        // direction has to be carried into VIEW space to name that plane. `omega` steps in
        // framebuffer pixels, whose +y runs DOWN the screen, while view +y runs up; the two axes
        // are otherwise the same scale (one pixel is `2 / (dims.y · clip_from_view[1][1])` of view
        // space along both), so the whole conversion is that one sign. Get it wrong and the plane
        // the normal is measured in is the MIRROR of the plane the samples are marched in, which
        // is not a subtle error: measured against a synthetic flat floor, it cost open ground a
        // third of its visibility before anything had occluded it.
        let dir = vec3<f32>(omega.x, -omega.y, 0.0);
        let axis = normalize(cross(dir, v));
        // The in-slice tangent — perpendicular to the normal, lying in the slice plane. Only its
        // SIGN is ever read (which side of the surface's zenith a sample fell on), so it is never
        // normalised.
        let tangent = cross(axis, n);
        if (dot(tangent, tangent) < 1e-10) {
            // The normal is perpendicular to this slice, so the slice sees the surface edge-on and
            // has no hemisphere to shut. Count it fully visible rather than let an arbitrary sign
            // place the samples.
            visibility += 1.0;
            continue;
        }

        var mask: u32 = 0u;
        for (var side = 0; side < 2; side += 1) {
            let s = select(1.0, -1.0, side == 1);
            for (var tap = 0; tap < STEPS; tap += 1) {
                let t = (f32(tap) + noise_step) / f32(STEPS);
                let reach = max(MIN_STEP_PX, radius_px * t * t);
                let spx = vec2<i32>(round(in.position.xy + s * omega * reach));
                if (spx.x < 0 || spx.y < 0 || spx.x > bounds.x || spx.y > bounds.y) {
                    continue;
                }
                if (depth_at(spx) <= 0.0) {
                    continue; // the sky shuts nothing
                }
                let sp = view_pos_at(spx, dims);
                let delta = sp - p;
                let dist_sq = dot(delta, delta);
                if (dist_sq < 1e-8 || dist_sq > range_sq) {
                    continue;
                }
                let to_sample = delta * inverseSqrt(dist_sq);

                // The sample's front face, and — pushed one thickness away from the camera — its
                // back. The span between them is what it actually shuts; with no back face every
                // occluder is a solid wall running to infinity.
                let back = normalize(delta - v * thickness);
                // Where the pair sits in the hemisphere ABOVE THE SURFACE: 0.5 is the zenith
                // (straight up the normal), 0 and 1 are the two horizons. The angle is taken from
                // the NORMAL rather than from the view vector, which is what makes a flat surface
                // exact: `dot(to_sample, n)` is identically zero for any sample lying in the
                // receiver's own plane, whatever pixel the march rounded to, so it lands on the
                // horizon dead on and shuts nothing. Routed through the view vector the same sample
                // misses its horizon by that rounding error, and `sector_bits` rounds every miss up
                // into a whole sector — which is a floor under the occlusion of every flat surface
                // in the frame. The side comes from the tangent, not from `s`, so a sample is
                // placed by where it actually fell rather than by which way the march was walking.
                let side_sign = select(-1.0, 1.0, dot(to_sample, tangent) >= 0.0);
                var horizon = vec2<f32>(
                    fast_acos(clamp(dot(to_sample, n), -1.0, 1.0)),
                    fast_acos(clamp(dot(back, n), -1.0, 1.0)),
                );
                horizon = saturate(0.5 + side_sign * horizon / PI);
                let bits = sector_bits(min(horizon.x, horizon.y), max(horizon.x, horizon.y));
                let fresh = bits & ~mask;
                mask |= bits;

                // Only what this sample shut FIRST may light us. That is what stops a row of trees
                // from each bouncing the same patch of lit grass into the same pixel.
                // And only a surface the receiver is not BEHIND. Under the thickness model the
                // sample is a slab one thickness deep; a receiver further from the camera than its
                // back face sees only that back, which the colour buffer never holds. Bouncing the
                // lit front face instead draws every object's colour as a halo onto what stands
                // behind it. It still shuts its sectors: the occlusion is real, the colour is not.
#ifndef SSILVB_AO_ONLY
                let behind_slab = sp.z - p.z > thickness;
                if (want_gi && fresh != 0u && !behind_slab) {
                    let n_dot_l = saturate(dot(n, to_sample));
                    if (n_dot_l > 0.0) {
                        // The emitter's own cosine is a ONE-SIDED TEST, not a weight. What the
                        // colour buffer holds at that pixel is the surface's outgoing radiance,
                        // and a Lambertian emitter's radiance is the same in every direction above
                        // it — its cosine is already spent in arriving at that value. Multiplying
                        // by it a second time only dims every bounce by the angle it happens to be
                        // seen at, which is what left the whole indirect half at a few levels out
                        // of 255. What the test IS for is the back of a surface: a wall whose lit
                        // face points away from this pixel must not light it through itself.
                        // The receiver's silhouette-aware normal: forward differences would cross
                        // an object's right and bottom silhouette and fabricate a sideways normal
                        // that passes this test.
                        let en = view_normal_at(spx, sp, dims, bounds);
                        if (dot(en, -to_sample) > 0.0) {
                            let c = scene_linear(spx);
                            let share = f32(countOneBits(fresh)) / f32(SECTORS);
                            radiance += c * (share * n_dot_l);
                        }
                    }
                }
#endif
            }
        }
        visibility += 1.0 - f32(countOneBits(mask)) / f32(SECTORS);
    }

    // The bounce leaves here as a RATIO to the receiver's own brightness, because the composite can
    // only multiply. What a bounce adds is `albedo · E`, and with no albedo buffer the receiver's
    // chroma at its brightest channel stands in for one: `albedo ≈ scene / max3(scene)`. Divided
    // back through the scene the composite multiplies, that is `E / max3(scene)` per channel. Left
    // un-normalised, the term rode on the receiver's LIT colour and came out `scene · E` — the
    // scene's exposure counted twice, a few percent of a few percent, invisible at any slider.
    let recv = scene_linear(px);
    let recv_peak = max(max(recv.r, max(recv.g, recv.b)), RECEIVER_FLOOR);
    return vec4<f32>(
        radiance / f32(SLICES) * fade / recv_peak,
        mix(1.0, visibility / f32(SLICES), fade),
    );
}

// ──────────────────────────────────────────────────────────────────────────── the denoise

// A separable depth-aware 5-tap. The gather is noisy by construction — three rotated slices per
// pixel is a stochastic estimate of an integral — and this frame has no temporal accumulator to
// average it out, so the whole denoise budget is spatial. Weights fall off with the LINEAR depth
// difference, so occlusion never crosses a silhouette; a plain box blur turns that into a halo.
fn blur(px: vec2<i32>, axis: vec2<i32>, bounds: vec2<i32>, dims: vec2<f32>) -> vec4<f32> {
    if (depth_at(px) <= 0.0) {
        return textureLoad(src_tex, px, 0);
    }
    let z0 = view_pos_at(px, dims).z;
    // A depth tolerance PROPORTIONAL to distance: 20 cm is a step on a doorframe two yards away and
    // is inside one texel of a hillside eighty yards out.
    let tolerance = max(0.03 * abs(z0), 0.02);
    var sum = vec4<f32>(0.0);
    var weight = 0.0;
    for (var i = -2; i <= 2; i += 1) {
        let q = clamp(px + axis * i, vec2<i32>(0), bounds);
        if (depth_at(q) <= 0.0) {
            continue;
        }
        let w = exp(-0.5 * f32(i * i)) * exp(-abs(view_pos_at(q, dims).z - z0) / tolerance);
        sum += textureLoad(src_tex, q, 0) * w;
        weight += w;
    }
    if (weight <= 0.0) {
        return textureLoad(src_tex, px, 0);
    }
    return sum / weight;
}

@fragment
fn fs_blur_h(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(depth_tex);
    return blur(
        vec2<i32>(in.position.xy),
        vec2<i32>(1, 0),
        vec2<i32>(dims_u) - vec2<i32>(1),
        vec2<f32>(dims_u),
    );
}

@fragment
fn fs_blur_v(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims_u = textureDimensions(depth_tex);
    return blur(
        vec2<i32>(in.position.xy),
        vec2<i32>(0, 1),
        vec2<i32>(dims_u) - vec2<i32>(1),
        vec2<f32>(dims_u),
    );
}

// ────────────────────────────────────────────────────────────────────────── the composite

// The per-pixel factor the blend multiplies into the main texture:
//
//   linear_out = linear_scene · occlusion  +  albedo · E_indirect
//              = linear_scene · (occlusion  +  E_indirect / max3(linear_scene))
//
// with the albedo stood in for by the receiver's chroma at full brightness — this renderer never
// separates one out. The gather has already divided the bounce through, so what arrives here is the
// bracket's second term and the whole composite stays a single multiply.

/// What one unit of the AO slider is worth. Both gains exist so that the sliders' MIDPOINT is the
/// intended look: the row reads 0–1 and ships at 0.5, so 0.5 has to land on the strength the world
/// is meant to be seen at, leaving the top half as headroom for someone who wants it heavier. With
/// the term applied raw, the midpoint was half of an already-subtle effect and read as nothing.
///
/// A gain of 2 makes slider 0.5 exactly the gather's own visibility, applied whole — the honest,
/// unexaggerated term — and slider 1.0 twice that, which crushes creases on purpose. `saturate`
/// downstream keeps the over-driven half in range.
const AO_GAIN: f32 = 2.0;

/// What one unit of the indirect slider is worth. See [AO_GAIN] for why the gains exist at all.
/// The gather's bounce, enclosed on every side by a uniform radiance L, sums to 2/π · L (the slice
/// average of the receiver's cosine), and the stand-in albedo is a full-brightness one where a real
/// surface reflects about half — so the honest term is ~0.8 × the gathered ratio. Honest is
/// invisible: a screen-space gather sees only the near field, and a hut's legs over a sunlit bank
/// moved by a level or two. Measured on `water-noon`, 4 (the midpoint) warms the shade under the
/// thatch and 8 (the top of the row) turns the legs and reeds plainly the bank's orange. This is
/// the ONLY place the indirect half is scaled.
const GI_GAIN: f32 = 8.0;

/// The ceiling on the composite factor. Occlusion alone can never exceed one; only the bounce can
/// push past it — at the top of the row, a dark crevice beside a sunlit wall gets near this. It
/// bounds the damage a degenerate gather could do.
const MAX_GAIN: f32 = 4.0;

@fragment
fn fs_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let indirect = textureLoad(src_tex, vec2<i32>(in.position.xy), 0);
    // At `strength.x == 0` the mix factor is 0 and this is exactly 1.0, so the off state stays a
    // bit-exact no-op no matter what the gain is.
    let occlusion = saturate(mix(1.0, saturate(indirect.a), params.strength.x * AO_GAIN));
    let bounce = max(indirect.rgb, vec3<f32>(0.0)) * (params.strength.y * GI_GAIN);
    // The sum must NOT be saturated: a bounce is the one thing here that legitimately pushes a
    // factor above one, and clamping it at one meant the whole colour-bleed half did nothing
    // wherever a surface was unoccluded — which is most of the frame. The target is float, so a
    // factor over one is representable; the cap is only there so a pathological gather cannot blow
    // the frame out.
    //
    let linear_factor = clamp(vec3<f32>(occlusion) + bounce, vec3<f32>(0.0), vec3<f32>(MAX_GAIN));
    // A modern frame is linear radiance and takes the factor as it is. A classic frame holds gamma
    // bytes, and ^(1/2.2) turns the linear-light factor into the gamma-space one with the same
    // effect. `pow` of exactly 1.0 is exactly 1.0, so an untouched pixel stays untouched either way.
    let factor = select(
        pow(linear_factor, vec3<f32>(1.0 / 2.2)),
        linear_factor,
        frame_is_linear(),
    );
    // Alpha is pinned by the blend state (Zero/One), not by what is returned here.
    return vec4<f32>(factor, 1.0);
}
