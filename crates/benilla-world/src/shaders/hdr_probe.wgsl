// fork: native HDR output — `WOW_HDR_PROBE=1`'s measurement (`hdr_probe.rs`): every pixel of the
// frame the window receives, in nits, folded into the peak and the counts above paper white.
// The frame is either scRGB (1.0 = 80 nits) or the gamma lane's paper-relative encode.

@group(0) @binding(0) var frame: texture_2d<f32>;

struct Params {
    paper_nits: f32,
    // 1: the gamma lane (the player-UI camera's target), 0: scRGB.
    gamma_lane: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

struct Totals {
    // The peak, in tenths of a nit.
    peak_decinits: atomic<u32>,
    above_paper: atomic<u32>,
    above_twice_paper: atomic<u32>,
    pixels: atomic<u32>,
}
@group(0) @binding(2) var<storage, read_write> totals: Totals;

var<workgroup> wg_peak: atomic<u32>;
var<workgroup> wg_above: atomic<u32>;
var<workgroup> wg_above2: atomic<u32>;
var<workgroup> wg_pixels: atomic<u32>;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let size = textureDimensions(frame);
    if (id.x < size.x && id.y < size.y) {
        let rgb = max(textureLoad(frame, vec2<i32>(id.xy), 0).rgb, vec3<f32>(0.0));
        var linear = rgb * (80.0 / params.paper_nits);
        if (params.gamma_lane != 0u) {
            linear = srgb_to_linear(rgb);
        }
        let nits = dot(linear, vec3<f32>(0.2126, 0.7152, 0.0722)) * params.paper_nits;
        atomicMax(&wg_peak, u32(clamp(nits * 10.0, 0.0, 4.0e9)));
        atomicAdd(&wg_pixels, 1u);
        if (nits > params.paper_nits * 1.01) {
            atomicAdd(&wg_above, 1u);
        }
        if (nits > params.paper_nits * 2.0) {
            atomicAdd(&wg_above2, 1u);
        }
    }
    workgroupBarrier();
    if (local == 0u) {
        atomicMax(&totals.peak_decinits, atomicLoad(&wg_peak));
        atomicAdd(&totals.above_paper, atomicLoad(&wg_above));
        atomicAdd(&totals.above_twice_paper, atomicLoad(&wg_above2));
        atomicAdd(&totals.pixels, atomicLoad(&wg_pixels));
    }
}
