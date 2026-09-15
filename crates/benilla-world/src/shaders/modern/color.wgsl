// Modern lighting: colour-space helpers shared by every modern lane (FORK.md P1).
//
// The classic lanes write gamma-encoded values and FfxGlow decodes once at the end of the frame.
// Modern lanes decode their INPUTS instead — albedo texels, vertex colours, packed DBC bytes — and
// write linear radiance. The transfer here is the exact piecewise sRGB curve, not a 2.2 power:
// the parity check against classic (relight R^2) is sensitive to the toe.
#define_import_path benilla_modern::color

fn srgb_to_linear_channel(c: f32) -> f32 {
    let x = max(c, 0.0);
    return select(pow((x + 0.055) / 1.055, 2.4), x / 12.92, x <= 0.04045);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_channel(c.r),
        srgb_to_linear_channel(c.g),
        srgb_to_linear_channel(c.b),
    );
}

fn linear_to_srgb_channel(c: f32) -> f32 {
    let x = max(c, 0.0);
    return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, x <= 0.0031308);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_channel(c.r),
        linear_to_srgb_channel(c.g),
        linear_to_srgb_channel(c.b),
    );
}

// Rec. 709 relative luminance of a linear colour.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
