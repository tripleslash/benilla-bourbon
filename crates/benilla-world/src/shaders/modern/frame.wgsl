// Modern lighting: the per-frame state every modern lane reads (FORK.md).
//
// Only the TYPE lives here. Where it is bound differs per lane (material group 3 at binding 120,
// static_gx's own group, the effect lane's group), so each entry shader declares the binding with
// its own group index — the struct is the shared contract, and its Rust mirror lands with the
// first consumer (P1 `modern/radiance.rs`), whose layout test pins the two together.
#define_import_path benilla_modern::frame

struct ModernFrame {
    // xyz: direction the key light TRAVELS (to-light = -xyz); w: key weight (sun/moon crossfade).
    key_dir: vec4<f32>,
    // rgb: key radiance, linear; w: pre-exposure applied in-shader to stay inside f16 range.
    key_radiance: vec4<f32>,
    // Sky irradiance, order-2 SH: coefficient i in rgb, w unused.
    sky_sh: array<vec4<f32>, 9>,
    // rgb: linear fog/media albedo tint; w: lighting quality tier (0..3).
    fog_tint: vec4<f32>,
}
