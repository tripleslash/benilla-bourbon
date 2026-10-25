// Modern lighting (FORK.md, P2a): the retained static world — ADT doodad cells,
// WMO groups, interior props — drawn depth-only into the modern sun's cascade shadow views.
// Group 1 is static_gx's own cell layout, bound unchanged; see `static_gx.wgsl` for its contract.
// Group 0: this cascade's light clip (`static_gx::render::modern_shadow::CasterView`).
struct CasterView {
    clip_from_world: mat4x4<f32>,
    // fork: a lamp cube face's fixture sphere (xyz its light, w the radius; 0 = none). Read only by
    // `static_gx_shadow.wgsl`'s `FIXTURE_CLIP` pipelines.
    clear: vec4<f32>,
}
@group(0) @binding(0) var<uniform> caster: CasterView;

struct GxCell {
    origin: vec4<f32>, // xyz = the bake's recentring origin: world = vertex + origin
}
@group(1) @binding(0) var<uniform> cell: GxCell;
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;

const WORD_TEXTURED: u32 = 2097152u; // 1 << 21, as static_gx.wgsl
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
}

struct ShadowVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) word: u32,
#ifdef FIXTURE_CLIP
    // fork: xyz = the vertex's offset from the lamp (interpolates exactly), w = the fixture radius.
    @location(2) from_light: vec4<f32>,
#endif
}

@vertex
fn vertex(v: GxVertex) -> ShadowVsOut {
    var out: ShadowVsOut;
    out.uv = v.uv;
    out.word = v.word;
#ifdef FIXTURE_CLIP
    out.from_light = vec4<f32>(v.position + cell.origin.xyz - caster.clear.xyz, caster.clear.w);
#endif
    // A killed item (record bit 0) collapses outside the clip volume, as in the main pass.
    if ((recs[v.word & 0xffffu].w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let world = v.position + cell.origin.xyz;
    out.position = caster.clip_from_world * vec4<f32>(world, 1.0);
    // Pancake casters that sit between the light and the cascade's near plane onto it, so they
    // still cast (Bevy's UNCLIPPED_DEPTH_ORTHO_EMULATION, vertex half).
    out.position.z = min(out.position.z, 1.0);
    return out;
}

// Only the cutout and lamp-atlas pipelines have a fragment stage: alpha-keyed leaves and railings
// cast their cut-out shape, and a lamp's own fixture (`ShadowLight::clear`, a sphere about the
// light) casts nothing into its cube. A sphere and not the cube's near plane: that clipped a body's
// legs and the ground beside a lamp too.
@fragment
fn fragment(in: ShadowVsOut) {
#ifdef FIXTURE_CLIP
    if (length(in.from_light.xyz) < in.from_light.w) {
        discard;
    }
#endif
#ifdef CUTOUT
    if ((in.word & WORD_TEXTURED) == 0u) {
        return;
    }
    let layer = i32(recs[in.word & 0xffffu].x);
    let a = textureSampleLevel(tex_array, samp_repeat, in.uv, layer, 1.0).a;
    if (a < VANILLA_ALPHA_KEY) {
        discard;
    }
#endif
}
