// FSR motion vectors for moving model parts (`fsr/velocity.rs`): each part drawn a second time into
// the velocity target, depth-tested read-only against the view's depth, writing the uv motion of
// its surface — previous minus current, jitter-free — and `w = 1`, "written".
//
// The raster position goes through the frame's JITTERED clip-from-world, the matrix the main pass
// drew the part with, so the depth test meets its own depth; the motion goes through the
// UNJITTERED current and previous matrices, so it carries no sub-pixel offset. SKINNED skins twice,
// with `entity_shadow.wgsl`'s code: once from the live palettes in the shared light buffer, once
// from last world frame's copy of it.

struct VelocityView {
    raster_clip_from_world: mat4x4<f32>,
    cur_clip_from_world: mat4x4<f32>,
    prev_clip_from_world: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> view: VelocityView;

struct VelocityPart {
    world_from_local: mat4x4<f32>,
    prev_world_from_local: mat4x4<f32>,
    rig_slot: u32,
    // 0 opaque, 1 alpha test (224/255), 2 blend (alpha ≥ 0.5), 3 add (max rgb ≥ 0.5).
    coverage: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(1) @binding(0) var<uniform> part: VelocityPart;

#ifdef SKINNED
// The shared light buffer, laid out exactly as `wow_model.wgsl`'s `WowLight`; only the rig
// regions are read.
struct SkinRegions {
    _header: array<vec4<f32>, 21>,
    _points: array<vec4<f32>, 512>,
    // fork: the environment block and the rain-shelter region (`lighting/env_frame.rs`), unread here.
    _env: array<vec4<f32>, 12>,
    _shelter_hdr: array<vec4<f32>, 2>,
    _shelter: array<u32, 16384>,
    _prop_probes: array<vec4<f32>, 57344>,
    rig_table: array<u32, 2048>,
    _rig_tint: array<u32, 2048>,
    rig_origin: array<vec4<f32>, 2048>,
    _matanim: array<vec4<f32>, 2048>,
    _water_clip: array<vec2<f32>, 2048>,
    palettes: array<vec4<f32>>,
}
@group(1) @binding(1) var<storage, read> skin: SkinRegions;
@group(1) @binding(2) var<storage, read> prev_skin: SkinRegions;
#endif

#ifdef TEXTURED
@group(2) @binding(0) var base_tex: texture_2d<f32>;
@group(2) @binding(1) var base_samp: sampler;
#endif

struct PartVertex {
    @location(0) position: vec3<f32>,
#ifdef TEXTURED
    @location(2) uv: vec2<f32>,
#endif
#ifdef SKINNED
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
}

struct PartVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) cur_clip: vec4<f32>,
    @location(1) prev_clip: vec4<f32>,
#ifdef TEXTURED
    @location(2) uv: vec2<f32>,
#endif
}

#ifdef SKINNED
// Twice, not one function over a pointer: a `storage` pointer parameter needs WGSL's
// `unrestricted_pointer_parameters`, which not every browser has.
fn skinned_cur(slot: u32, v: PartVertex) -> vec3<f32> {
    let base = skin.rig_table[slot];
    let b0 = 3u * (base + v.joint_indices.x);
    let b1 = 3u * (base + v.joint_indices.y);
    let b2 = 3u * (base + v.joint_indices.z);
    let b3 = 3u * (base + v.joint_indices.w);
    let w = v.joint_weights;
    let r0 = w.x * skin.palettes[b0] + w.y * skin.palettes[b1]
        + w.z * skin.palettes[b2] + w.w * skin.palettes[b3];
    let r1 = w.x * skin.palettes[b0 + 1u] + w.y * skin.palettes[b1 + 1u]
        + w.z * skin.palettes[b2 + 1u] + w.w * skin.palettes[b3 + 1u];
    let r2 = w.x * skin.palettes[b0 + 2u] + w.y * skin.palettes[b1 + 2u]
        + w.z * skin.palettes[b2 + 2u] + w.w * skin.palettes[b3 + 2u];
    // The palette rows are the affine's ROWS, measured from the rig origin (decision 0974).
    let p = vec4<f32>(v.position, 1.0);
    return vec3<f32>(dot(r0, p), dot(r1, p), dot(r2, p)) + skin.rig_origin[slot].xyz;
}
fn skinned_prev(slot: u32, v: PartVertex) -> vec3<f32> {
    let base = prev_skin.rig_table[slot];
    let b0 = 3u * (base + v.joint_indices.x);
    let b1 = 3u * (base + v.joint_indices.y);
    let b2 = 3u * (base + v.joint_indices.z);
    let b3 = 3u * (base + v.joint_indices.w);
    let w = v.joint_weights;
    let r0 = w.x * prev_skin.palettes[b0] + w.y * prev_skin.palettes[b1]
        + w.z * prev_skin.palettes[b2] + w.w * prev_skin.palettes[b3];
    let r1 = w.x * prev_skin.palettes[b0 + 1u] + w.y * prev_skin.palettes[b1 + 1u]
        + w.z * prev_skin.palettes[b2 + 1u] + w.w * prev_skin.palettes[b3 + 1u];
    let r2 = w.x * prev_skin.palettes[b0 + 2u] + w.y * prev_skin.palettes[b1 + 2u]
        + w.z * prev_skin.palettes[b2 + 2u] + w.w * prev_skin.palettes[b3 + 2u];
    // The palette rows are the affine's ROWS, measured from the rig origin (decision 0974).
    let p = vec4<f32>(v.position, 1.0);
    return vec3<f32>(dot(r0, p), dot(r1, p), dot(r2, p)) + prev_skin.rig_origin[slot].xyz;
}
#endif

@vertex
fn vertex(v: PartVertex) -> PartVsOut {
    var out: PartVsOut;
#ifdef SKINNED
    let world = skinned_cur(part.rig_slot, v);
    let prev_world = skinned_prev(part.rig_slot, v);
#else
    let world = (part.world_from_local * vec4<f32>(v.position, 1.0)).xyz;
    let prev_world = (part.prev_world_from_local * vec4<f32>(v.position, 1.0)).xyz;
#endif
    out.position = view.raster_clip_from_world * vec4<f32>(world, 1.0);
    out.cur_clip = view.cur_clip_from_world * vec4<f32>(world, 1.0);
    out.prev_clip = view.prev_clip_from_world * vec4<f32>(prev_world, 1.0);
#ifdef TEXTURED
    out.uv = v.uv;
#endif
    return out;
}

fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
}

@fragment
fn fragment(in: PartVsOut) -> @location(0) vec4<f32> {
#ifdef TEXTURED
    let texel = textureSampleLevel(base_tex, base_samp, in.uv, 0.0);
    if (part.coverage == 1u && texel.a < 0.8784314) {
        discard;
    }
    if (part.coverage == 2u && texel.a < 0.5) {
        discard;
    }
    if (part.coverage == 3u && max(texel.r, max(texel.g, texel.b)) < 0.5) {
        discard;
    }
#endif
    if (in.cur_clip.w <= 1.0e-6 || in.prev_clip.w <= 1.0e-6) {
        discard;
    }
    let cur = ndc_to_uv(in.cur_clip.xy / in.cur_clip.w);
    let prev = ndc_to_uv(in.prev_clip.xy / in.prev_clip.w);
    return vec4<f32>(prev - cur, 0.0, 1.0);
}
