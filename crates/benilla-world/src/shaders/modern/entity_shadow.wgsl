// Modern lighting (FORK.md, P2 stage 3): entity-lane model parts — fading and
// animated doodads, game objects, creatures and characters — drawn depth-only into the modern sun's
// cascades. SKINNED reads the owned palettes from the shared light buffer exactly as
// `wow_model.wgsl` does; CUTOUT alpha-keys the base texture at the vanilla 224/255 reference.
struct CasterView {
    clip_from_world: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> caster: CasterView;

struct EntityDraw {
    world_from_local: mat4x4<f32>,
    rig_slot: u32,
}
@group(1) @binding(0) var<uniform> part: EntityDraw;

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
#endif

#ifdef CUTOUT
@group(2) @binding(0) var base_tex: texture_2d<f32>;
@group(2) @binding(1) var base_samp: sampler;
#endif

struct PartVertex {
    @location(0) position: vec3<f32>,
#ifdef CUTOUT
    @location(2) uv: vec2<f32>,
#endif
#ifdef SKINNED
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
}

struct PartVsOut {
    @builtin(position) position: vec4<f32>,
#ifdef CUTOUT
    @location(0) uv: vec2<f32>,
#endif
}

@vertex
fn vertex(v: PartVertex) -> PartVsOut {
    var out: PartVsOut;
#ifdef SKINNED
    let slot = part.rig_slot;
    let base = skin.rig_table[slot];
    let b0 = 3u * (base + v.joint_indices.x);
    let b1 = 3u * (base + v.joint_indices.y);
    let b2 = 3u * (base + v.joint_indices.z);
    let b3 = 3u * (base + v.joint_indices.w);
    let w = v.joint_weights;
    let r0 = w.x * skin.palettes[b0] + w.y * skin.palettes[b1] + w.z * skin.palettes[b2] + w.w * skin.palettes[b3];
    let r1 = w.x * skin.palettes[b0 + 1u] + w.y * skin.palettes[b1 + 1u] + w.z * skin.palettes[b2 + 1u] + w.w * skin.palettes[b3 + 1u];
    let r2 = w.x * skin.palettes[b0 + 2u] + w.y * skin.palettes[b1 + 2u] + w.z * skin.palettes[b2 + 2u] + w.w * skin.palettes[b3 + 2u];
    // The palette rows are the affine's ROWS, measured from the rig origin (decision 0974).
    let p = vec4<f32>(v.position, 1.0);
    let world = vec3<f32>(dot(r0, p), dot(r1, p), dot(r2, p)) + skin.rig_origin[slot].xyz;
#else
    let world = (part.world_from_local * vec4<f32>(v.position, 1.0)).xyz;
#endif
    out.position = caster.clip_from_world * vec4<f32>(world, 1.0);
    // Pancake onto the near plane, as the other casters do.
    out.position.z = min(out.position.z, 1.0);
#ifdef CUTOUT
    out.uv = v.uv;
#endif
    return out;
}

#ifdef CUTOUT
@fragment
fn fragment(in: PartVsOut) {
    if (textureSampleLevel(base_tex, base_samp, in.uv, 0.0).a < 0.8784314) {
        discard;
    }
}
#endif
