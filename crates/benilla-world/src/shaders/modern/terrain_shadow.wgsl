// Modern lighting (FORK.md, P2): terrain cells drawn depth-only into the modern
// sun's cascades. Cell meshes are in absolute world coordinates (`terrain_stream::furnish`), and
// Bevy interleaves POSITION first, so the pipeline reads only location 0 at offset 0 whatever the
// stride.
struct CasterView {
    clip_from_world: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> caster: CasterView;

@vertex
fn vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    var p = caster.clip_from_world * vec4<f32>(position, 1.0);
    // Pancake onto the near plane: a hill between the sun and the cascade still casts.
    p.z = min(p.z, 1.0);
    return p;
}
