// The proxy is deliberately invisible in the camera's forward pass. Its material remains
// opaque, so Bevy still queues it for the depth-only directional-light shadow pass.
@fragment
fn fragment() -> @location(0) vec4<f32> {
    discard;
    return vec4<f32>(0.0);
}
