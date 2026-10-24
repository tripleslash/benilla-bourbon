// Modern lighting: the auto-exposure's screen meter (`modern/meter.rs`). The scene-referred frame,
// before any exposure, averaged into a 64x36 grid of log2 luminance; the CPU takes the grid's mean.

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> cells: array<f32>;

const GRID: vec2<u32> = vec2<u32>(64u, 36u);
// Taps a side in each cell; bilinear, so a cell reads a 2x box around each.
const TAPS: u32 = 4u;
// The floor under a black pixel: ten stops below the target, so one cannot sink the mean.
const LUMA_FLOOR: f32 = 0.0001;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= GRID) {
        return;
    }
    var sum = 0.0;
    for (var j = 0u; j < TAPS; j++) {
        for (var i = 0u; i < TAPS; i++) {
            let tap = (vec2<f32>(f32(i), f32(j)) + 0.5) / f32(TAPS);
            let uv = (vec2<f32>(id.xy) + tap) / vec2<f32>(GRID);
            let c = max(textureSampleLevel(scene, scene_sampler, uv, 0.0).rgb, vec3<f32>(0.0));
            let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            sum += log2(max(luma, LUMA_FLOOR));
        }
    }
    cells[id.y * GRID.x + id.x] = sum / f32(TAPS * TAPS);
}
