#define_import_path benilla::env_frame

// fork: the environment block, 12 rows (192 B) appended to the shared light buffer right after the
// point-light table (`lighting::global_light`, packed from the `EnvFrame` resource in
// `lighting/env_frame.rs`). All zero = no visual change: every reader treats zero as "feature off".
//
// Coordinates are Bevy world space (Y up, 1 unit = 1 yd), the space of the receivers'
// `world_position`. Member names carry no trailing digit: naga_oil refuses composable-module
// identifiers that its namer would rewrite, so the rows are `wind_a/b`, `wet_a`.
struct EnvFrame {
    wind_a: vec4<f32>, // dir_x, dir_y (unit, world XZ: .x = world x, .y = world z), base_heading (rad, fixed profile heading), gust 0..1
    wind_b: vec4<f32>, // travel (yd, integrated speed, wrapped at 4096), sway_strength, grass_strength, tree_strength
    wet_a: vec4<f32>,  // rain_rate 0..1, wetness 0..1, ripple_time_s, snow 0..1
    misc: vec4<f32>,   // bender_count, wet_ssr (1 = the wet reflection pass composites), 0, 0
    benders: array<vec4<f32>, 8>, // world xyz (Bevy space) + radius (yd); the first bender_count are live
}
