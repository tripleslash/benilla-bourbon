// WDL distant terrain: the coarse horizon hulls the reference draws past the detailed tiles,
// unlit, untextured, white vertex diffuse fogged with the scene's fog colour. The hull emitter
// `0x6bd780` sets its own fog pair, start 0 and end 1.0 (`0x6bd7ae`-`0x6bd7c8`), so beyond one
// yard the hull is the flat fog colour: a silhouette against the unfogged sky.
//
// The band is a backdrop, not the far half of a shared clip plane. The reference's far-band pass
// (`0x6841a0`, projection at `0x6842f2`) uses near `farclip - 33` (`[0x8101b0]`) and the depth
// range `[0.955, 0.96]` (`[0x80febc]` = 0.96, `[0x80fec4]` = 0.005): the band overlaps the detailed
// world by 33 yd, closing the seam, and stays behind all of it. On Bevy's infinite reverse-Z
// (depth = near / eye_z) that range becomes a clamp just below `near / farclip`, the farthest
// depth the detailed world keeps (it discards past the wall), still in front of the sky's 0.0.

// fork: modern lighting (FORK.md)
#import benilla_modern::colorspace
#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::{position_world_to_clip, view_z_to_depth_ndc},
    mesh_view_bindings::view,
}

/// How far the band reaches inside the far-clip wall: the reference's far-band near plane
/// `farclip - 33.0` (`[0x8101b0]`, about one WDL outer cell).
const WDL_OVERLAP: f32 = 33.0;

/// Keeps the band strictly behind a world fragment exactly at the wall (`GreaterEqual` passes a
/// tie); multiplicative, so it never crosses zero into the sky.
const WDL_DEPTH_PUSH: f32 = 0.999;

// The shared global light (`lighting::global_light`), mirrored up to the rows WDL reads. Rows 4
// and 5 are the scene fog block, right by construction: the horizon band is never inside a WMO.
struct WowLight {
    _light_ambient: vec4<f32>, // 0
    _light_diffuse: vec4<f32>, // 1
    _light_sun: vec4<f32>,     // 2
    _light_spec: vec4<f32>,    // 3
    fog_color: vec4<f32>,      // 4 rgb = Light.dbc row 7 (gamma 0..1); w = enable (>0.5 ⇒ blend)
    fog_params: vec4<f32>,     // 5 xy = scene fog start/end (unread); z = a modern frame's silhouette shade (`modern::look`); w = farclip wall (0 = off)
    _rows_6_19: array<vec4<f32>, 14>,
    point_count: vec4<f32>,    // 20 // fork: modern lighting (FORK.md) — y = modern flag
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> w: WowLight;

struct WdlVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
}

struct WdlFsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vertex(in: Vertex) -> WdlVsOut {
    var out: WdlVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    return out;
}

@fragment
fn fragment(in: WdlVsOut) -> WdlFsOut {
    // Planar eye-z, not radial: the band's near plane is a projection plane.
    let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
    let farclip = w.fog_params.w;

    // The near plane is `farclip - 33`, not the wall: the overlap closes the coarse-vs-fine seam.
    if (farclip > 0.0 && eye_z < farclip - WDL_OVERLAP) {
        discard;
    }

    // White vertex diffuse under the hull's own fog pair, saturated beyond one yard: the flat fog
    // colour, never the scene fog distances. With fog off the reference shows the bare white hull.
    var rgb = vec3<f32>(1.0);
    if (w.fog_color.w > 0.5) {
        rgb = w.fog_color.xyz;
    }
    // fork: modern lighting (FORK.md) — a modern frame draws the horizon band as an
    // unlit SILHOUETTE: a darker body of the fog's own colour, with no key light and no facets. The fog composite lays the height fog over it from its true distance,
    // so the far ranges' feet vanish into the haze and their crests rise above it.
    let modern = w.point_count.y > 0.5;
    if (modern) {
        rgb = clamp(w.fog_color.xyz * w.fog_params.z, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    var out: WdlFsOut;
    // Raw gamma out; the frame decodes once, in the FFXGlow combine.
    out.color = vec4<f32>(colorspace::modern_out(rgb, w.point_count.y), 1.0); // fork: modern lighting (FORK.md)
    // The depth clamp. `view_z_to_depth_ndc` takes view-space z, negative in front of the camera,
    // so the wall is at `-farclip`.
    var depth = in.clip_position.z;
    if (farclip > 0.0) {
        depth = min(depth, view_z_to_depth_ndc(-farclip) * WDL_DEPTH_PUSH);
        // fork: modern lighting (FORK.md) — a modern frame keeps the band's own
        // distance, pushed back by the overlap so it still lies behind everything the detailed world
        // draws (its near plane is `farclip − 33`, so the push lands at `farclip + 1` or beyond): the
        // fog composite reads this depth, and a band clamped to the wall was fogged as if it stood there.
        if (modern) {
            depth = min(in.clip_position.z, view_z_to_depth_ndc(-(eye_z + WDL_OVERLAP + 1.0)));
        }
    }
    out.depth = depth;
    return out;
}
