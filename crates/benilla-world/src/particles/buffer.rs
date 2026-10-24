//! The shared effect stream: one CPU vertex stream every dynamic-effect family writes each frame,
//! and the draw records slicing it. A writer calls [`EffectQuads::begin`], pushes world-space
//! vertices (whole quads in perimeter order, or a triangle list) and commits one draw for the
//! range; the render half rebases them camera-relative for upload (absolute f32 shears thin
//! geometry far from the origin) and sorts by [`EffectDraw::anchor`] view z plus the rung.

use std::ops::Range;

use benilla_formats::{ModelBlend, ParticleBlend};
use bevy::asset::AssetId;
use bevy::prelude::*;
use bevy::render::render_resource::Buffer;

/// One vertex of the lane: world-space in the stream, so instruments read world coordinates,
/// except on a [`EffectDrawSpec::cam_relative`] draw.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// Raw authored gamma-space RGBA; alpha is the blend weight, never encoded.
    pub color: [f32; 4],
}

/// The lane's blends: [`ParticleBlend`]'s four plus the multiplicative pair of decals and rain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EffectBlend {
    /// `(SRC_ALPHA, ONE)`, as premultiplied alpha after the shader's gamma `rgb·a` fold.
    Add,
    /// Standard alpha blending.
    Alpha,
    /// No blend, depth write on, drawn in the transparent bracket at the owner rung.
    Opaque,
    /// EGxBlend 1: [`EffectBlend::Opaque`]'s state plus an alpha test discarding below 224/255
    /// (`VANILLA_ALPHA_KEY_REF`, the mesh cutouts' ref; see [`ParticleBlend::AlphaKey`]).
    AlphaKey,
    /// `dst · lerp(1, src, α)`, bevy's `AlphaMode::Multiply` state: the blob shadow's
    /// `GL_DST_COLOR/GL_ZERO` with fade, and `ModelBlend::Mod` at α = 1.
    Multiply,
    /// `2·src·dst` as `(Dst, Src)`: rain's state.
    Mod2x,
}

impl From<ParticleBlend> for EffectBlend {
    fn from(blend: ParticleBlend) -> Self {
        match blend {
            ParticleBlend::Add => EffectBlend::Add,
            ParticleBlend::Alpha => EffectBlend::Alpha,
            ParticleBlend::AlphaKey => EffectBlend::AlphaKey,
            ParticleBlend::Opaque => EffectBlend::Opaque,
        }
    }
}

impl EffectBlend {
    /// The ground-fx mapping of a part's blend: `model_render`'s, with `additive` apart because M2
    /// modes 3 and 4 fold into [`ModelBlend::Blend`]. Deviation: `AlphaTest` maps to `Alpha`,
    /// because flat `Spells\` quads are blend batches and a cutout would bite a soft decal's fade;
    /// and the part draws unlit, as no lit ground-fx part is known.
    pub fn from_model(blend: ModelBlend, additive: bool) -> Self {
        if additive {
            // The material path's additive: the shader premultiplies `rgb·α` and returns α = 0.
            return EffectBlend::Add;
        }
        match blend {
            ModelBlend::Opaque => EffectBlend::Opaque,
            ModelBlend::AlphaTest | ModelBlend::Blend => EffectBlend::Alpha,
            ModelBlend::Mod => EffectBlend::Multiply,
            ModelBlend::Mod2x => EffectBlend::Mod2x,
        }
    }
}

/// fork: perf — one point of a point draw: the vertex stage builds its triangle or quad around
/// it ([`EffectShape`], `effect_points.wgsl`), so a field of them uploads these 32 bytes each and
/// no CPU touches a corner. Heavy rain was 2.6 MB of triangles built, copied and uploaded a frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectPoint {
    /// World position, absolute: the shader takes it camera-relative before it builds anything,
    /// so the corners keep their precision far from the origin.
    pub pos: [f32; 3],
    /// The shape's scalar ([`EffectShape`]).
    pub a: f32,
    /// The shape's vector.
    pub v: [f32; 3],
    /// The shape's second scalar.
    pub b: f32,
}

/// fork: perf — one live particle of a GPU-expanded pool, an instance of its draw: the stored-frame
/// position and velocity, the age, and what the packing reads off the twinkle table, which the
/// vertex stage then never needs. A particle the twinkle gate drops is never packed.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub pos: [f32; 3],
    pub age: f32,
    pub vel: [f32; 3],
    /// The normalized age `clamp(age / life, 0, 1)` the over-life ramps read.
    pub u_age: f32,
    /// The gated twinkle's size multiplier ([`benilla_formats::ParticleEmitterDef::twinkle`]).
    pub twinkle: f32,
    /// The particle's phase, whose bit 5 counter-rotates a negative spin (`quads::spin_angle`).
    pub phase: u32,
}

/// fork: perf — one GPU-expanded pool's record, one dynamic-offset uniform row the vertex stage
/// builds its particles' quads from: the CPU writer's per-pool inputs (`quads::expand_quads`).
/// Sixteen rows, one 256-byte uniform slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, bevy::render::render_resource::ShaderType)]
pub struct ParticlePool {
    /// The stored frame to the world, row-major (`xyz` the linear part, `w` the translation):
    /// the ride frame in world mode, the placement times `wow_to_bevy` in model mode.
    pub center_x: Vec4,
    pub center_y: Vec4,
    pub center_z: Vec4,
    /// The camera basis the quads face; `w`: the size unit (instance scale or the lane's).
    pub cam_right: Vec4,
    /// `w`: the model's render alpha.
    pub cam_up: Vec4,
    /// The XY-quad plane basis (zero unless flagged); `w`: the spin.
    pub plane_r: Vec4,
    /// `w`: the tail time.
    pub plane_u: Vec4,
    /// The three colour keys.
    pub color0: Vec4,
    pub color1: Vec4,
    pub color2: Vec4,
    /// The three size keys, then the split `mid`.
    pub scale: Vec4,
    /// The head ramps `(base, span)` for segments A and B.
    pub head: Vec4,
    /// The tail ramps likewise.
    pub tail: Vec4,
    /// The segment repeats, then the atlas columns and rows.
    pub atlas: Vec4,
    /// `x`: `head_tail | xy_quad << 2 | tail_clamps << 3 | committed << 4`; `yzw`: the committed
    /// light the RGB folds by.
    pub mode: Vec4,
    /// Spare, keeping the record one 256-byte slot.
    pub spare: Vec4,
}

/// fork: perf — what a point draw builds around each [`EffectPoint`]: the weather's four sprites.
/// The laws are in `effect_points.wgsl`; the numbers are the producer's.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EffectShape {
    /// A white triangle along a velocity (`v`): base `pos ∓ half_w·right` across the eye line,
    /// apex `pos + tilt·(tail·antiVel̂)`. fork: `near` > 0 (modern frames) shrinks it to nothing
    /// toward the eye, whole from `near` yards and gone inside a quarter of it; 0 keeps it whole.
    Streak {
        half_w: f32,
        tail: f32,
        near: f32,
        tilt: Quat,
    },
    /// A white camera-facing triangle `pos ∓ right·view_right`, `pos + up·view_up`, stepping the
    /// four frames of row `b` of a 4×4 atlas over `life` seconds of age `a`.
    Patter { right: f32, up: f32, life: f32 },
    /// A white screen-aligned quad of alpha `a`, sized in pixels of a reference screen:
    /// `max(px_min, px_at_eye·clamp01(1 − falloff·d))` at `d` yards, a pixel being
    /// `world_per_px` world units per unit of view depth.
    Flake {
        px_at_eye: f32,
        falloff: f32,
        px_min: f32,
        world_per_px: f32,
    },
    /// A camera-facing quad of half-extent `half` in `color`, each corner's alpha
    /// `a·linearstep(near, far, the corner's distance)`.
    Puff {
        half: f32,
        near: f32,
        far: f32,
        color: [f32; 3],
        /// fork: modern weather — the fragment's coverage is `effect_points::puff_falloff` in
        /// place of the texture.
        procedural: bool,
    },
}

impl EffectShape {
    /// Vertices drawn per point: a triangle, or a quad as two.
    pub(crate) fn vertices(self) -> u32 {
        match self {
            Self::Streak { .. } | Self::Patter { .. } => 3,
            Self::Flake { .. } | Self::Puff { .. } => 6,
        }
    }

    /// The two params rows `effect_points.wgsl` reads; the first row's `x` names the shape.
    pub(crate) fn rows(self) -> [Vec4; 2] {
        match self {
            Self::Streak {
                half_w,
                tail,
                near,
                tilt,
            } => [Vec4::new(1.0, half_w, tail, near), Vec4::from(tilt)],
            Self::Patter { right, up, life } => [Vec4::new(2.0, right, up, life), Vec4::ZERO],
            Self::Flake {
                px_at_eye,
                falloff,
                px_min,
                world_per_px,
            } => [
                Vec4::new(3.0, px_at_eye, falloff, px_min),
                Vec4::new(world_per_px, 0.0, 0.0, 0.0),
            ],
            Self::Puff {
                half,
                near,
                far,
                color,
                procedural,
            } => [
                Vec4::new(4.0, half, near, far),
                Vec3::from(color).extend(if procedural { 1.0 } else { 0.0 }),
            ],
        }
    }
}

/// How a draw's range is drawn.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EffectTopology {
    /// Whole quads: 4 perimeter-order corners each, indexed `[b, b+1, b+2, b, b+2, b+3]`.
    Quads,
    /// A plain triangle list: identity indices (the decal projector's fans).
    Tris,
    /// fork: perf — a range of [`EffectQuads::points`], each built into this shape by the vertex
    /// stage.
    Points(EffectShape),
    /// fork: perf — a range of [`EffectQuads::particles`], one particle pool whose head and tail
    /// quads the vertex stage builds from [`EffectQuads::pools`]`[pool]` (`wow_effect.wgsl`'s
    /// `EFFECT_PARTICLES`, the CPU writer's law, `quads::expand_quads`).
    Particles { pool: u32, quads: u32 },
}

/// One draw's fog colour policy, one canonical row of the shader's params uniform.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EffectFog {
    /// No fog (params.x = 0): file flag 0x8, the decals and foam.
    Off,
    /// The scene fog (params.x = 1): Alpha and Opaque blends.
    Scene,
    /// Toward black (params.x = 2), for Add blends, from M2's per-blend fog table (`0x70baf0`).
    Black,
    /// Toward white (params.x = 3), the table's Mod policy: ground-fx `ModelBlend::Mod` decals.
    White,
    /// Toward grey 0.5 (params.x = 4), the table's Mod2x policy: ground-fx Mod2x decals.
    Grey,
    /// Rain's forced grey fog (params.y = 1, zw 70..75), its Mod2x distance fade (`0x59d350`).
    Rain,
}

impl EffectFog {
    /// The policy for one particle or ribbon def.
    pub fn for_blend(flags: u32, blend: ParticleBlend) -> Self {
        if flags & 0x8 != 0 {
            EffectFog::Off
        } else if matches!(blend, ParticleBlend::Add) {
            EffectFog::Black
        } else {
            EffectFog::Scene
        }
    }

    /// The `0x70baf0` fog policy a model material packs in `clutter_fade.z` bits 4..7.
    pub fn from_model_policy(policy: u32) -> Self {
        match policy {
            1 => EffectFog::Black,
            2 => EffectFog::White,
            3 => EffectFog::Grey,
            4 => EffectFog::Off,
            _ => EffectFog::Scene,
        }
    }

    /// The slot index into the render-world params uniform (one canonical `vec4` per policy).
    pub fn slot(self) -> u32 {
        match self {
            EffectFog::Off => 0,
            EffectFog::Scene => 1,
            EffectFog::Black => 2,
            EffectFog::White => 3,
            EffectFog::Grey => 4,
            EffectFog::Rain => 5,
        }
    }
}

/// Where one draw's light comes from. The reference builds an `M2Material` from each emitter
/// record, so a particle is lit like a submesh, by its model's own light node (`0x672a20`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum EffectLighting {
    /// Unlit: every family but M2 emitters, and the emitters that set the file's unlit bit 0x1.
    #[default]
    None,
    /// The scene's light, per view in the shader against world up (a booth's buffer overrides
    /// it): a lit emitter outdoors, mostly `World\` environment sheets.
    Scene,
    /// The model's own committed light in a WMO room, never the sun (`0x6a7300`'s interior leg);
    /// one normal per quad makes it one RGB, already folded into the vertex colours.
    Committed,
}

/// fork: fsr (FORK.md) — how a draw's vertices moved since the last world frame, which FSR's
/// velocity pass draws as their motion (`fsr/velocity.rs`). A held frame runs no main world, so
/// "the last world frame" is always the one before this.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EffectMotion {
    /// Still in the world: its pixels move with the camera alone.
    #[default]
    Static,
    /// Every vertex moved by this world displacement, current minus previous.
    Rigid(Vec3),
    /// Each vertex's own displacement, pushed to [`EffectQuads::motion`] beside it.
    PerVertex,
}

/// A per-emitter `WowLight` buffer read in place of the world's; the glue-scene booths set it.
#[derive(Component, Clone)]
pub struct EffectLightOverride(pub Buffer);

/// One draw of the lane: a vertex range, its texture, blend and fog, and its sort point.
pub struct EffectDraw {
    /// The main-world camera whose view draws this (the world's, or a booth's).
    pub cam: Entity,
    pub(crate) texture: AssetId<Image>,
    pub(crate) blend: EffectBlend,
    pub(crate) topology: EffectTopology,
    pub(crate) fog: EffectFog,
    /// The shader applies the scene's light: [`EffectLighting::Scene`] alone, a pipeline-key axis.
    pub lit: bool,
    /// The sort point: the emitter anchor, ribbon head node or decal centre.
    pub(crate) anchor: Vec3,
    /// The rung added to the view-space sort distance (`sky_order`: positive draws later).
    pub(crate) bias: f32,
    /// The rasterizer depth-bias constant, nonzero only for coplanar draws, which then skip the
    /// rebase for the world meshes' own `clip_from_world`, so the tie it settles is like for like.
    pub(crate) raster_bias: i32,
    pub(crate) raster_slope: f32,
    pub(crate) cam_relative: bool,
    pub(crate) no_depth_test: bool,
    /// Vertex range in [`EffectQuads::verts`] (a multiple of 4 for quads, 3 for tris); a point
    /// draw's range is in [`EffectQuads::points`].
    pub range: Range<u32>,
    /// The producing entity, the item's identity in the phase (`item.entity.1`).
    pub main_entity: Entity,
    /// [`EffectLightOverride`]'s buffer; `None` is the world's.
    pub(crate) light: Option<Buffer>,
    pub(crate) clip: Option<Vec4>,
}

/// The frame's stream: cleared by [`begin_effect_frame`], filled by the families, extracted.
#[derive(Resource)]
pub struct EffectQuads {
    pub verts: Vec<EffectVertex>,
    /// fork: perf — the point draws' records ([`EffectTopology::Points`]).
    pub points: Vec<EffectPoint>,
    /// fork: perf — the GPU-expanded pools' particles and records ([`EffectTopology::Particles`]).
    pub particles: Vec<GpuParticle>,
    pub pools: Vec<ParticlePool>,
    pub draws: Vec<EffectDraw>,
    // fork: fsr (FORK.md) — each vertex's world displacement since the last world frame,
    /// parallel to [`Self::verts`] once a draw commits ([`EffectMotion`]).
    pub motion: Vec<[f32; 3]>,
    // fork: fsr (FORK.md) — whether [`Self::motion`] is kept this frame: only while a camera
    /// runs FSR, whose velocity pass is its one reader. Off, nothing writes or uploads it.
    track_motion: bool,
    /// Has [`begin_effect_frame`] run this frame? A commit before it would be erased, so
    /// [`Self::commit`] asserts it in debug builds: every writer needs
    /// `.after(begin_effect_frame)`. [`clear_effect_frame_flag`] resets it in `Last`.
    cleared_this_frame: bool,
}

impl Default for EffectQuads {
    fn default() -> Self {
        Self {
            verts: Vec::new(),
            points: Vec::new(),
            particles: Vec::new(),
            pools: Vec::new(),
            draws: Vec::new(),
            motion: Vec::new(),
            track_motion: false,
            // True, so a fixture with no clear can commit; the app's `Last` reset arms the check.
            cleared_this_frame: true,
        }
    }
}

/// Everything about one draw but its vertex range, as `commit_quads` and `commit_tris` take it.
pub struct EffectDrawSpec {
    pub cam: Entity,
    pub texture: AssetId<Image>,
    pub blend: EffectBlend,
    pub fog: EffectFog,
    /// The light the colour meets. The scene term is `clamp(ambient + diffuse·max(N·L, 0))` with
    /// N world up: the reference uploads one normal per draw, world +Z (`0x7b3fd0`).
    pub lighting: EffectLighting,
    pub anchor: Vec3,
    pub bias: f32,
    pub raster_bias: i32,
    /// The rasterizer depth-bias slope scale. The reference uses none: its one slope term
    /// (`0x59bf0a`, `glPolygonOffset` factor `0xc0800000` = −4.0) is in the GL arm, never built.
    pub raster_slope: f32,
    /// The producer wrote camera-relative vertices, so the upload rebase skips this draw: the
    /// snow flake's sub-centimetre geometry, which absolute f32 world coordinates cannot hold.
    /// [`Self::anchor`] stays absolute.
    pub cam_relative: bool,
    /// Depth compare `Always`: the weapon swing trail, whose callback turns the depth test off
    /// (EGxRs id `0x10` = 0, `0x6c686e`) so the swinger's shoulder never eats the arc.
    pub no_depth_test: bool,
    pub main_entity: Entity,
    pub light: Option<Buffer>,
    /// A target-pixel clip rect `(min.x, min.y, max.x, max.y)`; `None` is the whole target. UI
    /// model tiles share one atlas where the reference gives each `<Model>` its own viewport
    /// (`0x59c730`), so the fragment discards outside the pane's cell.
    pub clip: Option<Vec4>,
    // fork: fsr (FORK.md) — how the vertices moved since the last world frame.
    pub motion: EffectMotion,
}

impl EffectQuads {
    /// Open a draw: remember where its vertices start. fork: fsr (FORK.md) — the motion
    /// stream catches up with the vertices first, so a `PerVertex` writer's pushes line up.
    pub fn begin(&mut self) -> u32 {
        if self.track_motion {
            self.motion.resize(self.verts.len(), [0.0; 3]);
        }
        self.verts.len() as u32
    }

    /// fork: fsr (FORK.md) — whether this frame keeps the motion stream: a `PerVertex`
    /// writer pushes to [`Self::motion`], and keeps its last-world-frame history, only while it is.
    pub fn tracks_motion(&self) -> bool {
        self.track_motion
    }

    /// Close a quad draw over everything pushed since `begin`; an empty range commits nothing.
    pub fn commit_quads(&mut self, start: u32, spec: EffectDrawSpec) {
        debug_assert_eq!((self.verts.len() as u32 - start) % 4, 0, "whole quads only");
        self.commit(start, EffectTopology::Quads, spec);
    }

    /// Close a triangle-list draw over everything pushed since `begin`.
    pub fn commit_tris(&mut self, start: u32, spec: EffectDrawSpec) {
        debug_assert_eq!(
            (self.verts.len() as u32 - start) % 3,
            0,
            "whole triangles only"
        );
        self.commit(start, EffectTopology::Tris, spec);
    }

    fn commit(&mut self, start: u32, topology: EffectTopology, spec: EffectDrawSpec) {
        debug_assert!(
            self.cleared_this_frame,
            "effect-stream write before `begin_effect_frame`: this draw is about to be erased. \
             The producing system needs `.after(crate::particles::buffer::begin_effect_frame)` \
             (see `EffectQuads::cleared_this_frame`)."
        );
        let end = self.verts.len() as u32;
        // fork: fsr (FORK.md) — the draw's motion, parallel to its vertices.
        if self.track_motion {
            match spec.motion {
                EffectMotion::PerVertex => self.motion.resize(end as usize, [0.0; 3]),
                EffectMotion::Static => {
                    self.motion.resize(start as usize, [0.0; 3]);
                    self.motion.resize(end as usize, [0.0; 3]);
                }
                EffectMotion::Rigid(d) => {
                    self.motion.resize(start as usize, [0.0; 3]);
                    self.motion.resize(end as usize, d.to_array());
                }
            }
        }
        if end > start {
            self.draws.push(EffectDraw::new(spec, topology, start..end));
        }
    }

    /// fork: perf — commit one GPU-expanded pool: its particles, its record, and a draw of
    /// `quads` quads a particle (a head, a tail, or both). An empty pool commits nothing. A
    /// particle draw keeps no motion: the stream is only kept while FSR runs, and then every pool
    /// takes the CPU writer.
    pub fn commit_particles(
        &mut self,
        pool: ParticlePool,
        particles: &[GpuParticle],
        quads: u32,
        spec: EffectDrawSpec,
    ) {
        debug_assert!(
            self.cleared_this_frame,
            "effect-stream write before `begin_effect_frame`: this draw is about to be erased"
        );
        debug_assert!(
            !self.track_motion,
            "a GPU pool while the motion stream is kept"
        );
        if particles.is_empty() || quads == 0 {
            return;
        }
        let start = self.particles.len() as u32;
        self.particles.extend_from_slice(particles);
        let index = self.pools.len() as u32;
        self.pools.push(pool);
        self.draws.push(EffectDraw::new(
            spec,
            EffectTopology::Particles { pool: index, quads },
            start..self.particles.len() as u32,
        ));
    }

    /// fork: perf — open a point draw: remember where its points start.
    pub fn begin_points(&mut self) -> u32 {
        self.points.len() as u32
    }

    /// fork: perf — close a point draw over everything pushed to [`Self::points`] since
    /// `begin_points`; an
    /// empty range commits nothing. A point draw is free-floating, unlit and still in the world:
    /// the spec's raster bias, lighting and motion do not apply to it.
    pub fn commit_points(&mut self, start: u32, shape: EffectShape, spec: EffectDrawSpec) {
        debug_assert!(
            self.cleared_this_frame,
            "effect-stream write before `begin_effect_frame`: this draw is about to be erased"
        );
        debug_assert_eq!(spec.raster_bias, 0, "a point draw takes no raster bias");
        let end = self.points.len() as u32;
        if end > start {
            self.draws.push(EffectDraw::new(
                spec,
                EffectTopology::Points(shape),
                start..end,
            ));
        }
    }
}

impl EffectDraw {
    fn new(spec: EffectDrawSpec, topology: EffectTopology, range: Range<u32>) -> Self {
        Self {
            cam: spec.cam,
            texture: spec.texture,
            blend: spec.blend,
            topology,
            fog: spec.fog,
            lit: spec.lighting == EffectLighting::Scene,
            anchor: spec.anchor,
            bias: spec.bias,
            raster_bias: spec.raster_bias,
            raster_slope: spec.raster_slope,
            cam_relative: spec.cam_relative,
            no_depth_test: spec.no_depth_test,
            range,
            main_entity: spec.main_entity,
            light: spec.light,
            clip: spec.clip,
        }
    }
}

/// Clear the stream for a new frame; every family's writer runs after it. fork: fsr
/// (FORK.md) — and decide whether this frame keeps the motion stream: a camera runs FSR.
pub fn begin_effect_frame(
    mut quads: ResMut<EffectQuads>,
    fsr: Query<(), With<crate::fsr::FsrView>>,
) {
    quads.verts.clear();
    quads.points.clear();
    quads.particles.clear();
    quads.pools.clear();
    quads.draws.clear();
    quads.motion.clear();
    quads.track_motion = !fsr.is_empty();
    quads.cleared_this_frame = true;
}

/// Reset [`EffectQuads::cleared_this_frame`] in `Last`, after extraction.
pub fn clear_effect_frame_flag(mut quads: ResMut<EffectQuads>) {
    quads.cleared_this_frame = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M2 modes 2 to 4 fold into `ModelBlend::Blend`; flat `Spells\` ground quads are mode 4.
    #[test]
    fn additive_wins_over_the_folded_blend_enum() {
        for blend in [
            ModelBlend::Opaque,
            ModelBlend::AlphaTest,
            ModelBlend::Blend,
            ModelBlend::Mod,
            ModelBlend::Mod2x,
        ] {
            assert_eq!(
                EffectBlend::from_model(blend, true),
                EffectBlend::Add,
                "{blend:?} + additive must reach the pure-add state, not {:?}",
                EffectBlend::from_model(blend, false),
            );
        }
    }

    /// `model_render`'s mapping, with `AlphaTest` folded to `Alpha`.
    #[test]
    fn non_additive_keeps_the_material_paths_law() {
        assert_eq!(
            EffectBlend::from_model(ModelBlend::Opaque, false),
            EffectBlend::Opaque
        );
        assert_eq!(
            EffectBlend::from_model(ModelBlend::AlphaTest, false),
            EffectBlend::Alpha
        );
        assert_eq!(
            EffectBlend::from_model(ModelBlend::Blend, false),
            EffectBlend::Alpha
        );
        assert_eq!(
            EffectBlend::from_model(ModelBlend::Mod, false),
            EffectBlend::Multiply
        );
        assert_eq!(
            EffectBlend::from_model(ModelBlend::Mod2x, false),
            EffectBlend::Mod2x
        );
    }

    /// `effect_points.wgsl` reads a shape's numbers by lane: the first row's `x` names the
    /// shape, and the rest sit where its branch takes them.
    #[test]
    fn a_shape_packs_its_numbers_in_the_lanes_the_shader_reads() {
        let tilt = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9);
        let streak = EffectShape::Streak {
            half_w: 0.05,
            tail: 2.0,
            near: 4.0,
            tilt,
        };
        assert_eq!(
            streak.rows(),
            [
                Vec4::new(1.0, 0.05, 2.0, 4.0),
                Vec4::new(0.1, 0.2, 0.3, 0.9)
            ]
        );
        let patter = EffectShape::Patter {
            right: 0.25,
            up: 0.5,
            life: 0.75,
        };
        assert_eq!(patter.rows(), [Vec4::new(2.0, 0.25, 0.5, 0.75), Vec4::ZERO]);
        let flake = EffectShape::Flake {
            px_at_eye: 14.0,
            falloff: 0.02,
            px_min: 1.0,
            world_per_px: 0.001,
        };
        assert_eq!(
            flake.rows(),
            [
                Vec4::new(3.0, 14.0, 0.02, 1.0),
                Vec4::new(0.001, 0.0, 0.0, 0.0)
            ]
        );
        let puff = EffectShape::Puff {
            half: 6.0,
            near: 6.0,
            far: 18.0,
            color: [0.5, 0.6, 0.7],
            procedural: false,
        };
        assert_eq!(
            puff.rows(),
            [
                Vec4::new(4.0, 6.0, 6.0, 18.0),
                Vec4::new(0.5, 0.6, 0.7, 0.0)
            ]
        );
        let soft = EffectShape::Puff {
            half: 6.0,
            near: 6.0,
            far: 18.0,
            color: [0.5, 0.6, 0.7],
            procedural: true,
        };
        assert_eq!(soft.rows()[1], Vec4::new(0.5, 0.6, 0.7, 1.0));
        // A triangle a point, or a quad as two.
        assert_eq!(
            [streak, patter, flake, puff].map(EffectShape::vertices),
            [3, 3, 6, 6]
        );
    }

    /// The point library parses and validates as plain WGSL, its import path aside: the effect
    /// lane and FSR's velocity pass both build their point draws through it.
    #[test]
    fn the_point_library_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let source = include_str!("../shaders/effect_points.wgsl")
            .strip_prefix("#define_import_path benilla_world::effect_points")
            .expect("the library opens with its import path");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        assert!(
            module
                .functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some("expand")),
            "the entry both shaders call"
        );
    }

    /// A point draw rides the same records as a vertex draw, its range in the point stream.
    #[test]
    fn a_point_draw_commits_its_range_of_points() {
        let mut quads = EffectQuads::default();
        let shape = EffectShape::Patter {
            right: 1.0,
            up: 1.0,
            life: 1.0,
        };
        let spec = || EffectDrawSpec {
            cam: Entity::PLACEHOLDER,
            texture: AssetId::default(),
            blend: EffectBlend::Mod2x,
            fog: EffectFog::Rain,
            lighting: EffectLighting::None,
            anchor: Vec3::ZERO,
            bias: 0.0,
            raster_bias: 0,
            raster_slope: 0.0,
            cam_relative: false,
            no_depth_test: false,
            main_entity: Entity::PLACEHOLDER,
            light: None,
            clip: None,
            motion: EffectMotion::Static,
        };
        let start = quads.begin_points();
        quads.commit_points(start, shape, spec());
        assert!(quads.draws.is_empty(), "an empty range commits nothing");
        let point = EffectPoint {
            pos: [1.0, 2.0, 3.0],
            a: 0.0,
            v: [0.0; 3],
            b: 0.0,
        };
        quads.points.extend([point; 5]);
        let start = quads.begin_points();
        quads.points.extend([point; 3]);
        quads.commit_points(start, shape, spec());
        assert_eq!(quads.draws.len(), 1);
        assert_eq!(quads.draws[0].range, 5..8);
        assert_eq!(quads.draws[0].topology, EffectTopology::Points(shape));
        assert!(quads.verts.is_empty(), "no vertex is built on the CPU");
    }

    /// A writer without `.after(begin_effect_frame)` runs first when the clear carries a
    /// dependency it lacks; the graph is `ParticlePlugin`'s clear and `EntitiesPlugin`'s beam.
    mod write_order {
        use super::*;

        #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
        struct Place;

        fn joint_palette() {}
        fn finalize_rigs() {}
        fn face_billboards() {}

        fn writer(mut quads: ResMut<EffectQuads>) {
            let start = quads.begin();
            for _ in 0..4 {
                quads.verts.push(EffectVertex {
                    pos: [0.0; 3],
                    uv: [0.0; 2],
                    color: [1.0; 4],
                });
            }
            quads.commit_quads(
                start,
                EffectDrawSpec {
                    cam: Entity::PLACEHOLDER,
                    texture: AssetId::default(),
                    blend: EffectBlend::Add,
                    fog: EffectFog::Off,
                    lighting: EffectLighting::None,
                    anchor: Vec3::ZERO,
                    bias: 0.0,
                    raster_bias: 0,
                    raster_slope: 0.0,
                    cam_relative: false,
                    no_depth_test: false,
                    main_entity: Entity::PLACEHOLDER,
                    light: None,
                    clip: None,
                    motion: EffectMotion::Static,
                },
            );
        }

        /// `edged`: whether the writer declares `.after(begin_effect_frame)`.
        fn run(edged: bool) -> App {
            let mut app = App::new();
            app.init_resource::<EffectQuads>();
            app.add_systems(
                PostUpdate,
                (joint_palette, finalize_rigs, face_billboards)
                    .chain()
                    .in_set(Place),
            );
            let w = writer
                .in_set(Place)
                .after(joint_palette)
                .after(finalize_rigs);
            app.add_systems(
                PostUpdate,
                if edged {
                    w.after(begin_effect_frame)
                } else {
                    w
                },
            );
            app.add_systems(
                PostUpdate,
                begin_effect_frame
                    .in_set(Place)
                    .after(joint_palette)
                    .after(finalize_rigs)
                    // The dependency the writer lacks.
                    .after(face_billboards),
            );
            app.add_systems(Last, clear_effect_frame_flag);
            // Two frames: the check only bites after the first `Last` reset.
            app.update();
            app.update();
            app
        }

        /// With the edge, the writer's draw survives the frame and reaches extract.
        #[test]
        fn the_edge_keeps_the_draw() {
            let app = run(true);
            let quads = app.world().resource::<EffectQuads>();
            assert_eq!(quads.draws.len(), 1, "the draw must survive to extract");
            assert_eq!(quads.verts.len(), 4);
        }

        /// Without it, the write precedes the clear and the debug assert fires.
        #[test]
        #[should_panic(expected = "effect-stream write before `begin_effect_frame`")]
        fn without_the_edge_the_tripwire_names_it() {
            let _ = run(false);
        }
    }
}

/// Draw a batch of quads or triangles in the effect lane: the writer for everything but
/// particles, with the lane's defaults for what it does not set.
#[derive(bevy::ecs::system::SystemParam)]
pub struct WorldEffectDraw<'w> {
    quads: ResMut<'w, EffectQuads>,
}

impl<'w> WorldEffectDraw<'w> {
    /// fork: fsr (FORK.md) — whether this frame keeps the motion stream
    /// ([`EffectQuads::tracks_motion`]).
    pub fn tracks_motion(&self) -> bool {
        self.quads.tracks_motion()
    }

    /// Open a batch drawn through `cam` with `texture`; it commits when closed, if not empty.
    pub fn batch(&mut self, cam: Entity, texture: AssetId<Image>) -> EffectBatch<'_> {
        let start = self.quads.begin();
        EffectBatch {
            start,
            spec: EffectDrawSpec {
                cam,
                texture,
                blend: EffectBlend::Alpha,
                fog: EffectFog::Off,
                lighting: EffectLighting::None,
                anchor: Vec3::ZERO,
                bias: 0.0,
                raster_bias: 0,
                raster_slope: 0.0,
                cam_relative: false,
                no_depth_test: false,
                main_entity: Entity::PLACEHOLDER,
                light: None,
                clip: None,
                motion: EffectMotion::Static,
            },
            quads: &mut self.quads,
        }
    }
}

/// One open batch, pushed to and closed by its caller.
pub struct EffectBatch<'a> {
    quads: &'a mut EffectQuads,
    start: u32,
    spec: EffectDrawSpec,
}

impl EffectBatch<'_> {
    /// Additive: glows, rings, beams.
    pub fn additive(mut self) -> Self {
        self.spec.blend = EffectBlend::Add;
        self
    }

    /// `SRC_ALPHA / INV_SRC_ALPHA`, the default, for a reference alpha state (the swing trail's
    /// EGxRs `0x07 = 2`).
    pub fn alpha(mut self) -> Self {
        self.spec.blend = EffectBlend::Alpha;
        self
    }

    /// Multiply into what is drawn: the blob shadow.
    pub fn multiply(mut self) -> Self {
        self.spec.blend = EffectBlend::Multiply;
        self
    }

    /// An explicit blend, for a lane whose authored data carries one.
    pub fn blend(mut self, blend: EffectBlend) -> Self {
        self.spec.blend = blend;
        self
    }

    /// An explicit fog policy; the default is [`EffectFog::Off`].
    pub fn fog(mut self, fog: EffectFog) -> Self {
        self.spec.fog = fog;
        self
    }

    /// The world point this batch sorts by.
    pub fn anchored(mut self, at: Vec3) -> Self {
        self.spec.anchor = at;
        self
    }

    /// The draw-order rung: sort bias and raster depth-bias constant (`crate::sky_order::Rung`).
    pub fn rung(mut self, sort: f32, raster: i32) -> Self {
        self.spec.bias = sort;
        self.spec.raster_bias = raster;
        self
    }

    /// Occluded by nothing; see [`EffectDrawSpec::no_depth_test`].
    pub fn over_everything(mut self) -> Self {
        self.spec.no_depth_test = true;
        self
    }

    /// The entity this draw belongs to, for the render world's per-item bookkeeping.
    pub fn owner(mut self, entity: Entity) -> Self {
        self.spec.main_entity = entity;
        self
    }

    /// fork: fsr (FORK.md) — how the batch moved since the last world frame; a
    /// [`EffectMotion::PerVertex`] batch pushes each vertex's displacement to [`Self::motion_mut`].
    pub fn moving(mut self, motion: EffectMotion) -> Self {
        self.spec.motion = motion;
        self
    }

    /// fork: fsr (FORK.md) — the per-vertex motion sink, parallel to [`Self::verts_mut`].
    pub fn motion_mut(&mut self) -> &mut Vec<[f32; 3]> {
        &mut self.quads.motion
    }

    /// fork: fsr (FORK.md) — whether this frame keeps the motion stream
    /// ([`EffectQuads::tracks_motion`]); a `PerVertex` batch pushes nothing while it is off.
    pub fn tracks_motion(&self) -> bool {
        self.quads.tracks_motion()
    }

    /// Push vertices built elsewhere (the decal lanes project once and cache).
    pub fn vertices(&mut self, verts: &[EffectVertex]) {
        self.quads.verts.extend_from_slice(verts);
    }

    /// Push vertices from an iterator.
    pub fn extend(&mut self, verts: impl IntoIterator<Item = EffectVertex>) {
        self.quads.verts.extend(verts);
    }

    /// The vertex sink, for a producer that writes in place.
    pub fn verts_mut(&mut self) -> &mut Vec<EffectVertex> {
        &mut self.quads.verts
    }

    /// Close the batch as a triangle list.
    pub fn tris(self) {
        self.quads.commit_tris(self.start, self.spec);
    }

    /// Close it as quads.
    pub fn quads(self) {
        self.quads.commit_quads(self.start, self.spec);
    }
}
