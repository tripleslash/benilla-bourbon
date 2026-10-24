//! Quad expansion: one pool of live particles into camera-facing (or XY-plane) quads, the
//! reference's quad writer (`0x7b2a50` head, `0x7b3041` tail), shared by child emitters
//! (`0x7b5dd0`).

use benilla_assets::coords::wow_to_bevy;
use benilla_formats::ParticleEmitterDef;
use bevy::prelude::*;

use super::buffer::{EffectVertex, GpuParticle, ParticlePool};
use super::{rand01, Particle};

/// The 128-entry twinkle noise table, uniform in [0, 1) like the reference's `0xcf58f0` (filled at
/// startup by `0x706790`); a fixed seed, so the distribution matches but not the stream.
static TWINKLE_LUT: std::sync::LazyLock<[f32; 128]> = std::sync::LazyLock::new(|| {
    let mut s = 0xC0FF_EE11u32;
    let mut t = [0.0f32; 128];
    for v in &mut t {
        *v = rand01(&mut s);
    }
    t
});

/// Fill the twinkle table on the calling thread: the expansion reads it from the compute pool,
/// and a worker filling a lazy static while the browser's page thread asks traps the page.
pub(super) fn warm_tables() {
    let _ = TWINKLE_LUT[0];
}

/// Index `(floor(clamp(twinkleSpeed · age, 0, 255)) + phase) & 0x7f` (`0x7b2a86`); the reference's
/// phase is a pointer hash, ours a spawn-time random ([`Particle::phase`]).
fn twinkle_noise(twinkle_speed: f32, age: f32, phase: u32) -> f32 {
    let idx = ((twinkle_speed * age).clamp(0.0, 255.0) as u32).wrapping_add(phase) as usize & 0x7f;
    TWINKLE_LUT[idx]
}

/// `spin·age`, negated when negative on a particle whose 0x20-byte pool slot has address bit 5 set
/// (`0x7b2dda`), so a negative-spin emitter counter-rotates half its cloud. Our bit 5 is
/// [`Particle::phase`]'s, stable per particle where a pool index would not be.
pub(super) fn spin_angle(spin: f32, age: f32, phase: u32) -> f32 {
    let angle = spin * age;
    if angle < 0.0 && phase & 0x20 != 0 {
        -angle
    } else {
        angle
    }
}

/// The camera basis one frame of expansion billboards against.
pub(super) struct CamBasis {
    pub(crate) right: Vec3,
    pub(crate) up: Vec3,
}

/// How a cloud's stored coordinates reach the world (the two modes on [`super::Particle`]).
pub(super) struct DrawFrame {
    pub(crate) anchored: bool,
    /// World mode's ride frame `A`, the reference's `[ebp+8]` at `0x7b3d20`, folded on the
    /// `0x7b3f4f` leg; the identity off a transport. Model mode never reads it (`0x7b3efb`).
    pub(crate) ride: crate::ride_frame::StoredFrame,
    /// The owning model's render alpha, folded into every particle's alpha (`emitter+0x1a8`).
    pub(crate) alpha: f32,
    /// One model unit of half-extent in the stored frame: 1.0 in the world, a UI tile's pixels per
    /// unit (the reference's `768·√(a²+1)` FrameXML units, instance scale excluded, `0x7b2ba6`).
    pub(crate) size_scale: f32,
}

/// One particle's world-space quad centre, shared by [`expand_quads`] and the depth dump.
pub(super) fn particle_center(frame: &DrawFrame, placement: &Transform, p: &Particle) -> Vec3 {
    if frame.anchored {
        // World mode (`0x10` clear): no `rt+0x1fc` fold (`0x7b3f48`), only the ride frame,
        // `A · T · S` (`0x7b3f4f`); off a transport `A` is null, the store world (`0x7b3f95`).
        frame.ride.to_world(p.pos)
    } else {
        // Model mode (`0x10` set): the live emitter matrix, re-applied every frame (`0x7b3efb`).
        placement.transform_point(wow_to_bevy([p.pos.x, p.pos.y, p.pos.z]))
    }
}

/// fork: fsr (FORK.md) — each quad vertex's world displacement since the last world frame, in
/// [`expand_quads`]'s vertex order: every quad a particle draws moves with its centre. A world-mode
/// particle moved by its world velocity over `dt`; a model-mode one also rides the emitter from
/// `prev_placement` to `placement`.
pub(super) fn push_particle_motion(
    def: &ParticleEmitterDef,
    particles: &[Particle],
    frame: &DrawFrame,
    placement: &Transform,
    prev_placement: &Transform,
    dt: f32,
    out: &mut Vec<[f32; 3]>,
) {
    let per = 4 * (usize::from(def.head_tail != 1) + usize::from(def.head_tail >= 1));
    for p in particles {
        if draw_gated(def, p) {
            continue;
        }
        let delta = if frame.anchored {
            frame.ride.dir_to_world(p.vel) * dt
        } else {
            let prev = p.pos - p.vel * dt;
            placement.transform_point(wow_to_bevy(p.pos.to_array()))
                - prev_placement.transform_point(wow_to_bevy(prev.to_array()))
        };
        out.extend(std::iter::repeat_n(delta.to_array(), per));
    }
}

/// The twinklePercent draw gate; must match [`expand_quads`]'s `0x7b2adc` skip.
pub(super) fn draw_gated(def: &ParticleEmitterDef, p: &Particle) -> bool {
    let noise = twinkle_noise(def.twinkle_speed, p.age, p.phase);
    def.twinkle_percent < 1.0 && noise > def.twinkle_percent
}

/// One particle's rendered half-extent; must match the `half` term in [`expand_quads`].
pub(super) fn particle_half(
    def: &ParticleEmitterDef,
    placement: &Transform,
    p: &Particle,
    size_scale: f32,
) -> f32 {
    let noise = twinkle_noise(def.twinkle_speed, p.age, p.phase);
    let u_age = (p.age / p.life).clamp(0.0, 1.0);
    let scale = if def.scale_size_by_instance() {
        placement.scale.x.max(1e-4)
    } else {
        size_scale
    };
    def.over_life.sample(u_age).size * def.twinkle(noise) * scale
}

/// Expand one pool into the shared stream: world-space quads, corners in perimeter order for the
/// lane's `[0,1,2, 0,2,3]` indices.
pub(super) fn expand_quads(
    def: &ParticleEmitterDef,
    particles: &[Particle],
    frame: &DrawFrame,
    placement: &Transform,
    cam: &CamBasis,
    out: &mut Vec<EffectVertex>,
) {
    let anchored = frame.anchored;
    let (cam_right, cam_up) = (cam.right, cam.up);
    // Size takes the instance scale only when flagged (`0x7b2a50`), else the lane's size unit.
    let scale = if def.scale_size_by_instance() {
        placement.scale.x.max(1e-4)
    } else {
        frame.size_scale
    };
    // XY quads (file flag 0x1000, `0x7b41a3`) lie flat in the emitter's XY plane under the live
    // model matrix, its scale included, in both modes. The basis carries the emitter frame's
    // R(+Z, 90°) (`0x719114`) itself, since births take it at emission: X̂ to Ŷ, Ŷ to −X̂.
    let plane_basis = def.xy_quad().then(|| {
        let s = placement.scale.x.max(1e-4);
        (
            placement.rotation * (wow_to_bevy([0.0, 1.0, 0.0]) * s),
            placement.rotation * (wow_to_bevy([-1.0, 0.0, 0.0]) * s),
        )
    });
    // Atlas cell (`0x7b2bd5` head, `0x7b304e` tail): `col = idx & (cols − 1)`, `row = idx >>
    // log2(cols)`, cols a power of two. The row is never clamped: past the last cell V ≥ 1 and
    // the repeat sampler wraps to row 0, which 553 emitters reach at a flipbook's tail.
    let (cols, rows) = (def.tile_cols, def.tile_rows);
    let (inv_cols, inv_rows) = (1.0 / cols as f32, 1.0 / rows as f32);
    let cell_uv = |idx: u16| {
        let cx = f32::from(idx & (cols - 1));
        let cy = f32::from(idx >> cols.trailing_zeros());
        (
            (cx * inv_cols, (cx + 1.0) * inv_cols),
            (cy * inv_rows, (cy + 1.0) * inv_rows),
        )
    };
    for p in particles {
        let noise = twinkle_noise(def.twinkle_speed, p.age, p.phase);
        // The twinklePercent gate (`0x7b2adc`): below 1, a sample above it draws no quad.
        if def.twinkle_percent < 1.0 && noise > def.twinkle_percent {
            continue;
        }
        let u_age = (p.age / p.life).clamp(0.0, 1.0);
        let ol = def.over_life.sample(u_age);
        let (mut rgba, size) = (ol.color, ol.size);
        // The model's render alpha scales alpha alone, never RGB (`0x7b9b10`, the `0x7b9b42` fmul).
        rgba[3] *= frame.alpha;
        // Raw authored gamma RGB: additive stacks sum in gamma like the reference's bytes, and
        // the FFX chain's scene reads apply their 255 saturation (`ffx_glow.wgsl`), not this.
        let center = particle_center(frame, placement, p);
        // `size` is the half-extent (corners ±1.0, `0x7b2d0c`), times the gated twinkle and the
        // flagged instance scale; a twinkle range with min == max burns steady.
        let half = size * def.twinkle(noise) * scale;
        // Deviation: quads keep pool order within a cloud, without the reference's back-to-front
        // sort (`0x7b3a10`), because only an alpha-blended cloud can show the order; between
        // clouds, each draw record's anchor sorts it.
        let mut push_quad = |corners: [Vec3; 4], quv: [[f32; 2]; 4]| {
            for (c, t) in corners.iter().zip(quv) {
                out.push(EffectVertex {
                    pos: c.to_array(),
                    uv: t,
                    color: rgba,
                });
            }
        };
        // Head quad (particleType 0 or 2), drawn first (`0x7b2bc9`): a billboard, or the XY plane.
        if def.head_tail != 1 {
            let ((u0, u1), (v0, v1)) = cell_uv(ol.head_cell);
            let (base_r, base_u) = plane_basis.unwrap_or((cam_right, cam_up));
            // Spin (file +0x198): in-plane by [`spin_angle`] (`0x7b2ddc`); on an XY quad this
            // is the reference's Rodrigues about the plane normal, as `normal × base_r = base_u`.
            let (r, u) = if def.spin != 0.0 {
                let (sa, ca) = spin_angle(def.spin, p.age, p.phase).sin_cos();
                (
                    (base_r * ca + base_u * sa) * half,
                    (base_u * ca - base_r * sa) * half,
                )
            } else {
                (base_r * half, base_u * half)
            };
            push_quad(
                [
                    center - r - u,
                    center + r - u,
                    center + r + u,
                    center - r + u,
                ],
                [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],
            );
        }
        // Tail quad (particleType 1 or 2, `0x7b3041`): |velocity|·tailTime back along the motion,
        // 2·half wide in screen space, U from 0 at the particle to 1 at the tip, a billboard when
        // the velocity is view-parallel; its own flipbook cell (file +0x174..+0x17b, `0x7b3054`).
        if def.head_tail >= 1 {
            let ((u0, u1), (v0, v1)) = cell_uv(ol.tail_cell);
            let vel_world = if anchored {
                frame.ride.dir_to_world(p.vel)
            } else {
                placement.rotation * (placement.scale * wow_to_bevy(p.vel.to_array()))
            };
            let t_eff = if def.tail_clamps_to_age() {
                def.tail_time.min(p.age)
            } else {
                def.tail_time
            };
            let tail = -vel_world * t_eff;
            let (tr, tu) = (tail.dot(cam_right), tail.dot(cam_up));
            let l2 = tr * tr + tu * tu;
            if l2 < 7.7e-4 {
                // Degenerate: the reference's plain-billboard fallback (`0x7b33fa`).
                let (r, u) = (cam_right * half, cam_up * half);
                push_quad(
                    [
                        center - r - u,
                        center + r - u,
                        center + r + u,
                        center - r + u,
                    ],
                    [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],
                );
            } else {
                let inv_l = half / l2.sqrt();
                let perp = (cam_up * tr - cam_right * tu) * inv_l;
                push_quad(
                    [
                        center - perp,
                        center + perp,
                        center + tail + perp,
                        center + tail - perp,
                    ],
                    [[u0, v1], [u0, v0], [u1, v0], [u1, v1]],
                );
            }
        }
    }
}

/// fork: perf — the quads one particle of `def` draws on the GPU: a head, a tail, or both.
pub(super) fn quads_per_particle(def: &ParticleEmitterDef) -> u32 {
    u32::from(def.head_tail != 1) + u32::from(def.head_tail >= 1)
}

/// fork: perf — one pool's [`ParticlePool`] record, the inputs [`expand_quads`] reads of the pool
/// and its frame; `None` for a pool the vertex stage cannot draw (an atlas whose columns are not a
/// power of two, which the CPU writer's cell arithmetic also assumes).
pub(super) fn pool_record(
    def: &ParticleEmitterDef,
    frame: &DrawFrame,
    placement: &Transform,
    cam: &CamBasis,
    committed: Option<[f32; 3]>,
) -> Option<ParticlePool> {
    if def.tile_cols == 0 || !def.tile_cols.is_power_of_two() || def.tile_rows == 0 {
        return None;
    }
    // The stored frame to the world: [`particle_center`]'s two arms as one affine, whose linear
    // part also turns a velocity as the tail's `vel_world` does.
    let center = if frame.anchored {
        frame
            .ride
            .matrix()
            .unwrap_or(bevy::math::Affine3A::IDENTITY)
    } else {
        let w = bevy::math::Mat3A::from_cols(
            wow_to_bevy([1.0, 0.0, 0.0]).into(),
            wow_to_bevy([0.0, 1.0, 0.0]).into(),
            wow_to_bevy([0.0, 0.0, 1.0]).into(),
        );
        let m = bevy::math::Affine3A::from_scale_rotation_translation(
            placement.scale,
            placement.rotation,
            placement.translation,
        );
        bevy::math::Affine3A {
            matrix3: m.matrix3 * w,
            translation: m.translation,
        }
    };
    let row = |i: usize| {
        let m = center.matrix3;
        Vec4::new(m.x_axis[i], m.y_axis[i], m.z_axis[i], center.translation[i])
    };
    let scale = if def.scale_size_by_instance() {
        placement.scale.x.max(1e-4)
    } else {
        frame.size_scale
    };
    let (plane_r, plane_u) = if def.xy_quad() {
        let s = placement.scale.x.max(1e-4);
        (
            placement.rotation * (wow_to_bevy([0.0, 1.0, 0.0]) * s),
            placement.rotation * (wow_to_bevy([-1.0, 0.0, 0.0]) * s),
        )
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    };
    let ol = &def.over_life;
    let ramp = |r: &[benilla_formats::CellRamp; 2]| {
        let (b0, s0) = r[0].base_span();
        let (b1, s1) = r[1].base_span();
        Vec4::new(b0 as f32, s0 as f32, b1 as f32, s1 as f32)
    };
    let bits = u32::from(def.head_tail.min(3))
        | u32::from(def.xy_quad()) << 2
        | u32::from(def.tail_clamps_to_age()) << 3
        | u32::from(committed.is_some()) << 4;
    let light = Vec3::from(committed.unwrap_or([1.0; 3]));
    Some(ParticlePool {
        center_x: row(0),
        center_y: row(1),
        center_z: row(2),
        cam_right: cam.right.extend(scale),
        cam_up: cam.up.extend(frame.alpha),
        plane_r: plane_r.extend(def.spin),
        plane_u: plane_u.extend(def.tail_time),
        color0: Vec4::from_array(ol.color[0]),
        color1: Vec4::from_array(ol.color[1]),
        color2: Vec4::from_array(ol.color[2]),
        scale: Vec4::new(ol.scale[0], ol.scale[1], ol.scale[2], ol.mid),
        head: ramp(&ol.head_cells),
        tail: ramp(&ol.tail_cells),
        atlas: Vec4::new(
            ol.repeat[0],
            ol.repeat[1],
            f32::from(def.tile_cols),
            f32::from(def.tile_rows),
        ),
        mode: Vec4::new(bits as f32, light.x, light.y, light.z),
        spare: Vec4::ZERO,
    })
}

/// fork: perf — pack `particles` for the GPU expansion: the twinkle gate drops a particle here
/// ([`draw_gated`]), and its size multiplier rides the instance, so the vertex stage needs no
/// table.
pub(super) fn pack_gpu_particles(
    def: &ParticleEmitterDef,
    particles: &[Particle],
    out: &mut Vec<GpuParticle>,
) {
    for p in particles {
        let noise = twinkle_noise(def.twinkle_speed, p.age, p.phase);
        if def.twinkle_percent < 1.0 && noise > def.twinkle_percent {
            continue;
        }
        out.push(GpuParticle {
            pos: p.pos.to_array(),
            age: p.age,
            vel: p.vel.to_array(),
            u_age: (p.age / p.life).clamp(0.0, 1.0),
            twinkle: def.twinkle(noise),
            phase: p.phase,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_quads, spin_angle, CamBasis, DrawFrame, Particle};
    use bevy::prelude::{Quat, Transform, Vec3};

    /// The `0x7b2dda` negate: only a negative angle on a bit-5 particle flips.
    #[test]
    fn negative_spin_counter_rotates_the_bit5_half() {
        assert_eq!(spin_angle(-3.0, 0.5, 0x20), 1.5, "bit 5 set: negated");
        assert_eq!(spin_angle(-3.0, 0.5, 0x1f), -1.5, "bit 5 clear: kept");
        assert_eq!(
            spin_angle(3.0, 0.5, 0x20),
            1.5,
            "positive spin: never negated"
        );
        assert_eq!(spin_angle(3.0, 0.5, 0x1f), 1.5);
        assert_eq!(spin_angle(0.0, 0.5, 0xff), 0.0);
    }

    /// The reference folds `emitter+0x1a8`, the model's `CM2Model+0x19c`, into the alpha byte alone
    /// (`0x7b9b10`, the `0x7b9b42` fmul).
    #[test]
    fn the_models_render_alpha_scales_particle_alpha_and_nothing_else() {
        let mut def = crate::particles::tests::plain_def();
        def.over_life.color = [[0.8, 0.4, 0.2, 0.5]; 3];
        let pool = [Particle {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            age: 0.0,
            life: 1.0,
            phase: 0,
            fresh: false,
            quat: Quat::IDENTITY,
            angvel: Vec3::ZERO,
        }];
        let cam = CamBasis {
            right: Vec3::X,
            up: Vec3::Y,
        };
        let shoot = |alpha| {
            let frame = DrawFrame {
                anchored: true,
                ride: crate::ride_frame::StoredFrame::default(),
                alpha,
                size_scale: 1.0,
            };
            let mut out = Vec::new();
            expand_quads(&def, &pool, &frame, &Transform::IDENTITY, &cam, &mut out);
            out[0].color
        };

        let opaque = shoot(1.0);
        assert_eq!(opaque, [0.8, 0.4, 0.2, 0.5], "no fade: the authored ramp");
        let half = shoot(0.5);
        assert_eq!(
            [half[0], half[1], half[2]],
            [0.8, 0.4, 0.2],
            "RGB is untouched by the model alpha"
        );
        assert!((half[3] - 0.25).abs() < 1e-6, "alpha halves: {half:?}");
        assert_eq!(shoot(0.0)[3], 0.0, "an invisible model draws nothing");
    }
}
