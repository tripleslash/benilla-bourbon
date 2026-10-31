//! `TransportAnimation.dbc`, the keyframe paths of type-11 transports (elevators), and the
//! reference's cycle evaluator over them, in WoW coordinates. The server sends one anchor, the
//! create block's `UPDATE_FLAG_TRANSPORT` clock (time since creation, modulo the period), and the
//! client animates the car: bracket the keyframes around `(anchor + elapsed) % period`, lerp their
//! offsets, rotate by the spawn's `GAMEOBJECT_ROTATION` and add the spawn position (`0x5f6280`,
//! ticked by `0x5f5f10`), as vmangos's `ElevatorTransport::Update` does (`Transport.cpp:396-435`).
//! `TransportID` is the `gameobject_template` entry; `SequenceID`, the car's animation per span, is
//! not read.

use std::collections::HashMap;

use anyhow::{Context, Result};
use benilla_dbc::{FieldType, Schema, SchemaField};

use crate::dbc::{f32_at, parse, u32_at};
use crate::Chain;

const TRANSPORT_ANIMATION: &str = "DBFilesClient\\TransportAnimation.dbc";

/// One `TransportAnimation.dbc` row, a keyframe on a type-11 transport's local path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevatorKeyframe {
    /// Cumulative cycle time in ms; the last frame's is the period.
    pub time_ms: u32,
    /// Offset from the spawn point in WoW axes, before the spawn rotation.
    pub pos: [f32; 3],
}

/// `TransportAnimation.dbc` grouped by `TransportID`, each path's frames sorted by `TimeIndex`.
pub struct ElevatorPaths {
    by_entry: HashMap<u32, Vec<ElevatorKeyframe>>,
}

impl ElevatorPaths {
    /// A template entry's keyframes in time order: at least two, the first at `time_ms == 0`.
    pub fn entry(&self, template_entry: u32) -> Option<&[ElevatorKeyframe]> {
        self.by_entry.get(&template_entry).map(Vec::as_slice)
    }

    /// The number of transport entries with a path.
    pub fn len(&self) -> usize {
        self.by_entry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_entry.is_empty()
    }
}

fn schema() -> Schema {
    let mut s = Schema::new("TransportAnimation");
    for name in ["ID", "TransportID", "TimeIndex"] {
        s.add_field(SchemaField::new(name, FieldType::UInt32));
    }
    for name in ["PosX", "PosY", "PosZ"] {
        s.add_field(SchemaField::new(name, FieldType::Float32));
    }
    s.add_field(SchemaField::new("SequenceID", FieldType::UInt32));
    s
}

/// Read `TransportAnimation.dbc` off the patch chain. Deviation: a path that cannot drive a cycle
/// is dropped and its GO stays at its spawn point, because the reference faults or hangs: a last
/// frame at 0 divides by zero (`0x5f62a4`), and a first frame past 0, one frame among them, leaves
/// a time before it in no bracket, which `0x5f62bc`-`0x5f62e7` seek forever.
pub fn load_elevator_paths(chain: &mut Chain) -> Result<ElevatorPaths> {
    let bytes = chain
        .read_file(TRANSPORT_ANIMATION)
        .context("reading TransportAnimation.dbc")?;
    let rs = parse(&bytes, schema(), "TransportAnimation")?;
    let mut by_entry: HashMap<u32, Vec<ElevatorKeyframe>> = HashMap::new();
    for r in rs.records() {
        let Some(entry) = u32_at(r, 1) else { continue };
        let Some(time_ms) = u32_at(r, 2) else {
            continue;
        };
        let (Some(x), Some(y), Some(z)) = (f32_at(r, 3), f32_at(r, 4), f32_at(r, 5)) else {
            continue;
        };
        by_entry.entry(entry).or_default().push(ElevatorKeyframe {
            time_ms,
            pos: [x, y, z],
        });
    }
    for frames in by_entry.values_mut() {
        frames.sort_by_key(|f| f.time_ms);
    }
    by_entry.retain(|_, f| f.len() >= 2 && f[0].time_ms == 0 && f[f.len() - 1].time_ms > 0);
    Ok(ElevatorPaths { by_entry })
}

/// The cycle period in ms, the last keyframe's time (`0x5f6280`'s modulus).
pub fn elevator_period_ms(frames: &[ElevatorKeyframe]) -> u32 {
    frames.last().map_or(1, |f| f.time_ms).max(1)
}

/// The reference's type-11 evaluator (`0x5f6280`): the car's world position at `cycle_ms`, from
/// the spawn position and its `GAMEOBJECT_ROTATION` quaternion `(x, y, z, w)`, and whether the
/// bracketing span moves. Bit-exact to it: `A·(1−f) + B·f` on the stored floats, turned through the
/// spawn's 3×3 matrix, with the x87 at 53 bits and each store rounding to `f32` where the
/// reference stores (benilla-re's `gameobject_path_eval`).
pub fn elevator_sample(
    frames: &[ElevatorKeyframe],
    spawn_pos: [f32; 3],
    spawn_quat: [f32; 4],
    cycle_ms: u32,
) -> ([f32; 3], bool) {
    debug_assert!(frames.len() >= 2 && frames[0].time_ms == 0);
    let period = elevator_period_ms(frames);
    let target = cycle_ms % period;

    // The reference's bracket, found by search rather than its ring cursor.
    let hi = frames.partition_point(|f| f.time_ms <= target);
    let (prev, next) = (frames[hi - 1], frames[hi.min(frames.len() - 1)]);

    // The bracket is strict, so the span is never empty; the guard keeps a malformed one finite.
    let offset = if next.time_ms == prev.time_ms {
        turn_by_quat(lerp_nodes(prev.pos, prev.pos, 0.0), spawn_quat)
    } else {
        // A `fild` of the zero-extended difference, `fidiv` by the signed span, stored as a float.
        let frac = (f64::from(target - prev.time_ms)
            / f64::from(next.time_ms.wrapping_sub(prev.time_ms) as i32)) as f32;
        turn_by_quat(lerp_nodes(prev.pos, next.pos, frac), spawn_quat)
    };
    // The tick adds the offset to the fields' position at 53 bits and stores it.
    let world = std::array::from_fn(|i| (f64::from(spawn_pos[i]) + f64::from(offset[i])) as f32);
    (world, prev.pos != next.pos)
}

/// `A·(1−f) + B·f` with the reference's stores: x stores the sum; y stores only `(1−f)·A`, its sum
/// kept at 53 bits; z stores both products, its sum kept. `x` returns as a float and `y`, `z` as
/// the registers the turn reads.
fn lerp_nodes(a: [f32; 3], b: [f32; 3], frac: f32) -> (f32, f64, f64) {
    let (f, g) = (f64::from(frac), 1.0 - f64::from(frac));
    let x = (g * f64::from(a[0]) + f * f64::from(b[0])) as f32;
    let y = f64::from((g * f64::from(a[1])) as f32) + f * f64::from(b[1]);
    let z = f64::from((g * f64::from(a[2])) as f32) + f64::from((f * f64::from(b[2])) as f32);
    (x, y, z)
}

/// The offset turned through the spawn's 3×3 (`0x5f630c`-`0x5f6445`): `2qz`, the products with
/// `qw` and `qx` and `2qz·qz` stored as floats, `2qx`, `2qy`, `2qy·qy` and `2qz·qy` kept; each
/// entry stored (`0x5f8d20`), and `out[r] = m[r]·x + (m[r+3]·y + m[r+6]·z)`, stored.
fn turn_by_quat((x, y, z): (f32, f64, f64), q: [f32; 4]) -> [f32; 3] {
    let [qx, qy, qz, qw] = q.map(f64::from);
    let (two_x, two_y) = (qx + qx, qy + qy);
    let two_z = f64::from((qz + qz) as f32);
    let s = |v: f64| f64::from(v as f32);
    let (xw, yw, zw) = (s(two_x * qw), s(two_y * qw), s(two_z * qw));
    let (xx, yx, zx) = (s(two_x * qx), s(two_y * qx), s(two_z * qx));
    let (yy, zy, zz) = (two_y * qy, two_z * qy, s(two_z * qz));
    let m = [
        s(1.0 - (zz + yy)),
        s(yx + zw),
        s(zx - yw),
        s(yx - zw),
        s(1.0 - (zz + xx)),
        s(zy + xw),
        s(zx + yw),
        s(zy - xw),
        s(1.0 - (yy + xx)),
    ];
    let x = f64::from(x);
    std::array::from_fn(|r| (m[r] * x + (m[r + 3] * y + m[r + 6] * z)) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames() -> Vec<ElevatorKeyframe> {
        vec![
            ElevatorKeyframe {
                time_ms: 0,
                pos: [0.0, 0.0, 0.0],
            },
            ElevatorKeyframe {
                time_ms: 5000,
                pos: [0.0, 0.0, 0.0],
            },
            ElevatorKeyframe {
                time_ms: 10000,
                pos: [0.0, 0.0, -10.0],
            },
            ElevatorKeyframe {
                time_ms: 30000,
                pos: [0.0, 0.0, 0.0],
            },
        ]
    }

    const IDENT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    #[test]
    fn dwell_then_lerp_then_wrap() {
        let f = frames();
        let base = [100.0, 200.0, 50.0];
        let (p, moving) = elevator_sample(&f, base, IDENT, 2500);
        assert_eq!(p, base);
        assert!(!moving);
        // 7500 is halfway through the 5000 to 10000 span: z offset -5.
        let (p, moving) = elevator_sample(&f, base, IDENT, 7500);
        assert_eq!(p, [100.0, 200.0, 45.0]);
        assert!(moving);
        let (p0, _) = elevator_sample(&f, base, IDENT, 0);
        let (pw, _) = elevator_sample(&f, base, IDENT, 30000);
        assert_eq!(p0, pw);
        let (p, moving) = elevator_sample(&f, base, IDENT, 29999);
        assert!(moving);
        assert!((p[2] - 50.0).abs() < 0.01, "z = {}", p[2]);
    }

    /// A 90° yaw (1.12 transport spawns only yaw) turns `(x, y)` into `(-y, x)`.
    #[test]
    fn spawn_yaw_rotates_the_local_offset() {
        let half = std::f32::consts::FRAC_PI_4; // 90°/2
        let q = [0.0, 0.0, half.sin(), half.cos()];
        let f = vec![
            ElevatorKeyframe {
                time_ms: 0,
                pos: [10.0, 0.0, 0.0],
            },
            ElevatorKeyframe {
                time_ms: 1000,
                pos: [10.0, 0.0, 0.0],
            },
        ];
        let (p, _) = elevator_sample(&f, [0.0; 3], q, 500);
        assert!(
            (p[0] - 0.0).abs() < 1e-4 && (p[1] - 10.0).abs() < 1e-4,
            "{p:?}"
        );
    }

    /// The 5875 table: layout and load guarantees on three real paths.
    #[test]
    fn real_transport_animation_layout_sanity() {
        let data = crate::wow_data_or_skip!();
        let mut chain = Chain::open(&data).expect("open patch chain");
        let paths = load_elevator_paths(&mut chain).expect("load TransportAnimation.dbc");
        // Thunder Bluff's Mesa Elevator cars are templates 4170 and 4171, Undercity's 152614.
        let top = paths.entry(4170).expect("Mesa Elevator 4170");
        assert_eq!(elevator_period_ms(top), 30033);
        assert_eq!(top[0].time_ms, 0);
        assert_eq!(top[0].pos, [0.0, 0.0, 0.0]);
        assert!(top.windows(2).all(|w| w[0].time_ms <= w[1].time_ms));
        // The car moves on z only (x and y are ~2e-6 yd of noise), 61.24 yd down.
        assert!(top
            .iter()
            .all(|f| f.pos[0].abs() < 1e-3 && f.pos[1].abs() < 1e-3));
        assert!((top.iter().map(|f| f.pos[2]).fold(0.0f32, f32::min) + 61.244).abs() < 1e-3);
        let bottom = paths.entry(4171).expect("Mesa Elevator 4171");
        assert_eq!(elevator_period_ms(bottom), 30000);
        assert!(paths.entry(152614).is_some(), "Undercity elevator");
        assert!(paths.entry(999_999).is_none());
    }
}
