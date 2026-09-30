//! fork: perf — a scripted flight for the engine binary. `WOW_WORLDVIEW_FLY=<x,y,z>;<x,y,z>[;…]`
//! (WoW coordinates, yd) flies the viewer's camera along the waypoints and back at a steady
//! `WOW_WORLDVIEW_FLY_SPEED` yd/s (default 28, a fast flying mount), facing its heading and looking
//! `WOW_WORLDVIEW_PITCH` degrees down (default −10). With `WOW_PERF` it measures a fly-over — the
//! streaming, culling and draw load a moving camera meets — the same way in any two zones, where a
//! parked camera measures only one view.

use bevy::prelude::*;

use super::ViewCam;
use benilla_assets::coords::wow_to_bevy;

#[derive(Resource)]
struct FlyPath {
    points: Vec<Vec3>,
    speed: f32,
    pitch: f32,
    /// Yards flown since the start.
    flown: f32,
}

impl FlyPath {
    fn from_env() -> Option<Self> {
        let v = crate::launch_env::live_var("WOW_WORLDVIEW_FLY")?;
        let points: Vec<Vec3> = v
            .split(';')
            .filter_map(|p| {
                let c: Vec<f32> = p.split(',').filter_map(|c| c.trim().parse().ok()).collect();
                (c.len() == 3).then(|| wow_to_bevy([c[0], c[1], c[2]]))
            })
            .collect();
        if points.len() < 2 {
            warn!("worldview: WOW_WORLDVIEW_FLY needs two or more x,y,z waypoints");
            return None;
        }
        let num = |k: &str, d: f32| {
            crate::launch_env::live_var(k)
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(d)
        };
        Some(Self {
            points,
            speed: num("WOW_WORLDVIEW_FLY_SPEED", 28.0),
            pitch: num("WOW_WORLDVIEW_PITCH", -10.0).to_radians(),
            flown: 0.0,
        })
    }

    /// The eye and its heading `s` yards along the path, which runs to its end and back.
    fn at(&self, s: f32) -> (Vec3, Vec3) {
        let legs: Vec<f32> = self
            .points
            .windows(2)
            .map(|w| w[0].distance(w[1]))
            .collect();
        let total: f32 = legs.iter().sum::<f32>().max(1e-3);
        let t = s.rem_euclid(2.0 * total);
        let (mut d, back) = if t > total {
            (2.0 * total - t, true)
        } else {
            (t, false)
        };
        for (i, len) in legs.iter().enumerate() {
            if d <= *len || i + 1 == legs.len() {
                let (a, b) = (self.points[i], self.points[i + 1]);
                let dir = (b - a).normalize_or(Vec3::NEG_Z);
                let eye = a + dir * d.min(*len);
                return (eye, if back { -dir } else { dir });
            }
            d -= len;
        }
        (self.points[0], Vec3::NEG_Z)
    }
}

fn fly_path(
    time: Res<Time>,
    mut path: ResMut<FlyPath>,
    mut cam: Query<(&mut Transform, &mut ViewCam)>,
) {
    let Ok((mut xf, mut view)) = cam.single_mut() else {
        return;
    };
    path.flown += path.speed * time.delta_secs();
    let (eye, dir) = path.at(path.flown);
    // The controller's own yaw convention, so a hand on the mouse picks up where the path is.
    view.yaw = dir.x.atan2(dir.z) + std::f32::consts::PI;
    view.pitch = path.pitch;
    xf.translation = eye;
    xf.rotation = Quat::from_euler(EulerRot::YXZ, view.yaw, view.pitch, 0.0);
}

pub(super) fn plugin(app: &mut App) {
    let Some(path) = FlyPath::from_env() else {
        return;
    };
    info!(
        "worldview: flying {} waypoints at {} yd/s",
        path.points.len(),
        path.speed
    );
    app.insert_resource(path).add_systems(
        Update,
        fly_path
            .after(super::live_pose)
            .in_set(crate::schedule::WorldStage::Input),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path runs out and back, and the heading turns round at the far end.
    #[test]
    fn the_path_runs_out_and_back() {
        let path = FlyPath {
            points: vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
            speed: 1.0,
            pitch: 0.0,
            flown: 0.0,
        };
        let (eye, dir) = path.at(4.0);
        assert!((eye - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-4);
        assert!((dir - Vec3::X).length() < 1e-4);
        let (eye, dir) = path.at(16.0);
        assert!((eye - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-4);
        assert!((dir + Vec3::X).length() < 1e-4);
    }
}
