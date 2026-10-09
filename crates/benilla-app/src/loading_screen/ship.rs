//! The transport loading screen: the world map with a dotted route from the dock we leave to the
//! dock we reach, in place of the map's art, bar and tip. `SMSG_TRANSFER_PENDING` with a transport
//! block raises it (`0x401900` → `0x4072c0`) when we ride that transport and its path's leg has a
//! `LoadingScreenTaxiSplines.dbc` row; any miss leaves the plain screen. The transport's own tick
//! raises it too, when the transport under us moves onto its next leg without leaving the map
//! (`0x5f5461` → `0x407de0`), as the Grom'gol to Undercity zeppelin does at its teleport node.
//!
//! The route (`0x4075a0`) is a Catmull-Rom curve through the departure dock, the row's bend points
//! and the arrival dock, drawn as dots a fixed arc length apart. The dots are the progress bar:
//! dot `i` of `n` shows once the load fraction reaches `i/(n−1)` (`0x406fb0`). Positions are the
//! reference's: fractions of the 4:3 loading area, y up.

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use benilla_assets::WorldAssets;
use benilla_formats::{
    load_ship_splines, load_world_map_continent_catalog, ship_legs, Chain, ShipSplines,
    TaxiPathNode, TaxiPathNodes, WorldMapContinentCatalog,
};
use benilla_world::map_proj::{world_uv, WorldProj};

use super::LoadingScreen;
use crate::glue::art::tc_rect;
use crate::go_templates::GameObjectTemplates;
use crate::net::Guid;

/// Dots per unit of route length, `1/0.015` (`0x7ffd80`); the count truncates, with a floor of 1.
const DOTS_PER_UNIT: f32 = 66.666_67;
/// A dot's half extent (`0x8828b0`) and a dock marker's (`0x7ffdac`, `0x7ffda8`).
const DOT_HALF: Vec2 = Vec2::new(0.01, 0.015);
const MARKER_HALF: Vec2 = Vec2::new(0.02, 0.0265);
/// `DynamicElements.blp`'s quadrants as `[u0, u1, v0, v1]`: the dot (`0x882bac`), the bullseye on
/// the dock we leave and the cross on the dock we reach (`0x407900`).
const DOT_TC: [f32; 4] = [0.0, 0.5, 0.5, 1.0];
const DEPART_TC: [f32; 4] = [0.5, 1.0, 0.0, 0.5];
const ARRIVE_TC: [f32; 4] = [0.0, 0.5, 0.0, 0.5];
const ELEMENTS: &str = "Interface\\Glues\\LoadingScreens\\DynamicElements.blp";
/// The world sheet, 1002×668 over a 4×3 grid of 256 px tiles, `World1` to `World12` row by row,
/// the last column and row cropped (`0x4081e0`).
const SHEET: UVec2 = UVec2::new(1002, 668);
const TILE: u32 = 256;
const TILE_COLS: u32 = 4;
const TILE_ROWS: u32 = 3;
/// A segment's length is this many chords of its curve (`0x4537b8`).
const LENGTH_STEPS: u32 = 20;

/// A drawn route: the two dock markers and the dots between them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ShipRoute {
    depart: Vec2,
    arrive: Vec2,
    dots: Vec<Vec2>,
}

/// One Catmull-Rom span, from `p[1]` at 0 to `p[2]` at 1 (the basis at `0xb05e10`).
fn span_at(p: &[Vec2], t: f32) -> Vec2 {
    p[0] * (((-0.5 * t + 1.0) * t - 0.5) * t)
        + p[1] * ((1.5 * t - 2.5) * t * t + 1.0)
        + p[2] * (((-1.5 * t + 2.0) * t + 0.5) * t)
        + p[3] * ((0.5 * t - 0.5) * t * t)
}

/// A span's length as the reference caches it (`0x453760`): chords at steps of 0.05.
fn span_length(p: &[Vec2]) -> f32 {
    let step = 1.0 / LENGTH_STEPS as f32;
    let mut at = span_at(p, 0.0);
    let mut t = step;
    let mut length = 0.0;
    for _ in 0..LENGTH_STEPS {
        let next = span_at(p, t);
        length += at.distance(next);
        at = next;
        t += step;
    }
    length
}

impl ShipRoute {
    /// The route from `depart` through `bends` to `arrive` (`0x4075a0`). Each dock is doubled so
    /// the curve starts and ends on it.
    fn new(depart: Vec2, arrive: Vec2, bends: &[[f32; 2]]) -> Self {
        let mut points = Vec::with_capacity(bends.len() + 4);
        points.extend([depart, depart]);
        points.extend(bends.iter().map(|&b| Vec2::from(b)));
        points.extend([arrive, arrive]);
        let lengths: Vec<f32> = points.windows(4).map(span_length).collect();
        let total: f32 = lengths.iter().sum();

        // The point at arc fraction `t` (`0x453390` in arc-length mode): the ends are the docks,
        // and between them the span holding the arc length, linear in its own parameter
        // (`0x453840`).
        let at = |t: f32| {
            if t <= 0.0 {
                return depart;
            }
            if t >= 1.0 {
                return arrive;
            }
            let last = lengths.len() - 1;
            if last == 0 {
                return span_at(&points, t);
            }
            let target = t * total;
            let (mut span, mut before) = (0, 0.0);
            while span < last && before + lengths[span] <= target {
                before += lengths[span];
                span += 1;
            }
            span_at(&points[span..], (target - before) / lengths[span])
        };

        let count = ((total * DOTS_PER_UNIT) as i32).max(1);
        let step = 1.0 / count as f32;
        let mut t = 0.0f32;
        let dots = (0..=count)
            .map(|_| {
                let dot = at(t.clamp(0.0, 1.0));
                t += step;
                dot
            })
            .collect();
        Self {
            depart,
            arrive,
            dots,
        }
    }

    /// How many dots a load fraction shows (`0x406fb0`): those with `i/(n−1) <= progress`.
    fn dots_shown(&self, progress: f32) -> usize {
        let last = self.dots.len().saturating_sub(1) as f32;
        (0..self.dots.len())
            .take_while(|&i| i as f32 / last <= progress)
            .count()
    }
}

/// One leg of a transport path as the lookup reads it (`0x407364`).
struct RouteLeg {
    /// The map of the leg's first stop, matched against the map the transfer leaves.
    from_map: u32,
    /// The route to the next leg, when the splines table and both continents have it.
    route: Option<ShipRoute>,
}

/// Every transport path's legs with their routes, by `TaxiPath.dbc` id.
#[derive(Resource, Default)]
pub(crate) struct ShipRoutes {
    by_path: HashMap<u32, Vec<Option<RouteLeg>>>,
}

impl ShipRoutes {
    /// Build the table from the taxi nodes the transports already hold and the chain's two other
    /// tables; a table that does not read leaves no routes, so every transfer shows its map's art.
    pub(crate) fn load(nodes: &TaxiPathNodes, chain: &mut Chain) -> Self {
        let splines = match load_ship_splines(chain) {
            Ok(s) => s,
            Err(e) => {
                warn!("transport routes unavailable, plain loading screens: {e:#}");
                return Self::default();
            }
        };
        let continents = match load_world_map_continent_catalog(chain) {
            Ok(c) => c,
            Err(e) => {
                warn!("transport routes unavailable, plain loading screens: {e:#}");
                return Self::default();
            }
        };
        let routes = Self::build(nodes, &splines, &continents);
        info!("transport routes: {} paths drawn", routes.by_path.len());
        routes
    }

    fn build(
        nodes: &TaxiPathNodes,
        splines: &ShipSplines,
        continents: &WorldMapContinentCatalog,
    ) -> Self {
        // A stop on the world sheet (`0x4a72b0`), y flipped up as the screen draws it.
        let on_sheet = |stop: &TaxiPathNode| {
            let c = continents.get(stop.map_id)?;
            let proj = WorldProj {
                offset_u: c.offset_x,
                offset_v: c.offset_y,
                scale: c.scale,
            };
            let (u, v) = world_uv(proj, stop.pos[0], stop.pos[1]);
            Some(Vec2::new(u, 1.0 - v))
        };
        let mut by_path = HashMap::new();
        for (path_id, path) in nodes.paths() {
            let legs = ship_legs(path);
            let routed: Vec<Option<RouteLeg>> = legs
                .iter()
                .enumerate()
                .map(|(i, leg)| {
                    let leg = leg.as_ref()?;
                    let route = (|| {
                        let next = legs[(i + 1) % legs.len()].as_ref()?;
                        let bends = splines.points(path_id, i as u32)?;
                        Some(ShipRoute::new(
                            on_sheet(&leg.last_stop)?,
                            on_sheet(&next.first_stop)?,
                            bends,
                        ))
                    })();
                    Some(RouteLeg {
                        from_map: leg.first_stop.map_id,
                        route,
                    })
                })
                .collect();
            if routed.iter().flatten().any(|l| l.route.is_some()) {
                by_path.insert(path_id, routed);
            }
        }
        Self { by_path }
    }

    /// The route a transport on `path_id` draws when it leaves `from_map` (`0x407364`): that of
    /// the first leg whose first stop is on that map. The search ends at a leg with no stop.
    pub(crate) fn route(&self, path_id: u32, from_map: u32) -> Option<&ShipRoute> {
        self.by_path
            .get(&path_id)?
            .iter()
            .map_while(Option::as_ref)
            .find(|leg| leg.from_map == from_map)?
            .route
            .as_ref()
    }

    /// The route out of `path_id`'s leg `leg`, which the tick's entry names outright (`0x407de0`
    /// hands its index straight to `0x4073f0`).
    pub(crate) fn leg_route(&self, path_id: u32, leg: usize) -> Option<&ShipRoute> {
        self.by_path
            .get(&path_id)?
            .get(leg)?
            .as_ref()?
            .route
            .as_ref()
    }
}

/// The route screen's textures; the reference loads them at the raise and takes the plain screen
/// when any is missing (`0x4074b0`).
#[derive(Clone)]
struct ShipArt {
    /// `DynamicElements.blp` and its size in texels.
    elements: (Handle<Image>, Vec2),
    /// `World1` to `World12`, each with its size.
    tiles: Vec<(Handle<Image>, Vec2)>,
}

impl ShipArt {
    fn load(assets: &mut WorldAssets, images: &mut Assets<Image>) -> Option<Self> {
        // Blocking, like the backdrop: the frame it stalls is a loading frame.
        let mut sized = |path: &str| {
            let handle = assets.sprite_texture_blocking(path, images)?;
            let size = images.get(&handle)?.size_f32();
            Some((handle, size))
        };
        let elements = sized(ELEMENTS)?;
        let tiles = (1..=TILE_COLS * TILE_ROWS)
            .map(|n| sized(&format!("Interface\\WorldMap\\World\\World{n}.blp")))
            .collect::<Option<Vec<_>>>()?;
        Some(Self { elements, tiles })
    }
}

/// A route on the screen, held from its raise to the dismiss.
pub(super) struct ShipScreen {
    route: ShipRoute,
    art: ShipArt,
    /// Tells one raise's route from the next's, for [`paint_ship_route`].
    serial: u32,
}

impl LoadingScreen {
    /// Put `route` on the screen, or take the route off when its art is missing.
    fn show_route(
        &mut self,
        route: Option<&ShipRoute>,
        assets: Option<&mut WorldAssets>,
        images: &mut Assets<Image>,
    ) {
        let art = assets.and_then(|assets| ShipArt::load(assets, images));
        self.ship = route.zip(art).map(|(route, art)| {
            self.ship_serial = self.ship_serial.wrapping_add(1);
            ShipScreen {
                route: route.clone(),
                art,
                serial: self.ship_serial,
            }
        });
    }

    /// The capture instrument (`WOW_CAPTURE=loading-ship`): hold up the route screen of `path_id`
    /// leaving `from_map`, with no ride behind it, and never clear.
    pub(crate) fn hold_ship_for_capture(
        &mut self,
        routes: Option<&ShipRoutes>,
        (path_id, from_map): (u32, u32),
        assets: Option<&mut WorldAssets>,
        images: &mut Assets<Image>,
    ) {
        self.active = true;
        self.blackout = false;
        self.awaiting_snap = false;
        self.map = None;
        self.held = true;
        let route = routes.and_then(|r| r.route(path_id, from_map));
        self.show_route(route, assets, images);
    }
}

/// What the transfer's raise needs to find its route.
#[derive(SystemParam)]
pub(super) struct ShipLookup<'w, 's> {
    routes: Option<Res<'w, ShipRoutes>>,
    templates: Res<'w, GameObjectTemplates>,
    guids: Query<'w, 's, &'static Guid>,
}

impl ShipLookup<'_, '_> {
    /// The route a transfer on a transport asks for (`0x4072c0`): `transport` is its entry and
    /// the map it leaves, `deck` the transport we ride. `None` unless we ride the one the server
    /// named; else that leg's route, or `Some(None)` when the leg draws none.
    fn route(
        &self,
        deck: Option<Entity>,
        transport: Option<(u32, u32)>,
    ) -> Option<Option<&ShipRoute>> {
        let (entry, from_map) = transport?;
        let guid = self.guids.get(deck?).ok()?;
        if benilla_protocol::guid::entry(guid.0) != Some(entry) {
            return None;
        }
        let path_id = self.templates.get(guid.0).and_then(|t| t.mo_transport);
        Some(
            path_id
                .zip(self.routes.as_ref())
                .and_then(|(mo, routes)| routes.route(mo.taxi_path_id, from_map)),
        )
    }

    /// The route the tick's entry asks for (`0x407de0`): `deck` is the transport we ride, `left`
    /// the leg it left. `None` when the transport is gone; else that leg's route, or `Some(None)`
    /// when it draws none.
    fn leg_route(&self, deck: Entity, left: usize) -> Option<Option<&ShipRoute>> {
        let guid = self.guids.get(deck).ok()?;
        let path_id = self.templates.get(guid.0).and_then(|t| t.mo_transport);
        Some(
            path_id
                .zip(self.routes.as_ref())
                .and_then(|(mo, routes)| routes.leg_route(mo.taxi_path_id, left)),
        )
    }

    /// Give the screen the route of the leg the transport under us left; [`Self::board`]'s twin
    /// for the tick's raise.
    pub(super) fn board_leg(
        &self,
        screen: &mut LoadingScreen,
        deck: Entity,
        left: usize,
        assets: Option<&mut WorldAssets>,
        images: &mut Assets<Image>,
    ) {
        let Some(route) = self.leg_route(deck, left) else {
            return;
        };
        screen.show_route(route, assets, images);
        info!(
            "loading screen: the transport under us left leg {left} — {}",
            match screen.ship {
                Some(_) => "its route",
                None => "no drawn route, plain screen",
            }
        );
    }

    /// Give the screen the route of a transfer's raise. Riding the transport the server named,
    /// the screen's route becomes that leg's, or none when the leg draws no route or the art is
    /// missing; any other transfer leaves the screen's route as it is.
    pub(super) fn board(
        &self,
        screen: &mut LoadingScreen,
        deck: Option<Entity>,
        transport: Option<(u32, u32)>,
        assets: Option<&mut WorldAssets>,
        images: &mut Assets<Image>,
    ) {
        let Some(route) = self.route(deck, transport) else {
            return;
        };
        screen.show_route(route, assets, images);
        info!(
            "loading screen: riding transport {transport:?} (entry, map left) — {}",
            match screen.ship {
                Some(_) => "its route",
                None => "no drawn route, plain screen",
            }
        );
    }
}

/// The route's layer, the loading area's last child: it covers the art, the tip and the bar,
/// which the reference does not draw under a route (`0x406cd0`).
#[derive(Component, Default)]
pub(super) struct ShipLayer {
    /// The [`ShipScreen::serial`] its children were spawned for.
    painted: Option<u32>,
}

/// A route dot, by its place along the route.
#[derive(Component)]
pub(super) struct ShipDot(usize);

/// The layer's own bundle: the whole area, black where no tile reaches, hidden until a route.
pub(super) fn layer_bundle() -> impl Bundle {
    (
        ShipLayer::default(),
        Node {
            position_type: PositionType::Absolute,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        BackgroundColor(Color::BLACK),
        Visibility::Hidden,
    )
}

/// A textured quad in the layer: `min` and `size` in fractions of the area from its top left,
/// `tc` the texture's `[u0, u1, v0, v1]`.
fn quad(min: Vec2, size: Vec2, art: &(Handle<Image>, Vec2), tc: [f32; 4]) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(min.x * 100.0),
            top: Val::Percent(min.y * 100.0),
            width: Val::Percent(size.x * 100.0),
            height: Val::Percent(size.y * 100.0),
            ..default()
        },
        ImageNode {
            image: art.0.clone(),
            rect: Some(tc_rect(art.1, tc)),
            image_mode: NodeImageMode::Stretch,
            ..default()
        },
    )
}

/// A quad centred on a route point, whose y runs up.
fn centred(at: Vec2, half: Vec2, art: &(Handle<Image>, Vec2), tc: [f32; 4]) -> impl Bundle {
    quad(
        Vec2::new(at.x - half.x, 1.0 - at.y - half.y),
        half * 2.0,
        art,
        tc,
    )
}

/// Keep the layer in step with the screen's route: spawn the sheet, the dots and the two markers
/// when a route comes up, drop them when it goes, and show the dots the load has reached.
pub(super) fn paint_ship_route(
    mut commands: Commands,
    screen: Res<LoadingScreen>,
    mut layer: Query<(Entity, &mut ShipLayer, &mut Visibility), Without<ShipDot>>,
    mut dots: Query<(&ShipDot, &mut Visibility), Without<ShipLayer>>,
) {
    let Ok((entity, mut layer, mut visibility)) = layer.single_mut() else {
        return;
    };
    let ship = screen
        .ship
        .as_ref()
        .filter(|_| screen.active && !screen.blackout);
    visibility.set_if_neq(match ship {
        Some(_) => Visibility::Inherited,
        None => Visibility::Hidden,
    });
    let shown = ship.map_or(0, |s| s.route.dots_shown(screen.displayed));
    let dot_visibility = |i: usize| {
        if i < shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    };
    if layer.painted == ship.map(|s| s.serial) {
        for (dot, mut visibility) in &mut dots {
            visibility.set_if_neq(dot_visibility(dot.0));
        }
        return;
    }
    layer.painted = ship.map(|s| s.serial);
    commands.entity(entity).despawn_related::<Children>();
    let Some(ship) = ship else {
        return;
    };
    // Paint order is spawn order: the sheet, the dots, then the markers over the route's ends.
    commands.entity(entity).with_children(|layer| {
        let sheet = SHEET.as_vec2();
        for (n, tile) in ship.art.tiles.iter().enumerate() {
            let cell = UVec2::new(n as u32 % TILE_COLS, n as u32 / TILE_COLS) * TILE;
            let size = (SHEET.min(cell + TILE) - cell).as_vec2();
            let crop = size / TILE as f32;
            layer.spawn(quad(
                cell.as_vec2() / sheet,
                size / sheet,
                tile,
                [0.0, crop.x, 0.0, crop.y],
            ));
        }
        for (i, &dot) in ship.route.dots.iter().enumerate() {
            layer.spawn((
                ShipDot(i),
                centred(dot, DOT_HALF, &ship.art.elements, DOT_TC),
                dot_visibility(i),
            ));
        }
        layer.spawn(centred(
            ship.route.depart,
            MARKER_HALF,
            &ship.art.elements,
            DEPART_TC,
        ));
        layer.spawn(centred(
            ship.route.arrive,
            MARKER_HALF,
            &ship.art.elements,
            ARRIVE_TC,
        ));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: Vec2, b: Vec2) -> bool {
        a.distance(b) < 1e-4
    }

    #[test]
    fn a_route_runs_dock_to_dock_with_a_dot_every_fifteen_thousandths() {
        let (depart, arrive) = (Vec2::new(0.2, 0.5), Vec2::new(0.7, 0.5));
        let route = ShipRoute::new(depart, arrive, &[[0.45, 0.5]]);
        // Length 0.5: 33 steps, so 34 dots, the first and last on the docks.
        assert_eq!(route.dots.len(), 34);
        assert!(near(route.dots[0], depart));
        assert!(near(route.dots[33], arrive));
        assert!(
            route.dots.windows(2).all(|w| w[1].x > w[0].x),
            "the dots advance along the route"
        );
        assert!(route.dots.iter().all(|d| (d.y - 0.5).abs() < 1e-5));
    }

    #[test]
    fn the_curve_passes_through_its_bend_points() {
        let bend = Vec2::new(0.5, 0.6);
        let route = ShipRoute::new(Vec2::new(0.2, 0.3), Vec2::new(0.8, 0.3), &[bend.into()]);
        // The bend lies on the dotted line: within a chord's sag of the segment between the two
        // dots around it.
        let off_line = route
            .dots
            .windows(2)
            .map(|w| {
                let along = (w[1] - w[0]).normalize();
                let t = (bend - w[0]).dot(along).clamp(0.0, w[0].distance(w[1]));
                bend.distance(w[0] + along * t)
            })
            .fold(f32::MAX, f32::min);
        assert!(off_line < 1e-3, "the bend is {off_line} off the route");
        // And the dots keep their spacing through it.
        assert!(route
            .dots
            .windows(2)
            .all(|w| (0.003..0.03).contains(&w[0].distance(w[1]))));
    }

    #[test]
    fn a_route_too_short_for_a_step_still_draws_two_dots() {
        let route = ShipRoute::new(Vec2::new(0.5, 0.5), Vec2::new(0.505, 0.5), &[]);
        assert_eq!(route.dots.len(), 2);
    }

    fn route_fixture() -> ShipRoute {
        ShipRoute::new(Vec2::new(0.2, 0.5), Vec2::new(0.7, 0.5), &[[0.45, 0.5]])
    }

    /// A plain raise onto a live route screen leaves the route up (`0x406640` never touches
    /// `[0x8828e4]`); the dismiss frees it (`0x407f1e`).
    #[test]
    fn only_the_dismiss_takes_a_route_off_the_screen() {
        let mut screen = LoadingScreen::default();
        screen.ship = Some(ShipScreen {
            route: route_fixture(),
            art: ShipArt {
                elements: (Handle::default(), Vec2::ONE),
                tiles: Vec::new(),
            },
            serial: 1,
        });
        screen.raise("transfer pending", true, Some(1), 0.0);
        assert!(screen.ship.is_some(), "a raise keeps the route");
        screen.dismiss();
        assert!(screen.ship.is_none(), "the dismiss frees it");
    }

    /// `0x4072c0`'s gates: the route is the ridden transport's leg out of the map the packet
    /// names, and only when the packet names the transport we ride.
    #[test]
    fn a_transfer_draws_the_route_of_the_transport_we_ride() {
        use benilla_protocol::guid::HIGH_MO_TRANSPORT;
        const BOAT: u32 = 176231;
        const PATH: u32 = 292;

        let mut world = World::new();
        let mut templates = GameObjectTemplates::default();
        let mut data = [0i32; 24];
        data[0] = PATH as i32;
        templates.insert(BOAT, 15, "Proudmore's Treasure".into(), &data);
        world.insert_resource(templates);
        let legs = vec![
            Some(RouteLeg {
                from_map: 0,
                route: Some(route_fixture()),
            }),
            Some(RouteLeg {
                from_map: 1,
                route: None,
            }),
        ];
        world.insert_resource(ShipRoutes {
            by_path: HashMap::from([(PATH, legs)]),
        });
        let deck = world
            .spawn(Guid((u64::from(HIGH_MO_TRANSPORT) << 48) | u64::from(BOAT)))
            .id();
        let mut state = bevy::ecs::system::SystemState::<ShipLookup>::new(&mut world);
        let lookup = state.get(&world).expect("the lookup's resources are in");

        assert_eq!(
            lookup.route(Some(deck), Some((BOAT, 0))),
            Some(Some(&route_fixture())),
            "leaving map 0 aboard it: that leg's route"
        );
        assert_eq!(
            lookup.route(Some(deck), Some((BOAT, 1))),
            Some(None),
            "a leg with no spline row: the plain screen"
        );
        assert_eq!(
            lookup.route(Some(deck), Some((BOAT, 530))),
            Some(None),
            "no leg leaves that map"
        );
        assert_eq!(
            lookup.route(Some(deck), Some((20808, 0))),
            None,
            "the server named another transport"
        );
        assert_eq!(lookup.route(None, Some((BOAT, 0))), None, "not aboard");

        // The tick's entry names the leg outright.
        assert_eq!(
            lookup.leg_route(deck, 0),
            Some(Some(&route_fixture())),
            "the leg it left: that leg's route"
        );
        assert_eq!(lookup.leg_route(deck, 1), Some(None), "a leg with no row");
        assert_eq!(lookup.leg_route(deck, 7), Some(None), "no such leg");
        assert_eq!(
            lookup.route(Some(deck), None),
            None,
            "a portal, no transport"
        );
    }

    #[test]
    fn the_dots_fill_with_the_load() {
        let route = ShipRoute {
            depart: Vec2::ZERO,
            arrive: Vec2::ONE,
            dots: vec![Vec2::ZERO; 5],
        };
        assert_eq!(route.dots_shown(0.0), 1, "the first dot is up at once");
        assert_eq!(route.dots_shown(0.49), 2);
        assert_eq!(route.dots_shown(0.5), 3);
        assert_eq!(route.dots_shown(1.0), 5);
    }

    /// The 1.12 tables: each of the six drawn paths has a route out of either map, from the dock
    /// we leave to the dock we reach.
    #[test]
    fn real_routes_join_the_two_docks() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        let nodes = benilla_formats::load_taxi_path_nodes(&mut chain).expect("load TaxiPathNode");
        let routes = ShipRoutes::load(&nodes, &mut chain);
        assert_eq!(routes.by_path.len(), 6);

        // Ratchet (Kalimdor, west of centre) to Booty Bay (Eastern Kingdoms, the south tip).
        let out = routes
            .route(241, 1)
            .expect("the Ratchet boat leaving Kalimdor");
        assert!(near(out.depart, Vec2::new(0.28287, 0.47385)), "{out:?}");
        assert!(near(out.arrive, Vec2::new(0.71845, 0.18675)), "{out:?}");
        assert!(near(out.dots[0], out.depart));
        assert!(near(*out.dots.last().unwrap(), out.arrive));
        let back = routes
            .route(241, 0)
            .expect("and leaving the Eastern Kingdoms");
        assert!(near(back.depart, out.arrive) && near(back.arrive, out.depart));

        assert!(routes.route(241, 530).is_none(), "no leg on that map");

        // Grom'gol to Undercity never leaves map 0, so the tick asks by leg: leg 0 leaves
        // Grom'gol in the south for Undercity, leg 1 is the way back.
        let north = routes.leg_route(301, 0).expect("the Grom'gol zeppelin");
        assert!(north.depart.y < north.arrive.y, "it flies north: {north:?}");
        let south = routes.leg_route(301, 1).expect("the way back");
        assert!(near(south.depart, north.arrive) && near(south.arrive, north.depart));
        assert!(routes.leg_route(301, 2).is_none());
        // By departure map both its legs match map 0, and the lookup takes the first.
        assert_eq!(routes.route(301, 0), Some(north));
        assert!(routes.route(6, 0).is_none(), "a flight path draws no route");
    }
}
