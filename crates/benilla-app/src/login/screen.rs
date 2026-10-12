//! The login screen's layout: the reference `AccountLogin.xml` arrangement in Bevy UI, scaled by
//! `height / 768` (the glue virtual screen), over the `UI_MainMenu` scene. Each control carries
//! its `AccountLogin.xml` size and anchor at its spawn site.
//!
//! Deviation: the Credits, Cinematics, TOS, Community Site and Manage Account buttons are absent,
//! because the screen keeps only what logging in needs. Deviation (this fork): so is the Quit
//! button (`AccountLoginExitButton`), because in a browser tab closing the client is the
//! browser's gesture.
//!
//! Under Turtle WoW 1.18.1 the screen takes that client's `AccountLogin.xml` seats: its own logo
//! at 236×118, the Blizzard logo at BOTTOM (0,20), a 173-wide Login button at TOP (11,−519) and
//! the checkbox under that button. Deviation: Turtle's six link buttons are absent, as the
//! Community Site button is.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::char_select::wow_font;
use crate::glue::art::{GlueArt, BACKDROP, GOLD};
use crate::glue::widgets::{
    abs, glue_button, glue_edit_box, outlined_text, outlined_text_centered, overlay,
    paint_glue_field, ArtSwap, GlueBtnKind, GlueFieldPart, GlueText, Hilight,
};
use crate::glue_strings::GlueStrings;
use crate::portrait::{PortraitImages, PortraitSource, GLUE_SLOT};
use benilla_assets::WorldAssets;

use super::{ClientState, Field, LoginForm};

const SCREEN_Z: i32 = 1100;
/// `GlueFontDisableSmall`'s color (GlueFonts.xml), the `AccountLoginRealmName` readout's grey.
const DISABLED_GREY: Color = Color::srgb(0.5, 0.5, 0.5);
/// `DEFAULT_TOOLTIP_COLOR` (AccountLogin.lua): the edit boxes' backdrop tint (border rgb, bg rgb).
const BOX_BORDER: Color = Color::srgb(0.8, 0.8, 0.8);
const BOX_FILL: Color = Color::srgb(0.09, 0.09, 0.09);

/// One clickable control on the screen.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoginAction {
    FocusAccount,
    FocusPassword,
    Login,
    ToggleSave,
}

/// Root of the login screen; `with_art`, `s` (the glue scale) and `turtle` (the client whose
/// seats it took) record what the tree was built with, so late art, a window resize or a change
/// of flavor rebuilds it.
#[derive(Component)]
pub(super) struct LoginUi {
    with_art: bool,
    s: f32,
    turtle: bool,
}
/// The account box's row items, painted by [`refresh_boxes`].
#[derive(Component, Clone)]
pub(super) struct AccountText;
/// The password box's row items, displayed as the `*` mask.
#[derive(Component, Clone)]
pub(super) struct PasswordText;
/// The checkbox's checked overlay, shown while the form's save flag is set.
#[derive(Component)]
pub(super) struct CheckMark;
/// The checkbox's hover highlight, driven by [`refresh_checkbox`].
#[derive(Component)]
pub(super) struct CheckHilight;
/// The realmlist readout — the ref's own `AccountLoginRealmName` slot, rewritten in place by
/// [`refresh_realmlist`] when the address changes.
#[derive(Component)]
pub(super) struct RealmlistReadout;

/// Spawn the screen tree once its prerequisites exist (the initial state's `OnEnter` fires before
/// the MPQ chain and booth slots do), and rebuild an artless early spawn when the art lands.
pub(super) fn materialize_screen(
    mut commands: Commands,
    existing: Query<(Entity, &LoginUi)>,
    assets: Res<AssetServer>,
    portraits: Res<PortraitImages>,
    mut art: ResMut<GlueArt>,
    world_assets: Option<ResMut<WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
    mut add_mats: ResMut<Assets<crate::glue::add_material::AddUiMaterial>>,
    strings: Option<Res<GlueStrings>>,
    form: Res<LoginForm>,
    realmlist: Res<crate::realmlist::Realmlist>,
    window: Query<&Window, With<PrimaryWindow>>,
    time: Res<Time>,
) {
    if let Some(mut wa) = world_assets {
        art.ensure_loaded(&mut wa, &mut images, &mut add_mats);
    }
    let with_art = art.button_up.is_some();
    let s = crate::glue::screen_scale(window.single().ok());
    let turtle = crate::flavor::turtle();
    match existing.single() {
        Ok((root, ui)) => {
            if (!ui.with_art && with_art) || ui.s != s || ui.turtle != turtle {
                commands.entity(root).despawn();
                spawn_screen(
                    &mut commands,
                    &assets,
                    &portraits,
                    &art,
                    strings.as_deref(),
                    &form,
                    &realmlist,
                    &window,
                );
            }
        }
        Err(_) => {
            if with_art || time.elapsed_secs() > 1.0 {
                spawn_screen(
                    &mut commands,
                    &assets,
                    &portraits,
                    &art,
                    strings.as_deref(),
                    &form,
                    &realmlist,
                    &window,
                );
                // The page's boot overlay drops on this signal — the first spawn (artless or
                // not) is the first frame a person can actually see and use.
                crate::webprogress::signal_ready_once();
            }
        }
    }
}

fn spawn_screen(
    commands: &mut Commands,
    assets: &AssetServer,
    portraits: &PortraitImages,
    art: &GlueArt,
    strings: Option<&GlueStrings>,
    form: &LoginForm,
    realmlist: &crate::realmlist::Realmlist,
    window: &Query<&Window, With<PrimaryWindow>>,
) {
    let font = wow_font(assets);
    // The edit boxes type in `GlueEditBoxFont`, ARIALN (GlueFonts.xml).
    let edit_font: Handle<Font> = assets.load("mpq://Fonts/ARIALN.ttf");
    let scene_image = match portraits.0.get(GLUE_SLOT) {
        Some(PortraitSource::Live(h)) => Some(h.clone()),
        _ => None,
    };
    let s = crate::glue::screen_scale(window.single().ok());
    let px = |v: f32| Val::Px(v * s);
    // Whose `AccountLogin.xml` seats the screen takes (the module header).
    let turtle = crate::flavor::turtle();
    let empty = GlueStrings::default();
    let strings = strings.unwrap_or(&empty);

    let root = commands
        .spawn((
            crate::glue::NativeGlue,
            LoginUi {
                with_art: art.button_up.is_some(),
                s,
                turtle,
            },
            GlobalZIndex(SCREEN_Z),
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            BackgroundColor(BACKDROP),
        ))
        .with_children(|ui| {
            // The fullscreen `UI_MainMenu` ModelFFX scene, window-sized: the pillarbox bars are the
            // booth camera's own clear inside the target, so the pane must cover the window.
            if let Some(image) = scene_image {
                ui.spawn((ImageNode::new(image), overlay()));
            }
        })
        .id();

    // The chrome hangs off the canvas, the boxed scene's rect, so it stays out of the bars.
    let mut canvas = commands.spawn((crate::glue::glue_canvas(), ChildOf(root)));
    canvas.with_children(|ui| {
        // The WoW logo (`AccountLoginLogo`, 256×128 at TOPLEFT (3,−7), OVERLAY). Turtle's is its
        // own file at 236×118 in the same seat; it is no [`GlueLogo`](crate::glue::art::GlueLogo),
        // which a data change repoints at the shared logo.
        let (logo, logo_w, logo_h) = if turtle {
            (art.login_logo.as_ref().or(art.logo.as_ref()), 236.0, 118.0)
        } else {
            (art.logo.as_ref(), 256.0, 128.0)
        };
        if let Some(logo) = logo {
            let mut node = ui.spawn((
                ImageNode::new(logo.clone()),
                abs(s, 3.0, 7.0, logo_w, logo_h),
            ));
            if !turtle {
                node.insert(crate::glue::art::GlueLogo);
            }
        }

        // The Blizzard logo (100×100 at BOTTOM (0,8), ARTWORK; Turtle's at (0,20)) under the
        // `BLIZZ_DISCLAIMER` line at BOTTOM (0,10), an authored overlap.
        if let Some(blizz) = &art.blizzard_logo {
            ui.spawn((Node {
                position_type: PositionType::Absolute,
                bottom: px(if turtle { 20.0 } else { 8.0 }),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                ..default()
            },))
                .with_children(|c| {
                    c.spawn((
                        ImageNode::new(blizz.clone()),
                        Node {
                            width: px(100.0),
                            height: px(100.0),
                            ..default()
                        },
                    ));
                });
        }
        // No `justifyH`, so CENTER: each line of a two-line disclaimer (Turtle WoW's) centres.
        outlined_text_centered(
            ui,
            Node {
                position_type: PositionType::Absolute,
                bottom: px(10.0),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            (),
            (),
            GlueText {
                text: strings.text(
                    "BLIZZ_DISCLAIMER",
                    "Copyright 2004-2006  Blizzard Entertainment. All Rights Reserved.",
                ),
                size: 12.0, // GlueFontNormalSmall
                color: GOLD,
                wrap: false,
            },
            &font,
            s,
        );

        // `AccountLoginVersion` (GlueFontNormalSmall at BOTTOMLEFT (0,10)): `VERSION_TEMPLATE`
        // filled with versionType, version, internalVersion, buildType and date.
        let version = {
            let template = strings.text("VERSION_TEMPLATE", "%s %s (%s) (%s)\n%s");
            let build = benilla_protocol::CLIENT_BUILD.to_string();
            let mut out = template.to_string();
            for piece in [
                "Version",
                "1.12.1",
                build.as_str(),
                "Release",
                "Sep 19 2006",
            ] {
                out = out.replacen("%s", piece, 1);
            }
            out
        };
        outlined_text(
            ui,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                bottom: px(10.0),
                ..default()
            },
            (),
            (),
            GlueText {
                text: &version,
                size: 12.0,
                color: GOLD,
                wrap: false,
            },
            &font,
            s,
        );

        // The account box (160×37 at BOTTOM (8,345)) and password box (160×37 at BOTTOM
        // (8,270)): a 16·s left margin in a centred row makes the +8 x offset.
        for (bottom, label, action, marker_account) in [
            (
                345.0,
                strings.text("ACCOUNT_NAME", "Account Name"),
                LoginAction::FocusAccount,
                true,
            ),
            (
                270.0,
                strings.text("PASSWORD", "Account Password"),
                LoginAction::FocusPassword,
                false,
            ),
        ] {
            // The label is a 256×64 centred rect anchored BOTTOM to the box's TOP at (0,−23), so
            // it centres on the box at +8, the same left margin as the box row.
            outlined_text(
                ui,
                Node {
                    position_type: PositionType::Absolute,
                    bottom: px(bottom + 37.0 - 23.0),
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    height: px(64.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    margin: UiRect::left(px(16.0)),
                    ..default()
                },
                (),
                (),
                GlueText {
                    text: label,
                    size: 15.0, // GlueFontNormal
                    color: GOLD,
                    wrap: false,
                },
                &font,
                s,
            );
            ui.spawn((Node {
                position_type: PositionType::Absolute,
                bottom: px(bottom),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                ..default()
            },))
                .with_children(|row| {
                    row.spawn(Node {
                        margin: UiRect::left(px(16.0)),
                        ..default()
                    })
                    .with_children(|slot| {
                        if marker_account {
                            glue_edit_box(
                                slot,
                                art,
                                &edit_font,
                                (action, Button),
                                AccountText,
                                (160.0, 37.0),
                                (BOX_BORDER, BOX_FILL),
                                (15.0, 0.0, 0.0, 5.0), // AccountLogin.xml TextInsets
                                s,
                            );
                        } else {
                            glue_edit_box(
                                slot,
                                art,
                                &edit_font,
                                (action, Button),
                                PasswordText,
                                (160.0, 37.0),
                                (BOX_BORDER, BOX_FILL),
                                (15.0, 0.0, 0.0, 5.0), // AccountLogin.xml TextInsets
                                s,
                            );
                        }
                    });
                });
        }

        // Login (`GlueButtonTemplate` 170×45 at TOP (8,−519); Turtle's 173×45 at TOP (11,−519)):
        // the left margin in a centred row is twice the x offset.
        let (login_w, login_dx) = if turtle { (173.0, 11.0) } else { (170.0, 8.0) };
        ui.spawn((Node {
            position_type: PositionType::Absolute,
            top: px(519.0),
            width: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            ..default()
        },))
            .with_children(|row| {
                row.spawn(Node {
                    margin: UiRect::left(px(2.0 * login_dx)),
                    ..default()
                })
                .with_children(|slot| {
                    glue_button(
                        slot,
                        art,
                        &font,
                        LoginAction::Login,
                        strings.text("LOGIN", "Login"),
                        login_w,
                        45.0,
                        GlueBtnKind::Normal,
                        s,
                    );
                });
            });

        // The realmlist readout, the address this client will dial, in the reference's
        // `AccountLoginRealmName` seat. It is the TOS button's own string: 256 wide,
        // right-justified, TOPRIGHT at that button's BOTTOMRIGHT (−8,−10), and the button's
        // BOTTOM sits at `AccountLoginExitButton`'s TOP + 80, the Quit button 150×38 at
        // BOTTOMRIGHT (−5,29). Resolved: top 631, right 13; it keeps that seat with both buttons
        // gone. Deviation (this fork): a readout only, with no editor behind it, because the browser
        // client's address comes from `/config.json` alone and a saved copy would outrank the
        // hosted one. Native keeps the `realmList` CVar.
        outlined_text(
            ui,
            Node {
                position_type: PositionType::Absolute,
                right: px(13.0),
                top: px(631.0),
                ..default()
            },
            (),
            RealmlistReadout,
            GlueText {
                text: realmlist.address(),
                size: 12.0, // GlueFontDisableSmall
                color: DISABLED_GREY,
                wrap: false,
            },
            &font,
            s,
        );

        // The Remember Account Name checkbox (20×20 at (17, top 653), resolved from its anchor
        // under the absent Community button) and its label at LEFT+24. Turtle anchors its
        // BOTTOMLEFT to the Login button's at (−15,−15): 90.5 left of the centre, top 559.
        let (check_left, check_margin, check_top) = if turtle {
            (Val::Percent(50.0), px(-90.5), 559.0)
        } else {
            (px(17.0), Val::Px(0.0), 653.0)
        };
        ui.spawn((Node {
            position_type: PositionType::Absolute,
            left: check_left,
            top: px(check_top),
            margin: UiRect::left(check_margin),
            height: px(20.0),
            align_items: AlignItems::Center,
            flex_direction: FlexDirection::Row,
            ..default()
        },))
            .with_children(|row| {
                let mut b = row.spawn((
                    LoginAction::ToggleSave,
                    Button,
                    Node {
                        width: px(20.0),
                        height: px(20.0),
                        ..default()
                    },
                ));
                match &art.checkbox {
                    Some(check) => {
                        b.insert((
                            ImageNode::new(check.up.clone()),
                            ArtSwap {
                                up: check.up.clone(),
                                down: check.down.clone(),
                            },
                        ));
                        b.with_children(|inner| {
                            inner.spawn((
                                CheckMark,
                                if form.save {
                                    Visibility::Inherited
                                } else {
                                    Visibility::Hidden
                                },
                                ImageNode::new(check.checked.clone()),
                                overlay(),
                            ));
                            if let Some(hi) = &check.hi {
                                inner.spawn((
                                    CheckHilight,
                                    Hilight,
                                    Visibility::Hidden,
                                    bevy::ui_render::ui_material::MaterialNode(hi.clone()),
                                    overlay(),
                                ));
                            }
                        });
                    }
                    None => {
                        b.insert(BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.15)));
                        b.with_children(|inner| {
                            inner.spawn((
                                CheckMark,
                                if form.save {
                                    Visibility::Inherited
                                } else {
                                    Visibility::Hidden
                                },
                                Text::new("x"),
                                TextFont {
                                    font: font.clone().into(),
                                    font_size: FontSize::Px(14.0 * s),
                                    ..default()
                                },
                                TextColor(GOLD),
                            ));
                        });
                    }
                }
                row.spawn((
                    Text::new(strings.text("SAVE_ACCOUNT_NAME", "Remember Account Name")),
                    TextFont {
                        font: font.clone().into(),
                        font_size: FontSize::Px(10.0 * s), // the authored FontHeight 10
                        ..default()
                    },
                    TextColor(GOLD),
                    TextShadow {
                        offset: Vec2::new(s, s),
                        color: Color::BLACK,
                    },
                    Node {
                        margin: UiRect::left(px(4.0)), // LEFT+24 from the checkbox's left edge
                        ..default()
                    },
                ));
            });
    });
}

/// Paint both boxes from their [`EditBoxState`]s through the shared [`paint_glue_field`].
#[allow(clippy::type_complexity)]
pub(super) fn refresh_boxes(
    form: Res<LoginForm>,
    mut account: Query<
        (&GlueFieldPart, Option<&mut Text>, &mut Visibility),
        (With<AccountText>, Without<PasswordText>),
    >,
    mut password: Query<(&GlueFieldPart, Option<&mut Text>, &mut Visibility), With<PasswordText>>,
) {
    paint_glue_field(
        &form.account,
        form.focus == Field::Account,
        account.iter_mut(),
    );
    paint_glue_field(
        &form.password,
        form.focus == Field::Password,
        password.iter_mut(),
    );
}

/// The realmlist readout, rewritten in place when the address changes
/// (`AccountLoginRealmName:SetText`); `sync_outlines` carries it to the outline copies. It
/// compares every frame, not on `Res::is_changed`, because a rebuilt tree can postdate the change.
pub(super) fn refresh_realmlist(
    realmlist: Res<crate::realmlist::Realmlist>,
    mut texts: Query<&mut Text, With<RealmlistReadout>>,
) {
    for mut t in &mut texts {
        if t.0 != realmlist.address() {
            t.0 = realmlist.address().to_string();
        }
    }
}

/// The checkbox's checked overlay and hover ring; it is not a `GlueBtn`, so the shared button
/// pass skips it.
#[allow(clippy::type_complexity)]
pub(super) fn refresh_checkbox(
    form: Res<LoginForm>,
    boxes: Query<(&Interaction, &Children), With<ArtSwap>>,
    mut marks: Query<&mut Visibility, (With<CheckMark>, Without<CheckHilight>)>,
    mut hilights: Query<&mut Visibility, With<CheckHilight>>,
) {
    for mut vis in &mut marks {
        let want = if form.save {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
    for (interaction, children) in &boxes {
        for child in children {
            if let Ok(mut vis) = hilights.get_mut(*child) {
                let want = if *interaction != Interaction::None {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                if *vis != want {
                    *vis = want;
                }
            }
        }
    }
}

/// Leaving the login screen: drop the tree and any open dialog.
pub(super) fn exit_login(
    mut commands: Commands,
    roots: Query<Entity, With<LoginUi>>,
    mut dialog: ResMut<crate::glue::dialog::GlueDialog>,
) {
    for e in &roots {
        commands.entity(e).despawn();
    }
    if let Some(root) = dialog.root.take() {
        commands.entity(root).despawn();
    }
    dialog.close();
}

/// Leaving the native login screen drops its scene; the next screen sets its own the same frame.
pub(super) fn clear_login_scene(mut preview: ResMut<crate::portrait::GluePreview>) {
    preview.scene = None;
    preview.look = None;
}

/// `WOW_LOGIN_SHOT_OUT=<path>`: one PNG of the login screen once it has settled.
pub(super) fn debug_login_shot(
    mut commands: Commands,
    state: Res<State<ClientState>>,
    time: Res<Time>,
    mut entered_at: Local<Option<f32>>,
    mut done: Local<bool>,
) {
    if *done || *state.get() != ClientState::Login {
        return;
    }
    let Ok(out) = std::env::var("WOW_LOGIN_SHOT_OUT") else {
        *done = true;
        return;
    };
    let start = *entered_at.get_or_insert(time.elapsed_secs());
    if time.elapsed_secs() - start < 8.0 {
        return;
    }
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(out.clone()));
    info!("login: shot instrument writing {out}");
    *done = true;
}
