//! The camera-view verbs from the reference's Lua table at `0x84f7a0`: `SetView`, `SaveView`,
//! `ResetView`, `NextView`, `PrevView` and `FlipCameraYaw`. Each acts on engine state the VM does
//! not hold, so each call queues a [`CameraViewRequest`] for the app's camera rig.
//!
//! The indexed verbs share one prologue and return silently, never reaching `luaL_error`
//! (`0x6f4940`), on anything but a number that truncates to 1 to 5; `FlipCameraYaw` keeps its
//! argument's fraction. `SaveView(1)` and `ResetView(1)` act on first person (`0x50fa30` has no
//! view-0 gate; only `Bindings.xml` lacks the bindings). `NextView` and `PrevView` stop at the
//! ends and never wrap (`0x50faa0`, `0x50fac0`).

use mlua::{Lua, Value};

/// The number of camera views, the rows of the reference's default-string table `0x84f488`.
pub const CAMERA_VIEW_COUNT: u8 = 5;

/// A camera-view intent for the app. Indices are internal, `0..CAMERA_VIEW_COUNT`, range-checked
/// here where the reference checks them, in the Lua handler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraViewRequest {
    /// `SetView(n)`: make view `n-1` live (distance, pitch and the saved yaw offset).
    Set(u8),
    /// `SaveView(n)`: store the live pose into view `n-1`.
    Save(u8),
    /// `ResetView(n)`: put view `n-1` back to its shipped default.
    Reset(u8),
    /// `NextView()`: the view above the current one, if any.
    Next,
    /// `PrevView()`: the view below the current one, if any.
    Prev,
    /// `FlipCameraYaw(degrees)`: add degrees to the camera's yaw; `Bindings.xml:795` passes 180.
    FlipYaw(f32),
    /// `CameraZoomIn(n)` / `CameraZoomOut(n)` (a Turtle WoW VM's, [`install_turtle`]): move the
    /// zoom `n` yards toward the head, or away for a negative `n`.
    Zoom(f32),
}

impl super::UiScript {
    /// Drain the camera-view intents queued since the last call.
    pub fn take_camera_view_requests(&mut self) -> Vec<CameraViewRequest> {
        drain_field!(self, camera_view_requests)
    }
}

/// The indexed verbs' shared prologue: the internal index, or `None` for a silent return.
fn view_arg(lua: &Lua, v: Value) -> Option<u8> {
    // Is-number (`0x6f34d0`) and `tonumber` (`0x6f3620`) take a numeric string; the cast
    // truncates toward zero, as `0x40a2b0` does.
    let n = lua.coerce_number(v).ok().flatten()? as i64 as i32;
    (1..=i32::from(CAMERA_VIEW_COUNT))
        .contains(&n)
        .then(|| (n - 1) as u8)
}

/// Register the six camera-view globals.
pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let g = lua.globals();

    // SetView(n), `0x50b5b0`. The reference's `0x512e90` also writes the `cameraView` CVar, so
    // the view survives a restart; the app does both.
    g.set(
        "SetView",
        lua.create_function(|lua, n: Value| {
            if let Some(view) = view_arg(lua, n) {
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model app_data");
                model
                    .camera_view_requests
                    .push(CameraViewRequest::Set(view));
            }
            Ok(())
        })?,
    )?;

    // SaveView(n), `0x50b600` → `0x50fa30`: the camera's target distance, pitch and yaw go into
    // the slot and each to its archived CVar.
    g.set(
        "SaveView",
        lua.create_function(|lua, n: Value| {
            if let Some(view) = view_arg(lua, n) {
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model app_data");
                model
                    .camera_view_requests
                    .push(CameraViewRequest::Save(view));
            }
            Ok(())
        })?,
    )?;

    // ResetView(n), `0x50b640` → `0x50fae0`: re-parses the view's default string (`0x84f488`)
    // into the slot and re-applies the view if it is live.
    g.set(
        "ResetView",
        lua.create_function(|lua, n: Value| {
            if let Some(view) = view_arg(lua, n) {
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model app_data");
                model
                    .camera_view_requests
                    .push(CameraViewRequest::Reset(view));
            }
            Ok(())
        })?,
    )?;

    // NextView() and PrevView(), `0x50b680` and `0x50b690`: neither reads the Lua stack.
    g.set(
        "NextView",
        lua.create_function(|lua, ()| {
            let mut model = lua
                .app_data_mut::<crate::script::ModelCell>()
                .expect("model app_data");
            model.camera_view_requests.push(CameraViewRequest::Next);
            Ok(())
        })?,
    )?;
    g.set(
        "PrevView",
        lua.create_function(|lua, ()| {
            let mut model = lua
                .app_data_mut::<crate::script::ModelCell>()
                .expect("model app_data");
            model.camera_view_requests.push(CameraViewRequest::Prev);
            Ok(())
        })?,
    )?;

    // FlipCameraYaw(degrees), `0x50b6a0`: adds `degrees × π/180` to the camera's yaw, fraction
    // kept; a non-number returns silently.
    g.set(
        "FlipCameraYaw",
        lua.create_function(|lua, degrees: Value| {
            if let Some(d) = lua.coerce_number(degrees).ok().flatten() {
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model app_data");
                model
                    .camera_view_requests
                    .push(CameraViewRequest::FlipYaw(d as f32));
            }
            Ok(())
        })?,
    )?;

    Ok(())
}

/// Register `CameraZoomIn` and `CameraZoomOut` (`0x50b400`, `0x50b450`) in a Turtle WoW VM, whose
/// barbershop calls `CameraZoomOut(2)` as it seats the player (`Turtle_Barbershop.lua`
/// `Barbershop:SetupCamera`). A 1.12.1 VM keeps the reference's pair, binding inputs
/// (`input_verbs`); this replaces it in a Turtle VM. A number argument, a numeric string included (`0x6f34d0`), is the
/// amount, anything else 1.0 (`0x50b42d`); the camera then zooms for `amount /
/// cameraDistanceMoveSpeed` seconds at that speed (`0x50fc60`, `0x50fcd0`), `amount` yards.
pub(super) fn install_turtle(lua: &Lua) -> mlua::Result<()> {
    let g = lua.globals();
    for (name, sign) in [("CameraZoomIn", 1.0f32), ("CameraZoomOut", -1.0f32)] {
        g.set(
            name,
            lua.create_function(move |lua, amount: Value| {
                let yards = lua.coerce_number(amount).ok().flatten().unwrap_or(1.0) as f32;
                let mut model = lua
                    .app_data_mut::<crate::script::ModelCell>()
                    .expect("model app_data");
                model
                    .camera_view_requests
                    .push(CameraViewRequest::Zoom(sign * yards));
                Ok(())
            })?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CameraViewRequest;
    use crate::script::{UiFlavor, UiScript};

    /// A Turtle VM's zoom pair queues its amount, 1.0 for anything that is not a number; a 1.12.1
    /// VM's pair is the reference's binding input (`input_verbs`) and queues no view request.
    #[test]
    fn a_turtle_vm_zooms_by_its_argument_and_a_vanilla_one_keeps_the_binding_verbs() {
        let mut vanilla = UiScript::new().unwrap();
        vanilla.run("CameraZoomOut(2) CameraZoomIn()").unwrap();
        assert!(vanilla.take_camera_view_requests().is_empty());

        let mut s = UiScript::new().unwrap();
        s.set_flavor(UiFlavor::Turtle);
        s.run(r#"CameraZoomOut(2) CameraZoomIn("0.5") CameraZoomIn() CameraZoomOut({})"#)
            .unwrap();
        assert_eq!(
            s.take_camera_view_requests(),
            vec![
                CameraViewRequest::Zoom(-2.0),
                CameraViewRequest::Zoom(0.5),
                CameraViewRequest::Zoom(1.0),
                CameraViewRequest::Zoom(-1.0),
            ]
        );
    }
}
