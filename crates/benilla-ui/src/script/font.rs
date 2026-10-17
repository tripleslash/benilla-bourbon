//! Font objects: a `<Font name=…>` or a `CreateFont` font is a Lua object, published as the global
//! of its name with its own method table. Stock FrameXML never calls this API; addons do.
//!
//! The reference's class is `CSimpleFont` (`0x87a454`). Its method table (`0x87c7c8`) has 22
//! entries (`mov edx,0x16` at `0x7a10d5`), and its lookup `0x7a1100` has no base-class fallback.
//! All are here; `CopyFontObject` is the one a FontString lacks. A named font is published by `0x783870` through `SetName 0x784150` and `CreateLuaHandle
//! 0x701bd0`, which leaves a non-nil `_G[name]` alone (`0x701cb8`).
//!
//! A setter repaints every FontString that inherits the font. The reference links dependents into
//! the font's list (`+0x74`/`+0x78`) and each setter ends in `NotifyDependents 0x784180`, calling
//! `OnFontChanged` (`0x77e4b0` on a font, `0x773530` then `0x770800` on a FontString); here each
//! setter writes the [`FontObject`] and calls [`propagate`]. A local setter clears its property's
//! bit in the dependent's inherit mask (`+0x2c`, FontString `+0xd4`) and nothing restores it, so a
//! colour set on a FontString survives a later `SetFontObject`: [`RegionData::font_explicit`] is
//! that mask. `+0x38` is not it; its high bits mark a held value, which the inheriting merge also
//! sets (`0x7709d3`).
//!
//! A `<Font inherits=…>` chain is flattened at load (`Loader::do_font`) and kept linked
//! ([`FontLink`]): a setter on a parent re-derives each font that inherits it ([`cascade`]), as the
//! reference links them live (`LoadXML 0x783c30` calls the `SetFontObject` link `0x770c60` for
//! `inherits=`, `0x783ce6`). pfUI restyles the client by `GameFontNormal:SetFont`, which has to
//! reach `GameFontHighlight` and every other font declared from it.

use mlua::{Lua, Table, Value};

use super::binding_abi;
use super::object::publish_global;
use crate::justify;

use super::{FontObject, FontShadow, Model, RegionData};

/// The shared metatable of every font-object handle.
const REG_FONT_META: &str = "__benilla_font_meta";
/// The font-object method table, which is the metatable's `__index`.
const REG_FONT_METHODS: &str = "__benilla_font_methods";
/// name → handle, so `GameFontNormal == GameFontNormal` and a re-declared `<Font>` keeps identity.
const REG_FONT_WRAPPERS: &str = "__benilla_font_wrappers";

// ── The handle ───────────────────────────────────────────────────────────────────────────────

/// Get or create the Lua handle for a named font object. Keyed by name, which every font has (an
/// unnamed `<Font>` is dropped at parse), so a re-declared `<Font>` updates its record in place.
///
/// `T[0]` holds the name, not a frame id, so a font handle passed where a frame is wanted is
/// rejected rather than decoded.
pub(crate) fn wrapper(lua: &Lua, name: &str) -> mlua::Result<Table> {
    let wrappers: Table = lua.named_registry_value(REG_FONT_WRAPPERS)?;
    if let Value::Table(t) = wrappers.get::<Value>(name)? {
        return Ok(t);
    }
    let t = lua.create_table()?;
    t.raw_set(0, name)?;
    let meta: Table = lua.named_registry_value(REG_FONT_META)?;
    t.set_metatable(Some(meta))?;
    wrappers.set(name, t.clone())?;
    Ok(t)
}

/// Publish the handle as `_G[name]`, leaving a non-nil global alone as [`publish_global`] does.
pub(crate) fn publish(lua: &Lua, name: &str) -> mlua::Result<()> {
    let t = wrapper(lua, name)?;
    publish_global(lua, name, &t)
}

/// The name a font handle holds in `T[0]`.
fn name_of(this: &Table) -> mlua::Result<String> {
    match this.raw_get::<Value>(0)? {
        Value::String(s) => Ok(s.to_str()?.to_string()),
        _ => Err(mlua::Error::runtime(
            "not a font object (missing T[0] name identity)",
        )),
    }
}

/// A font argument as a registry name, `None` for nil: an object, a name or nil, the three forms of
/// the reference's usage `SetFontObject(font or "font" or nil)` (`0x87c5cc`). Anything else
/// raises, as the reference's `luaL_error 0x6f4940` does.
pub(super) fn resolve(verb: &str, v: &Value) -> mlua::Result<Option<String>> {
    match v {
        Value::Nil => Ok(None),
        Value::Table(t) => name_of(t).map(Some).map_err(|_| {
            mlua::Error::runtime(format!(
                "{verb}: the argument is a table but not a font object \
                 (a frame or region cannot be a font)"
            ))
        }),
        Value::String(s) => Ok(Some(s.to_str()?.to_string())),
        other => Err(mlua::Error::runtime(format!(
            "{verb}: expected a font object, a font name, or nil, got {}",
            other.type_name()
        ))),
    }
}

/// [`resolve`] refusing nil, for `CopyFontObject`, which has nothing to copy from (`0x7a01b0`).
fn resolve_required(verb: &str, v: &Value) -> mlua::Result<String> {
    resolve(verb, v)?.ok_or_else(|| {
        mlua::Error::runtime(format!(
            "{verb}: expected a font object or a font name, got nil"
        ))
    })
}

// ── The live link: a font object to the regions that inherit it ──────────────────────────────

/// Copy a font object's paint onto one region, skipping each property the region set itself
/// ([`RegionData::font_explicit`]) and each the font does not hold: the reference gates every merge
/// on the source's held-value bits (`+0x38`, the merge `0x770910`), so an empty `CreateFont` font
/// changes nothing.
///
/// The outline is always copied, as [`Outline`](super::Outline) has no unset state; only an empty
/// `CreateFont` font on an outlined FontString differs from the reference, clearing the outline.
pub(crate) fn repaint(d: &mut RegionData, fo: &FontObject) {
    let ex = d.font_explicit;
    if !ex.face {
        if let Some(f) = &fo.font {
            d.font_path = Some(f.clone());
        }
    }
    if !ex.height {
        if let Some(h) = fo.height {
            d.font_height = Some(h);
        }
    }
    if !ex.outline {
        d.outline = fo.outline;
    }
    if !ex.shadow {
        if let Some(s) = fo.shadow {
            d.font_shadow = Some(s);
        }
    }
    if !ex.color {
        if let Some(c) = fo.color {
            d.set_vertex_color(Some(c));
        }
    }
    if !ex.justify_h {
        if let Some(j) = fo.justify_h {
            d.justify.set_h(j);
        }
    }
    if !ex.justify_v {
        if let Some(j) = fo.justify_v {
            d.justify.set_v(j);
        }
    }
    // Copied as given: a FontString's reader clamps it, as its setter `0x772240` does at the merge
    // (`0x7708fd`).
    if !ex.spacing {
        if let Some(sp) = fo.spacing {
            d.spacing = sp;
        }
    }
}

/// Repaint every region that inherits `name`, as the reference's live parent link does. Linear in
/// regions: font-object setters are addon configuration, never per frame. Button state fonts are
/// stored by name and re-resolved at every `extract`, so they need no push.
pub(crate) fn propagate(model: &mut Model, name: &str) {
    let Some(fo) = model.font_object(name).cloned() else {
        return;
    };
    for d in model.region_data.values_mut() {
        if d.font_object.as_deref() == Some(name) {
            repaint(d, &fo);
        }
    }
    // A fan-out cannot name its regions without a second pass, so every region is re-measured.
    model.touch_measure_all();
}

/// A declared font's live link to the font it inherits.
#[derive(Clone, Debug, Default)]
pub(crate) struct FontLink {
    /// The parent's lowercased name.
    pub(crate) parent: String,
    /// What this font holds itself, declared or set since; every other property is the parent's.
    pub(crate) own: FontObject,
}

/// `own` over `parent`: a property the font holds wins, the rest is inherited. An outline of
/// none is the unset state here, as [`Outline`](super::Outline) has no other.
fn derive(parent: &FontObject, own: &FontObject) -> FontObject {
    FontObject {
        font: own.font.clone().or_else(|| parent.font.clone()),
        height: own.height.or(parent.height),
        shadow: own.shadow.or(parent.shadow),
        color: own.color.or(parent.color),
        outline: if own.outline == super::Outline::default() {
            parent.outline
        } else {
            own.outline
        },
        justify_h: own.justify_h.or(parent.justify_h),
        justify_v: own.justify_v.or(parent.justify_v),
        spacing: own.spacing.or(parent.spacing),
    }
}

/// Re-derive every font that inherits `parent` (a lowercased name), repaint its regions, and go
/// on down the chain. A chain deeper than any real one stops, so a declared cycle cannot spin.
pub(crate) fn cascade(model: &mut Model, parent: &str, depth: u8) {
    if depth > 16 {
        return;
    }
    let Some(parent_record) = model.font_objects_by_lower.get(parent).cloned() else {
        return;
    };
    let children: Vec<(String, FontObject)> = model
        .font_links
        .iter()
        .filter(|(_, link)| link.parent == parent)
        .map(|(child, link)| (child.clone(), derive(&parent_record, &link.own)))
        .collect();
    for (child, record) in children {
        if model.font_objects_by_lower.get(&child) == Some(&record) {
            continue;
        }
        model
            .font_objects_by_lower
            .insert(child.clone(), record.clone());
        for d in model.region_data.values_mut() {
            if d.font_object
                .as_deref()
                .is_some_and(|name| name.eq_ignore_ascii_case(&child))
            {
                repaint(d, &record);
            }
        }
        cascade(model, &child, depth + 1);
    }
}

/// Run `f` on the named font object's record, then [`propagate`] and [`cascade`]: every setter's
/// one write path. What the setter changed becomes the font's own, so its parent no longer
/// supplies it.
fn edit<R>(lua: &Lua, this: &Table, f: impl FnOnce(&mut FontObject) -> R) -> mlua::Result<R> {
    let name = name_of(this)?;
    let key = name.to_ascii_lowercase();
    let mut model = lua.app_data_mut::<Model>().expect("model");
    let record = model.font_objects_by_lower.entry(key.clone()).or_default();
    let before = record.clone();
    let out = f(record);
    let after = record.clone();
    if let Some(link) = model.font_links.get_mut(&key) {
        let own = &mut link.own;
        if before.font != after.font {
            own.font = after.font.clone();
        }
        if before.height != after.height {
            own.height = after.height;
        }
        if before.shadow != after.shadow {
            own.shadow = after.shadow;
        }
        if before.color != after.color {
            own.color = after.color;
        }
        if before.outline != after.outline {
            own.outline = after.outline;
        }
        if before.justify_h != after.justify_h {
            own.justify_h = after.justify_h;
        }
        if before.justify_v != after.justify_v {
            own.justify_v = after.justify_v;
        }
        if before.spacing != after.spacing {
            own.spacing = after.spacing;
        }
    }
    propagate(&mut model, &name);
    cascade(&mut model, &key, 0);
    Ok(out)
}

/// A copy of the named font object's record.
fn read(lua: &Lua, this: &Table) -> mlua::Result<FontObject> {
    let name = name_of(this)?;
    let model = lua.app_data_ref::<Model>().expect("model");
    Ok(model.font_object(&name).cloned().unwrap_or_default())
}

// ── install: the method table, the metatable and CreateFont ──────────────────────────────────

/// Build the font-object method table and metatable, and register `CreateFont`.
pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    lua.set_named_registry_value(REG_FONT_WRAPPERS, lua.create_table()?)?;

    let m = lua.create_table()?;

    // ── identity ────────────────────────────────────────────────────────────────────────────
    m.set(
        "GetObjectType",
        lua.create_function(|_, _this: Table| Ok("Font"))?,
    )?;
    // `1` or nil, never a boolean, like the Region's (`0x7a1290`): the Font's own `IsObjectType`
    // (`0x79fe60`) answers a number. Its argument handling is untraced; mlua's coercion applies.
    m.set(
        "IsObjectType",
        lua.create_function(|_, (_this, ty): (Table, String)| {
            Ok(binding_abi::flag(ty.eq_ignore_ascii_case("font")))
        })?,
    )?;
    m.set(
        "GetName",
        lua.create_function(|_, this: Table| name_of(&this))?,
    )?;

    // ── the font-object-on-font-object pair ─────────────────────────────────────────────────
    // SetFontObject(other): on a Font, the same copy as CopyFontObject, since Font chains are
    // flattened (module doc); nil is a no-op, with no link to sever.
    m.set(
        "SetFontObject",
        lua.create_function(|lua, (this, other): (Table, Value)| {
            match resolve("SetFontObject", &other)? {
                Some(_) => copy_from(lua, &this, &other, "SetFontObject"),
                None => Ok(()),
            }
        })?,
    )?;
    // CopyFontObject(other): refuses nil (`0x7a01b0` value-copies and re-parents).
    m.set(
        "CopyFontObject",
        lua.create_function(|lua, (this, other): (Table, Value)| {
            resolve_required("CopyFontObject", &other)?;
            copy_from(lua, &this, &other, "CopyFontObject")
        })?,
    )?;
    // GetFontObject(): always nil; with Font chains flattened, a Font has no parent link.
    m.set(
        "GetFontObject",
        lua.create_function(|_, _this: Table| Ok(Value::Nil))?,
    )?;

    // ── face ────────────────────────────────────────────────────────────────────────────────
    // SetFont(path, height [, flags]) → 1, or nil when the font fails to load, which does not raise
    // and writes nothing: an empty path, one the host's probe refuses, or a height of 0 or less.
    m.set(
        "SetFont",
        lua.create_function(
            |lua, (this, file, height, flags): (Table, Value, Value, Value)| {
                // The argument gate every kind shares (`0x79f210`); a bad path or height raises
                // (`0x87c69c`), and anything but a string or a number is no flags.
                let super::font_block::FontArgs {
                    path,
                    height,
                    flags,
                } = super::font_block::set_font_args(lua, &this, &file, &height, &flags)?;
                if !super::font_block::instance_loads(lua, &path, height) {
                    return Ok(Value::Nil);
                }
                edit(lua, &this, |fo| {
                    fo.font = Some(path);
                    fo.outline = flags;
                    fo.height = Some(height);
                })?;
                Ok(Value::Number(1.0))
            },
        )?,
    )?;
    // GetFont() → path, height, flags. An unset Font answers nil, 0, "": the constructor
    // `0x783a40` zeroes the height and stores a NULL path (`0x41e3a0`), which pushes as nil.
    m.set(
        "GetFont",
        lua.create_function(|lua, this: Table| {
            let fo = read(lua, &this)?;
            let path = match fo.font {
                Some(p) => Value::String(lua.create_string(&p)?),
                None => Value::Nil,
            };
            Ok((path, fo.height.unwrap_or(0.0), fo.outline.as_str()))
        })?,
    )?;

    // ── colour, and the alpha that is its fourth channel ────────────────────────────────────
    // Shape C on r, g, b (`SetTextColor 0x79f4d0`): nil or a non-number is 0, never a raise.
    m.set(
        "SetTextColor",
        lua.create_function(
            |lua, (this, r, g, b, a): (Table, Value, Value, Value, Value)| {
                let a = super::object::alpha_f32(lua, &a);
                let (r, g, b) = (
                    super::object::tonumber_f32(lua, &r),
                    super::object::tonumber_f32(lua, &g),
                    super::object::tonumber_f32(lua, &b),
                );
                // Alpha is staged 1.0 (`0x79f59b`) and never read back, so a bare colour call
                // resets it.
                edit(lua, &this, |fo| fo.color = Some([r, g, b, a]))
            },
        )?,
    )?;
    m.set(
        "GetTextColor",
        lua.create_function(|lua, this: Table| {
            let c = read(lua, &this)?.color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
            Ok((c[0], c[1], c[2], c[3]))
        })?,
    )?;
    // SetAlpha/GetAlpha are the text colour's alpha: a FontInstance has one packed colour
    // (`+0x58`) and no alpha of its own.
    m.set(
        "SetAlpha",
        lua.create_function(|lua, (this, a): (Table, f32)| {
            edit(lua, &this, |fo| {
                let c = fo.color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
                fo.color = Some([c[0], c[1], c[2], a]);
            })
        })?,
    )?;
    m.set(
        "GetAlpha",
        lua.create_function(|lua, this: Table| Ok(read(lua, &this)?.color.map_or(1.0, |c| c[3])))?,
    )?;

    // ── shadow ──────────────────────────────────────────────────────────────────────────────
    m.set(
        "SetShadowColor",
        lua.create_function(
            // Shape C on r, g, b (`Font:SetShadowColor 0x79f730`): never raises.
            |lua, (this, r, g, b, a): (Table, Value, Value, Value, Value)| {
                let a = super::object::alpha_f32(lua, &a);
                let (r, g, b) = (
                    crate::script::object::tonumber_f32(lua, &r),
                    crate::script::object::tonumber_f32(lua, &g),
                    crate::script::object::tonumber_f32(lua, &b),
                );
                edit(lua, &this, |fo| {
                    let offset = fo.shadow.map_or([0.0, 0.0], |s| s.offset);
                    fo.shadow = Some(FontShadow {
                        offset,
                        color: [r, g, b, a],
                    });
                })
            },
        )?,
    )?;
    m.set(
        "GetShadowColor",
        lua.create_function(|lua, this: Table| {
            let c = read(lua, &this)?
                .shadow
                .map_or([0.0, 0.0, 0.0, 1.0], |s| s.color);
            Ok((c[0], c[1], c[2], c[3]))
        })?,
    )?;
    m.set(
        "SetShadowOffset",
        lua.create_function(|lua, (this, x, y): (Table, f32, f32)| {
            edit(lua, &this, |fo| {
                let color = fo.shadow.map_or([0.0, 0.0, 0.0, 1.0], |s| s.color);
                fo.shadow = Some(FontShadow {
                    offset: [x, y],
                    color,
                });
            })
        })?,
    )?;
    m.set(
        "GetShadowOffset",
        lua.create_function(|lua, this: Table| {
            let o = read(lua, &this)?.shadow.map_or([0.0, 0.0], |s| s.offset);
            Ok((o[0], o[1]))
        })?,
    )?;

    // ── line spacing ────────────────────────────────────────────────────────────────────────
    // The shared pair (`0x7a09f0` → `0x79fb40`, `0x7a0ab0` → `0x79fbe0`): `+0x50` as given, a
    // negative included; a FontString that inherits it clamps it.
    m.set(
        "SetSpacing",
        lua.create_function(|lua, (this, v): (Table, Value)| {
            let sp = super::font_block::spacing_arg(lua, &this, &v)?;
            edit(lua, &this, |fo| fo.spacing = Some(sp))
        })?,
    )?;
    m.set(
        "GetSpacing",
        lua.create_function(|lua, this: Table| Ok(read(lua, &this)?.spacing.unwrap_or(0.0)))?,
    )?;

    // ── justification ───────────────────────────────────────────────────────────────────────
    // Parsing is [`crate::justify`]'s, shared with the FontString pair: `SStrCmpI` over the whole
    // string, no trim. `None` means the font does not specify the axis (inheritance), not a
    // cleared axis; a fresh object reads CENTER/MIDDLE, the constructor default `0x212`.
    m.set(
        "SetJustifyH",
        lua.create_function(|lua, (this, j): (Table, String)| {
            let jh = match justify::parse_h(&j) {
                justify::Set::To(jh) => jh,
                justify::Set::Clears => return Ok(()),
                justify::Set::NoMatch => return Err(justify::usage_h("Font")),
            };
            edit(lua, &this, |fo| fo.justify_h = Some(jh))
        })?,
    )?;
    m.set(
        "GetJustifyH",
        lua.create_function(|lua, this: Table| {
            Ok(justify::name_h(
                read(lua, &this)?.justify_h.unwrap_or_default(),
            ))
        })?,
    )?;
    m.set(
        "SetJustifyV",
        lua.create_function(|lua, (this, j): (Table, String)| {
            let jv = match justify::parse_v(&j) {
                justify::Set::To(jv) => jv,
                justify::Set::Clears => return Ok(()),
                justify::Set::NoMatch => return Err(justify::usage_v("Font")),
            };
            edit(lua, &this, |fo| fo.justify_v = Some(jv))
        })?,
    )?;
    m.set(
        "GetJustifyV",
        lua.create_function(|lua, this: Table| {
            Ok(justify::name_v(
                read(lua, &this)?.justify_v.unwrap_or_default(),
            ))
        })?,
    )?;

    let meta = lua.create_table()?;
    // `__index` is the method table itself, not a function: a table index costs ~9 ns against
    // ~200 ns through a Rust dispatcher, and nothing mutates the table after this point.
    meta.set("__index", m.clone())?;
    lua.set_named_registry_value(REG_FONT_METHODS, m)?;
    lua.set_named_registry_value(REG_FONT_META, meta)?;

    lua.globals()
        .set("CreateFont", lua.create_function(create_font)?)?;
    Ok(())
}

/// `SetFontObject`/`CopyFontObject` on a Font: take the other object's paint wholesale.
fn copy_from(lua: &Lua, this: &Table, other: &Value, verb: &str) -> mlua::Result<()> {
    let src = resolve_required(verb, other)?;
    let paint = {
        let model = lua.app_data_ref::<Model>().expect("model");
        model.font_object(&src).cloned().ok_or_else(|| {
            mlua::Error::runtime(format!(
                "{verb}: no font object named '{src}' is registered"
            ))
        })?
    };
    edit(lua, this, |fo| *fo = paint)
}

/// The font a new plate or chat bubble takes: `FrameScript_GetText` on `NAMEPLATE_FONT`
/// (`0x703bf0`; plates `0x7cb3ee`/`0x7cb490`, bubbles `0x4b0b36`). Its `_G[key]` arm (`0x703c12` →
/// `0x704350`) gates on `lua_isstring` (`0x6f3510`) and reads `lua_tostring` (`0x6f3690`), so a
/// string or a number answers its text and any other value the shared "". `None` is that empty
/// path, which `SetFont` rejects (`0x44d05c`, `0x44d040` answers 0) and leaves the FontString with
/// no font.
pub(super) fn nameplate_font(lua: &Lua) -> Option<String> {
    let value = lua.globals().get::<Value>("NAMEPLATE_FONT").ok()?;
    let text = lua.coerce_string(value).ok()??;
    Some(text.to_string_lossy()).filter(|s| !s.is_empty())
}

/// `CreateFont(name)`: mint a font object, publish it as the global `name` and return it; addons
/// use both. An existing name returns that object unchanged and unrepublished (`0x7839ab`).
///
/// The reference's gate (`0x706048`) takes a number or any string, the empty one included, and
/// raises on anything else. A fresh object holds nothing, so it copies nothing ([`repaint`]).
fn create_font(lua: &Lua, name: Option<String>) -> mlua::Result<Table> {
    let name = name.ok_or_else(|| {
        mlua::Error::runtime("CreateFont: a font name is required (it becomes a global)")
    })?;
    {
        let mut model = lua.app_data_mut::<Model>().expect("model");
        model
            .font_objects_by_lower
            .entry(name.to_ascii_lowercase())
            .or_default();
    }
    let t = wrapper(lua, &name)?;
    publish_global(lua, &name, &t)?;
    Ok(t)
}
