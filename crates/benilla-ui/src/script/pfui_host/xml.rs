//! The FrameXML a ClassicAPI client reads beyond 1.12's, in pfUI's VM: `parentKey` and
//! `parentArray`, `mixin`, `<KeyValues>`, a script's `method=`, `function=` and `inherit=`,
//! intrinsic templates (`<EventFrame>`, `CreateFrame("DropdownButton")`), `inherits` lists,
//! `relativeKey`, `useParentLevel` and a texture's `atlas`. The loader calls each hook; in a VM the
//! extension is not installed in, every hook does nothing.

use crate::framexml::Element;
use mlua::{Function, Lua, ObjectLike, Table, Value};

/// The handlers' Lua side: a method or global looked up when the script runs, and two handlers
/// run in order.
const HELPERS: &str = r#"
local getglobal, unpack, error = getglobal, unpack, error
local helpers = {}
function helpers.method(name)
  return function(self, ...)
    local method = self[name]
    if not method then error("script method " .. name .. " not found", 2) end
    return method(self, unpack(arg, 1, arg.n))
  end
end
function helpers.global(name)
  return function(...)
    local func = getglobal(name)
    if not func then error("script function " .. name .. " not found", 2) end
    return func(unpack(arg, 1, arg.n))
  end
end
function helpers.chain(first, second)
  return function(...)
    first(unpack(arg, 1, arg.n))
    return second(unpack(arg, 1, arg.n))
  end
end
return helpers
"#;

const REG_HELPERS: &str = "benilla.pfui.xml_helpers";
/// Each frame's intrinsic handlers until its levels have loaded: `frame -> { {script, func,
/// postcall}, ... }`.
const REG_INTRINSIC: &str = "benilla.pfui.intrinsic_scripts";

pub(crate) fn install(lua: &Lua) -> mlua::Result<()> {
    let helpers: Table = lua.load(HELPERS).set_name("@benilla-pfui/xml.lua").eval()?;
    lua.set_named_registry_value(REG_HELPERS, helpers)?;
    let pending = lua.create_table()?;
    let weak = lua.create_table()?;
    weak.set("__mode", "k")?;
    pending.set_metatable(Some(weak))?;
    lua.set_named_registry_value(REG_INTRINSIC, pending)?;
    let util = lua.create_table()?;
    util.set(
        "GetTemplateInfo",
        lua.create_function(|lua, name: Value| {
            let Value::String(name) = name else {
                return Ok(Value::Nil);
            };
            template_info(lua, &name.to_string_lossy())
        })?,
    )?;
    lua.globals().set("C_XMLUtil", util)
}

/// `C_XMLUtil.GetTemplateInfo`: a template's tag, its own `<Size>`, its own `<KeyValues>` as
/// written and its `inherits`, or `nil` for a name the registry lacks.
fn template_info(lua: &Lua, name: &str) -> mlua::Result<Value> {
    let el = {
        let model = lua
            .app_data_ref::<crate::script::ModelCell>()
            .expect("model");
        let templates = model.framexml_templates.borrow();
        match templates.get(name) {
            Some(el) => el.clone(),
            None => return Ok(Value::Nil),
        }
    };
    let info = lua.create_table()?;
    info.set("type", el.tag.as_str())?;
    let size = el
        .children
        .iter()
        .find(|c| c.tag.eq_ignore_ascii_case("Size"));
    let axis = |key: &str| -> f64 {
        size.and_then(|s| {
            s.attr(key).or_else(|| {
                s.children
                    .iter()
                    .find(|c| c.tag.eq_ignore_ascii_case("AbsDimension"))
                    .and_then(|d| d.attr(key))
            })
        })
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0.0)
    };
    info.set("width", axis("x"))?;
    info.set("height", axis("y"))?;
    let key_values = lua.create_table()?;
    for list in el
        .children
        .iter()
        .filter(|c| c.tag.eq_ignore_ascii_case("KeyValues"))
    {
        for entry in &list.children {
            let row = lua.create_table()?;
            row.set("key", entry.attr("key").unwrap_or(""))?;
            row.set("keyType", entry.attr("keyType").unwrap_or("string"))?;
            row.set("type", entry.attr("type").unwrap_or("string"))?;
            row.set("value", entry.attr("value").unwrap_or(""))?;
            key_values.raw_set(key_values.raw_len() + 1, row)?;
        }
    }
    info.set("keyValues", key_values)?;
    if let Some(inherits) = el.attr("inherits").filter(|i| !i.is_empty()) {
        info.set("inherits", inherits)?;
    }
    Ok(Value::Table(info))
}

pub(crate) fn active(lua: &Lua) -> bool {
    super::installed(lua).unwrap_or(false)
}

fn helper(lua: &Lua, name: &str) -> mlua::Result<Function> {
    lua.named_registry_value::<Table>(REG_HELPERS)?.get(name)
}

fn names(list: &str) -> impl Iterator<Item = &str> {
    list.split(',').map(str::trim).filter(|n| !n.is_empty())
}

/// An intrinsic template's base type and name, for a tag or `CreateFrame` type 1.12 lacks:
/// `<Frame name="EventFrame" intrinsic="true">` makes `EventFrame` a Frame.
pub(crate) fn intrinsic(lua: &Lua, kind: &str) -> Option<(String, String)> {
    if !active(lua) {
        return None;
    }
    let model = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model");
    let templates = model.framexml_templates.borrow();
    let el = templates.get(kind)?;
    el.attr_is_true("intrinsic")
        .then(|| (el.tag.clone(), el.name().unwrap_or(kind).to_string()))
}

/// Whether every name of a template list is registered, for `CreateFrame`'s check.
pub(crate) fn all_templates_known(lua: &Lua, list: &str) -> bool {
    let model = lua
        .app_data_ref::<crate::script::ModelCell>()
        .expect("model");
    let templates = model.framexml_templates.borrow();
    names(list).all(|name| templates.contains(name))
}

/// `mixin` and `secureMixin` of each level, the template's first: each named global table's
/// fields copied onto the object, before anything of the load reads them.
pub(crate) fn apply_mixins<'a>(
    lua: &Lua,
    levels: impl IntoIterator<Item = &'a Element>,
    object: &Table,
) -> mlua::Result<()> {
    if !active(lua) {
        return Ok(());
    }
    for level in levels {
        for attr in ["mixin", "secureMixin"] {
            let Some(list) = level.attr(attr) else {
                continue;
            };
            for name in names(list) {
                match lua.globals().get::<Value>(name)? {
                    Value::Table(mixin) => {
                        for pair in mixin.pairs::<Value, Value>() {
                            let (key, value) = pair?;
                            object.set(key, value)?;
                        }
                    }
                    _ => warn(lua, format!("{attr}=\"{name}\": no such table")),
                }
            }
        }
    }
    Ok(())
}

/// One `<KeyValues>` element: each `<KeyValue key value type keyType>` as a field of the object,
/// `type` one of `string` (the default), `number`, `boolean`, `global` or `nil`.
pub(crate) fn apply_key_values(
    lua: &Lua,
    key_values: &Element,
    object: &Table,
) -> mlua::Result<()> {
    for entry in &key_values.children {
        let Some(key) = entry.attr("key") else {
            continue;
        };
        let number = entry
            .attr("keyType")
            .filter(|t| t.eq_ignore_ascii_case("number"))
            .and_then(|_| key.trim().parse::<f64>().ok());
        let key = match number {
            Some(n) => Value::Number(n),
            None => Value::String(lua.create_string(key)?),
        };
        let raw = entry.attr("value").unwrap_or("");
        let value = match entry.attr("type").map(str::to_ascii_lowercase).as_deref() {
            Some("number") => raw.trim().parse::<f64>().map_or(Value::Nil, Value::Number),
            Some("boolean") => Value::Boolean(raw.eq_ignore_ascii_case("true")),
            Some("global") => lua.globals().get::<Value>(raw)?,
            Some("nil") => Value::Nil,
            _ => Value::String(lua.create_string(raw)?),
        };
        object.set(key, value)?;
    }
    Ok(())
}

/// The last level's value of `attr`, the object's own node first.
fn own_attr<'a>(levels: &[&'a Element], attr: &str) -> Option<&'a str> {
    levels.iter().rev().find_map(|level| level.attr(attr))
}

/// A frame just made under `parent`: its `parentKey` and `parentArray` on the parent, and
/// `useParentLevel`, before its contents and its `OnLoad`.
pub(crate) fn frame_created(
    lua: &Lua,
    levels: &[&Element],
    frame: &Table,
    parent: Option<&Table>,
) -> mlua::Result<()> {
    if !active(lua) {
        return Ok(());
    }
    let Some(parent) = parent else {
        return Ok(());
    };
    link_to_parent(lua, levels, frame, parent)?;
    if own_attr(levels, "useParentLevel").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
        let level: Value = parent.call_method("GetFrameLevel", ())?;
        frame.call_method::<()>("SetFrameLevel", level)?;
    }
    Ok(())
}

/// A texture or font string just loaded in `frame`: its `parentKey` and `parentArray`, then a
/// texture's `atlas`, after the rest of the element.
pub(crate) fn region_created(
    lua: &Lua,
    levels: &[&Element],
    region: &Table,
    frame: &Table,
) -> mlua::Result<()> {
    if !active(lua) {
        return Ok(());
    }
    link_to_parent(lua, levels, region, frame)?;
    if let Some(atlas) = own_attr(levels, "atlas").filter(|a| !a.is_empty()) {
        let size = own_attr(levels, "useAtlasSize").is_some_and(|v| v.eq_ignore_ascii_case("true"));
        if region.get::<Value>("SetAtlas")?.is_function() {
            region.call_method::<()>("SetAtlas", (atlas, size))?;
        }
    }
    Ok(())
}

fn link_to_parent(
    lua: &Lua,
    levels: &[&Element],
    object: &Table,
    parent: &Table,
) -> mlua::Result<()> {
    if let Some(key) = own_attr(levels, "parentKey").filter(|k| !k.is_empty()) {
        parent.set(key, object.clone())?;
    }
    if let Some(key) = own_attr(levels, "parentArray").filter(|k| !k.is_empty()) {
        let list = match parent.get::<Value>(key)? {
            Value::Table(list) => list,
            _ => {
                let list = lua.create_table()?;
                parent.set(key, list.clone())?;
                list
            }
        };
        list.raw_set(list.raw_len() + 1, object.clone())?;
    }
    Ok(())
}

/// A script element's handler from `method=` (the object's method) or `function=` (a global),
/// looked up when it runs; `None` for a body handler.
pub(crate) fn attr_handler(lua: &Lua, handler: &Element) -> Option<mlua::Result<Function>> {
    if !active(lua) {
        return None;
    }
    if let Some(name) = handler.attr("method").filter(|n| !n.is_empty()) {
        return Some(helper(lua, "method").and_then(|f| f.call(name)));
    }
    let name = handler.attr("function").filter(|n| !n.is_empty())?;
    Some(helper(lua, "global").and_then(|f| f.call(name)))
}

/// A level's handler as the extension's VM loads it: `method=` or `function=` over the body
/// handler `func`, then `inherit=` over the handler the levels before set.
pub(crate) fn handler(
    lua: &Lua,
    handler: &Element,
    frame: &Table,
    func: Option<Function>,
) -> mlua::Result<Option<Function>> {
    let func = match attr_handler(lua, handler) {
        Some(made) => Some(made?),
        None => func,
    };
    if !active(lua) || handler.attr("inherit").is_none() {
        return Ok(func);
    }
    let before = frame.call_method("GetScript", handler.tag.as_str())?;
    inherit(lua, handler, before, func)
}

/// `inherit="prepend"` runs the handler the levels before set, then this one; `"append"` the
/// other way round. Without it, this one replaces it, as in 1.12.
fn inherit(
    lua: &Lua,
    handler: &Element,
    before: Option<Function>,
    func: Option<Function>,
) -> mlua::Result<Option<Function>> {
    let (Some(before), Some(func)) = (before, func.clone()) else {
        return Ok(func);
    };
    let chain = helper(lua, "chain")?;
    match handler
        .attr("inherit")
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("prepend") => chain.call((before, func)).map(Some),
        Some("append") => chain.call((func, before)).map(Some),
        _ => Ok(Some(func)),
    }
}

/// An intrinsic template's handlers wait until the levels after it have loaded: each then runs
/// before the frame's own (`intrinsicOrder="precall"`, the default) or after it (`"postcall"`).
pub(crate) fn capture_intrinsic(
    lua: &Lua,
    scripts: &Element,
    frame: &Table,
    compile: &mut dyn FnMut(&Element, &str) -> Option<Function>,
) -> mlua::Result<()> {
    let pending: Table = lua.named_registry_value(REG_INTRINSIC)?;
    let list = match pending.raw_get::<Value>(frame.clone())? {
        Value::Table(list) => list,
        _ => {
            let list = lua.create_table()?;
            pending.raw_set(frame.clone(), list.clone())?;
            list
        }
    };
    for handler in &scripts.children {
        let func = match attr_handler(lua, handler) {
            Some(made) => Some(made?),
            None => compile(handler, &handler.tag),
        };
        let Some(func) = func else {
            continue;
        };
        let post = handler
            .attr("intrinsicOrder")
            .is_some_and(|o| o.eq_ignore_ascii_case("postcall"));
        let entry = lua.create_table()?;
        entry.raw_set(1, handler.tag.as_str())?;
        entry.raw_set(2, func)?;
        entry.raw_set(3, post)?;
        list.raw_set(list.raw_len() + 1, entry)?;
    }
    Ok(())
}

/// Join the captured intrinsic handlers with the frame's own, once its levels have loaded and
/// before its `OnLoad` fires.
pub(crate) fn join_intrinsic(
    lua: &Lua,
    frame: &Table,
    onload: &mut Option<Function>,
) -> mlua::Result<()> {
    if !active(lua) {
        return Ok(());
    }
    let pending: Table = lua.named_registry_value(REG_INTRINSIC)?;
    let Value::Table(list) = pending.raw_get::<Value>(frame.clone())? else {
        return Ok(());
    };
    pending.raw_set(frame.clone(), Value::Nil)?;
    let chain = helper(lua, "chain")?;
    for entry in list.sequence_values::<Table>() {
        let entry = entry?;
        let script: String = entry.raw_get(1)?;
        let intrinsic: Function = entry.raw_get(2)?;
        let post: bool = entry.raw_get(3)?;
        let is_load = script.eq_ignore_ascii_case("OnLoad");
        let own: Option<Function> = if is_load {
            onload.take()
        } else {
            frame.call_method("GetScript", script.as_str())?
        };
        let joined = match own {
            None => intrinsic,
            Some(own) if post => chain.call((own, intrinsic))?,
            Some(own) => chain.call((intrinsic, own))?,
        };
        if is_load {
            *onload = Some(joined);
        } else {
            frame.call_method::<()>("SetScript", (script.as_str(), joined))?;
        }
    }
    Ok(())
}

/// The object a `relativeKey` names from the anchored `object`: `$parent` steps to the parent,
/// any other part reads that field, so `$parent.ScrollBar` is the sibling under that key.
pub(crate) fn relative_key(lua: &Lua, object: &Table, key: &str) -> Option<Table> {
    if !active(lua) {
        return None;
    }
    let mut at = object.clone();
    for part in key.split('.').map(str::trim).filter(|p| !p.is_empty()) {
        let next: Value = if part.eq_ignore_ascii_case("$parent") {
            at.call_method("GetParent", ()).ok()?
        } else {
            at.get(part).ok()?
        };
        match next {
            Value::Table(t) => at = t,
            _ => return None,
        }
    }
    Some(at)
}

/// An `<Anchor>` that gives its offset as its own `x` and `y`, without an `<Offset>`.
pub(crate) fn anchor_offset(lua: &Lua, anchor: &Element) -> Option<Element> {
    (active(lua) && (anchor.attr("x").is_some() || anchor.attr("y").is_some()))
        .then(|| anchor.clone())
}

/// Whether a length node gives only some of its keys, which ClassicAPI reads with the rest 0
/// where 1.12 refuses the node.
pub(crate) fn partial_length(lua: &Lua, el: &Element, keys: &[&str]) -> bool {
    active(lua) && keys.iter().any(|k| el.attr(k).is_some())
}

fn warn(lua: &Lua, line: String) {
    lua.app_data_mut::<crate::script::ModelCell>()
        .expect("model")
        .record_warning(line);
}
