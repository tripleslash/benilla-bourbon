//! ClassicAPI 1.20's base-library additions that read the VM's stack, and `newproxy`.

use mlua::benilla_lua::value::V;
use mlua::benilla_lua::vm::{LuaError, Vm};
use mlua::benilla_lua::Value;
use mlua::Lua;

/// The tag of a `newproxy` userdata, which no host object carries.
const PROXY_TAG: u64 = 0x7072_6f78_7900_0001;

/// `newproxy([proxy])`: a new empty userdata, with no metatable (`false` or none), a new empty
/// one (`true`), or the one of a userdata `newproxy(true)` made; anything else is an error.
fn newproxy(vm: &mut Vm, nargs: usize) -> Result<usize, LuaError> {
    let meta = match vm.arg(nargs, 0).unpack() {
        V::Nil | V::Bool(false) => None,
        V::Bool(true) => Some(vm.heap.new_table(0, 0)),
        V::User(u) if vm.heap.user_tag(u) == Some(PROXY_TAG) && vm.heap.user(u).meta.is_some() => {
            vm.heap.user(u).meta
        }
        _ => return Err(vm.arg_error(1, "boolean or proxy expected")),
    };
    let proxy = vm.new_user(PROXY_TAG, meta);
    vm.push(proxy);
    Ok(1)
}

/// Where a Lua function is defined: its chunk's short name and its first line; nothing for a
/// native. A native, so an error handler reads it while the raise is in flight.
fn function_source(vm: &mut Vm, nargs: usize) -> Result<usize, LuaError> {
    let V::Func(f) = vm.arg(nargs, 0).unpack() else {
        return Ok(0);
    };
    let (short, line) = match vm.heap.func(f) {
        mlua::benilla_lua::heap::FuncObj::Lua(l) => {
            let proto = l.proto();
            (
                mlua::benilla_lua::chunkid_n(&proto.source, 60),
                proto.line_defined,
            )
        }
        _ => return Ok(0),
    };
    let short = vm.str(&short);
    vm.push(short);
    vm.push(Value::Num(f64::from(line)));
    Ok(2)
}

/// `level` (counted from this native, as `debugstack` counts) as the count of its locals and of
/// its upvalues, then each name and value; nothing for a level with none to read.
fn frame_variables(vm: &mut Vm, nargs: usize) -> Result<usize, LuaError> {
    let level = vm.arg(nargs, 0).as_num().unwrap_or(1.0).max(0.0) as usize;
    let Some((locals, upvalues)) = vm.frame_variables(level) else {
        return Ok(0);
    };
    let count = locals.len() + upvalues.len();
    vm.push(Value::Num(locals.len() as f64));
    vm.push(Value::Num(upvalues.len() as f64));
    for (name, value) in locals.into_iter().chain(upvalues) {
        let name = vm.str(&name);
        vm.push(name);
        vm.push(value);
    }
    Ok(2 + 2 * count)
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    // `GetCallstackHeight()`: the call levels on the stack, the caller being level 1 and a native
    // function a level, as `debugstack` counts them.
    lua.globals().set(
        "GetCallstackHeight",
        lua.create_function(|lua, ()| {
            let mut height = 0usize;
            while lua.inspect_stack(height + 1, |_| ()).is_some() {
                height += 1;
            }
            Ok(height)
        })?,
    )?;
    lua.globals()
        .set("newproxy", lua.create_native("newproxy", newproxy)?)?;
    // Where a Lua function is defined, for `debuglocals`: its chunk and line; nothing for a native.

    lua.load(include_str!("baselib.lua"))
        .set_name("@benilla-pfui/baselib.lua")
        .call::<()>((
            lua.create_native("debuglocals", frame_variables)?,
            lua.create_native("debuglocals", function_source)?,
        ))
}
