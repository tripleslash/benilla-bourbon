use mlua::{Lua, MultiValue, Table, Value, Variadic};

/// The Lua 5.1 functions pfUI's code calls that the 1.12 libraries lack, each set only where the
/// library has no entry of that name: `string.match` rides the 5.1 matcher `string.find` already
/// is, and `gmatch` is `gfind`, its 5.1 form. The one stock binding replaced is `unpack`, by its
/// 5.1 form, which is the stock function when called without a range.
const LUA51_LIBS: &str = r#"
local find, getn = string.find, table.getn
if string.match == nil then
  string.match = function(s, pattern, init)
    local r = { find(s, pattern, init) }
    local n = getn(r)
    if n == 0 then return nil end
    if n == 2 then return string.sub(s, r[1], r[2]) end
    local caps = {}
    for i = 3, n do caps[i - 2] = r[i] end
    return unpack(caps)
  end
end
if string.gmatch == nil then string.gmatch = string.gfind end
-- 5.1's `unpack(t, i, j)`: 1.12's takes the table alone and stops at its first nil when it has
-- no `n`, so `unpack(values, 1, n)` would drop what follows a nil argument. With no range it is
-- the stock function.
local unpack50 = unpack
unpack = function(t, i, j)
  if i == nil and j == nil then return unpack50(t) end
  i = i or 1
  j = j or getn(t)
  local out = { n = j - i + 1 }
  if out.n < 1 then return end
  for k = i, j do out[k - i + 1] = t[k] end
  return unpack50(out)
end
"#;

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    // The parser takes 5.1's `...`, `#` and `%` on this state only; every other client keeps the
    // 1.12 grammar.
    lua.set_pfui_lua51(true);
    // 5.1's `_G`, the globals table itself, which 1.12's client opens without (benilla #1792)
    // and pfUI's code indexes throughout.
    if lua.globals().raw_get::<Value>("_G")?.is_nil() {
        lua.globals().raw_set("_G", lua.globals())?;
    }
    lua.load(LUA51_LIBS)
        .set_name("=benilla-pfui lua51")
        .exec()?;
    let string: Table = lua.globals().get("string")?;
    // 5.1's `luaopen_string` gives every string the metatable `{ __index = string }`, which makes
    // `s:lower()` a call; 1.12's strings have none. The registry keeps the table alive.
    let meta = lua.create_table()?;
    meta.raw_set("__index", string.clone())?;
    lua.set_named_registry_value("benilla.pfui.string_metatable", meta.clone())?;
    lua.set_pfui_string_metatable(Some(&meta));
    if string.raw_get::<Value>("reverse")?.is_nil() {
        string.raw_set(
            "reverse",
            lua.create_function(|lua, s: mlua::String| {
                let mut bytes = s.as_bytes().to_vec();
                bytes.reverse();
                lua.create_string(&bytes)
            })?,
        )?;
    }
    let math: Table = lua.globals().get("math")?;
    let unary: [Unary; 3] = [
        ("cosh", f64::cosh),
        ("sinh", f64::sinh),
        ("tanh", f64::tanh),
    ];
    for (name, f) in unary {
        if math.raw_get::<Value>(name)?.is_nil() {
            math.raw_set(name, lua.create_function(move |_, x: f64| Ok(f(x)))?)?;
        }
    }
    if math.raw_get::<Value>("fmod")?.is_nil() {
        // C's `fmod`, the result taking the dividend's sign, as 5.0's `math.mod` is.
        math.raw_set(
            "fmod",
            lua.create_function(|_, (a, b): (f64, f64)| Ok(a % b))?,
        )?;
    }
    if math.raw_get::<Value>("modf")?.is_nil() {
        math.raw_set(
            "modf",
            lua.create_function(|_, x: f64| {
                let int = x.trunc();
                let frac = if x.is_infinite() { 0.0 } else { x - int };
                Ok((int, frac))
            })?,
        )?;
    }
    if math.raw_get::<Value>("huge")?.is_nil() {
        math.raw_set("huge", f64::INFINITY)?;
    }
    lua.globals().set(
        "select",
        lua.create_function(|_, values: Variadic<Value>| {
            let Some(index) = values.first() else {
                return Err(mlua::Error::runtime("select: index expected"));
            };
            let count = values.len() - 1;
            if matches!(index, Value::String(s) if s.as_bytes().as_ref() == b"#") {
                return Ok(MultiValue::from_vec(vec![Value::Integer(count as i64)]));
            }
            let index = match index {
                Value::Integer(i) => *i,
                Value::Number(n) if n.is_finite() => *n as i64,
                Value::String(s) => {
                    let n = s
                        .to_str()?
                        .parse::<f64>()
                        .map_err(|_| mlua::Error::runtime("select: invalid index"))?;
                    if !n.is_finite() {
                        return Err(mlua::Error::runtime("select: invalid index"));
                    }
                    n as i64
                }
                _ => return Err(mlua::Error::runtime("select: invalid index")),
            };
            let first = if index < 0 {
                count as i64 + index + 1
            } else {
                index
            };
            if first < 1 {
                return Err(mlua::Error::runtime("select: index out of range"));
            }
            Ok(MultiValue::from_vec(
                values.into_iter().skip(first as usize).collect(),
            ))
        })?,
    )?;
    Ok(())
}

/// A math function by its global name.
type Unary = (&'static str, fn(f64) -> f64);
