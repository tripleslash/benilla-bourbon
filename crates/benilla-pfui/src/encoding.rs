//! pfUI's profile transport: ordinary CBOR/JSON data, zlib framing, and RFC 4648 Base64.

use std::collections::HashSet;
use std::io::Cursor;

use base64::Engine;
use ciborium::Value as Data;
use mlua::{Lua, Value};

const MAX_BYTES: usize = 8 * 1024 * 1024;
const MAX_DEPTH: usize = 64;

fn error(message: impl std::fmt::Display) -> mlua::Error {
    mlua::Error::runtime(format!("C_EncodingUtil: {message}"))
}

fn check_size(bytes: &[u8]) -> mlua::Result<()> {
    if bytes.len() > MAX_BYTES {
        Err(error("profile exceeds 8 MiB"))
    } else {
        Ok(())
    }
}

fn to_data(value: Value, seen: &mut HashSet<usize>, depth: usize) -> mlua::Result<Data> {
    if depth > MAX_DEPTH {
        return Err(error("profile nesting exceeds 64 levels"));
    }
    Ok(match value {
        Value::Nil => Data::Null,
        Value::Boolean(value) => Data::Bool(value),
        Value::Integer(value) => Data::Integer(value.into()),
        Value::Number(value) if value.is_finite() => {
            if value.fract() == 0.0 && value >= i64::MIN as f64 && value < -(i64::MIN as f64) {
                Data::Integer((value as i64).into())
            } else {
                Data::Float(value)
            }
        }
        Value::String(value) => {
            check_size(&value.as_bytes())?;
            match value.to_str() {
                Ok(text) => Data::Text(text.to_string()),
                Err(_) => Data::Bytes(value.as_bytes().to_vec()),
            }
        }
        Value::Table(table) => {
            let identity = table.to_pointer() as usize;
            if !seen.insert(identity) {
                return Err(error("cyclic table"));
            }
            let mut entries = Vec::new();
            for pair in table.pairs::<Value, Value>() {
                let (key, value) = pair?;
                entries.push((
                    to_data(key, seen, depth + 1)?,
                    to_data(value, seen, depth + 1)?,
                ));
                if entries.len() > MAX_BYTES {
                    return Err(error("too many profile entries"));
                }
            }
            seen.remove(&identity);
            // Nonempty contiguous 1-based integer keys form an array. Holes and mixed keys
            // remain a CBOR map so a Lua config's numeric keys are preserved.
            let array = !entries.is_empty() && entries.iter().all(|(key, _)| {
                matches!(key, Data::Integer(i) if i128::from(*i) >= 1 && i128::from(*i) <= entries.len() as i128)
            });
            if array {
                entries.sort_by_key(|(key, _)| {
                    if let Data::Integer(i) = key {
                        i128::from(*i)
                    } else {
                        0
                    }
                });
                Data::Array(entries.into_iter().map(|(_, value)| value).collect())
            } else {
                Data::Map(entries)
            }
        }
        _ => return Err(error("profile contains unsupported Lua value")),
    })
}

fn to_lua(lua: &Lua, value: Data, depth: usize) -> mlua::Result<Value> {
    if depth > MAX_DEPTH {
        return Err(error("profile nesting exceeds 64 levels"));
    }
    Ok(match value {
        Data::Null => Value::Nil,
        Data::Bool(value) => Value::Boolean(value),
        Data::Integer(value) => Value::Number(i128::from(value) as f64),
        Data::Float(value) if value.is_finite() => Value::Number(value),
        Data::Text(value) => Value::String(lua.create_string(value)?),
        Data::Bytes(value) => Value::String(lua.create_string(value)?),
        Data::Array(values) => {
            let table = lua.create_table()?;
            for (i, value) in values.into_iter().enumerate() {
                table.raw_set(i + 1, to_lua(lua, value, depth + 1)?)?;
            }
            Value::Table(table)
        }
        Data::Map(entries) => {
            let table = lua.create_table()?;
            for (key, value) in entries {
                let key = to_lua(lua, key, depth + 1)?;
                if !matches!(key, Value::Boolean(_) | Value::Number(_) | Value::String(_)) {
                    return Err(error("unsupported CBOR map key"));
                }
                table.raw_set(key, to_lua(lua, value, depth + 1)?)?;
            }
            Value::Table(table)
        }
        _ => return Err(error("unsupported CBOR value")),
    })
}

fn to_json(value: Data) -> mlua::Result<serde_json::Value> {
    use serde_json::Value as Json;
    Ok(match value {
        Data::Null => Json::Null,
        Data::Bool(value) => Json::Bool(value),
        Data::Integer(value) => {
            let value = i64::try_from(i128::from(value)).map_err(error)?;
            Json::Number(value.into())
        }
        Data::Float(value) => Json::Number(
            serde_json::Number::from_f64(value).ok_or_else(|| error("nonfinite number"))?,
        ),
        Data::Text(value) => Json::String(value),
        Data::Array(values) => Json::Array(
            values
                .into_iter()
                .map(to_json)
                .collect::<mlua::Result<_>>()?,
        ),
        Data::Map(entries) => {
            let mut object = serde_json::Map::new();
            for (key, value) in entries {
                let Data::Text(key) = key else {
                    return Err(error("JSON object keys must be strings"));
                };
                object.insert(key, to_json(value)?);
            }
            Json::Object(object)
        }
        _ => return Err(error("unsupported JSON value")),
    })
}

fn json_to_data(value: serde_json::Value) -> Data {
    use serde_json::Value as Json;
    match value {
        Json::Null => Data::Null,
        Json::Bool(value) => Data::Bool(value),
        Json::Number(value) => match value.as_i64() {
            Some(value) => Data::Integer(value.into()),
            None => Data::Float(value.as_f64().unwrap()),
        },
        Json::String(value) => Data::Text(value),
        Json::Array(values) => Data::Array(values.into_iter().map(json_to_data).collect()),
        Json::Object(entries) => Data::Map(
            entries
                .into_iter()
                .map(|(key, value)| (Data::Text(key), json_to_data(value)))
                .collect(),
        ),
    }
}

pub(super) fn install(lua: &Lua) -> mlua::Result<()> {
    let api = lua.create_table()?;
    api.set(
        "SerializeCBOR",
        lua.create_function(|lua, value: Value| {
            let value = to_data(value, &mut HashSet::new(), 0)?;
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(&value, &mut bytes).map_err(error)?;
            check_size(&bytes)?;
            lua.create_string(bytes)
        })?,
    )?;
    api.set(
        "DeserializeCBOR",
        lua.create_function(|lua, bytes: mlua::String| {
            let bytes = bytes.as_bytes();
            check_size(&bytes)?;
            let mut reader = Cursor::new(bytes.as_ref());
            let value = ciborium::de::from_reader_with_recursion_limit(&mut reader, MAX_DEPTH)
                .map_err(error)?;
            if reader.position() as usize != bytes.len() {
                return Err(error("trailing CBOR data"));
            }
            to_lua(lua, value, 0)
        })?,
    )?;
    api.set(
        "SerializeJSON",
        lua.create_function(|lua, value: Value| {
            let value = to_json(to_data(value, &mut HashSet::new(), 0)?)?;
            let bytes = serde_json::to_vec(&value).map_err(error)?;
            check_size(&bytes)?;
            lua.create_string(bytes)
        })?,
    )?;
    api.set(
        "DeserializeJSON",
        lua.create_function(|lua, bytes: mlua::String| {
            let bytes = bytes.as_bytes();
            check_size(&bytes)?;
            let value = serde_json::from_slice(&bytes).map_err(error)?;
            to_lua(lua, json_to_data(value), 0)
        })?,
    )?;
    api.set(
        "CompressString",
        lua.create_function(|lua, bytes: mlua::String| {
            check_size(&bytes.as_bytes())?;
            lua.create_string(miniz_oxide::deflate::compress_to_vec_zlib(
                &bytes.as_bytes(),
                6,
            ))
        })?,
    )?;
    api.set(
        "DecompressString",
        lua.create_function(|lua, bytes: mlua::String| {
            check_size(&bytes.as_bytes())?;
            let result = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
                &bytes.as_bytes(),
                MAX_BYTES,
            )
            .map_err(error)?;
            lua.create_string(result)
        })?,
    )?;
    api.set(
        "EncodeBase64",
        lua.create_function(|lua, bytes: mlua::String| {
            check_size(&bytes.as_bytes())?;
            lua.create_string(base64::engine::general_purpose::STANDARD.encode(bytes.as_bytes()))
        })?,
    )?;
    api.set(
        "DecodeBase64",
        lua.create_function(|lua, bytes: mlua::String| {
            check_size(&bytes.as_bytes())?;
            let result = base64::engine::general_purpose::STANDARD
                .decode(bytes.as_bytes())
                .map_err(error)?;
            lua.create_string(result)
        })?,
    )?;
    lua.globals().set("C_EncodingUtil", api)
}
