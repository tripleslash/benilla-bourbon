//! The race/class/faction subset pfUI needs, read from the owner's localized DBCs.

use std::collections::BTreeMap;
use std::io::Cursor;

use benilla_dbc::{DbcParser, FieldType, Record, RecordSet, Schema, SchemaField, StringRef, Value};
use mlua::{Lua, Table};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Race {
    pub id: u32,
    pub name: String,
    pub file: String,
    pub faction: Option<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct Catalog {
    pub classes: BTreeMap<String, String>,
    /// Class tokens in ChrClasses.dbc ID order.
    pub class_order: Vec<String>,
    pub races: BTreeMap<u32, Race>,
}

impl Catalog {
    /// `read` is the host's read-only MPQ provider. Locale slots match the 1.12 DBC convention;
    /// an empty translation falls back to the first populated name.
    pub fn read(
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
        locale: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let slot = [
            "enUS", "koKR", "frFR", "deDE", "zhCN", "zhTW", "esES", "esMX",
        ]
        .iter()
        .position(|candidate| *candidate == locale)
        .unwrap_or(0);
        let classes = rows(read, "ChrClasses", 17)?;
        let races = rows(read, "ChrRaces", 29)?;
        let factions = rows(read, "FactionTemplate", 14)?;
        let groups = rows(read, "FactionGroup", 12)?;
        let mut names = BTreeMap::new();
        let mut ordered = Vec::new();
        for row in classes.records() {
            let token = string(&classes, row, 14)?;
            let name = localized(&classes, row, 5, slot)?;
            if !token.is_empty() && !name.is_empty() {
                ordered.push((number(row, 0)?, token.clone()));
                names.insert(token, name);
            }
        }
        ordered.sort();
        let mut result = BTreeMap::new();
        for row in races.records() {
            let id = number(row, 0)?;
            let template = number(row, 2)?;
            let mask = factions
                .records()
                .iter()
                .find(|r| number(r, 0).ok() == Some(template))
                .map(|r| number(r, 3))
                .transpose()?
                .unwrap_or(0);
            let mut faction = None;
            for group in groups.records() {
                let bit = number(group, 1)?;
                let name = localized(&groups, group, 3, slot)?;
                if bit < 32 && mask & (1 << bit) != 0 && !name.is_empty() {
                    faction = Some((string(&groups, group, 2)?, name));
                    break;
                }
            }
            result.insert(
                id,
                Race {
                    id,
                    name: localized(&races, row, 17, slot)?,
                    file: string(&races, row, 15)?,
                    faction,
                },
            );
        }
        Ok(Self {
            classes: names,
            class_order: ordered.into_iter().map(|(_, token)| token).collect(),
            races: result,
        })
    }

    pub fn install(&self, lua: &Lua) -> mlua::Result<()> {
        if !lua.globals().get::<mlua::Value>("C_CreatureInfo")?.is_nil() {
            return Err(mlua::Error::runtime("C_CreatureInfo already installed"));
        }
        let api = lua.create_table()?;
        let races = self.races.clone();
        api.set(
            "GetRaceInfo",
            lua.create_function(move |lua, id: mlua::Value| {
                let id = numeric_id(lua, id)?;
                let Some(race) = id.and_then(|id| races.get(&id)) else {
                    return Ok(None::<Table>);
                };
                let table = lua.create_table()?;
                table.set("raceID", race.id)?;
                table.set("raceName", race.name.as_str())?;
                table.set("clientFileString", race.file.as_str())?;
                Ok(Some(table))
            })?,
        )?;
        let races = self.races.clone();
        api.set(
            "GetFactionInfo",
            lua.create_function(move |lua, id: mlua::Value| {
                let id = numeric_id(lua, id)?;
                let Some((tag, name)) = id
                    .and_then(|id| races.get(&id))
                    .and_then(|r| r.faction.as_ref())
                else {
                    return Ok(None::<Table>);
                };
                let table = lua.create_table()?;
                table.set("groupTag", tag.as_str())?;
                table.set("name", name.as_str())?;
                Ok(Some(table))
            })?,
        )?;
        api.set(
            "GetCreatureID",
            lua.create_function(benilla_ui::script::pfui_host::units::creature_id)?,
        )?;
        // The DBC's class IDs order the list; the reference client has no such table.
        lua.globals()
            .set("CLASS_SORT_ORDER", self.class_order.clone())?;
        lua.globals().set("C_CreatureInfo", api)
    }
}

pub(crate) fn numeric_id(lua: &Lua, id: mlua::Value) -> mlua::Result<Option<u32>> {
    Ok(lua
        .coerce_number(id)?
        .filter(|n| n.is_finite() && *n >= 1.0 && *n <= f64::from(u32::MAX))
        .map(|n| n as u32))
}

pub(crate) fn rows(
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
    name: &str,
    fields: usize,
) -> Result<RecordSet, Box<dyn std::error::Error>> {
    let path = format!("DBFilesClient/{name}.dbc");
    let bytes = read(&path).ok_or_else(|| format!("missing {path}"))?;
    let mut schema = Schema::new(name);
    for i in 0..fields {
        schema.add_field(SchemaField::new(format!("f{i}"), FieldType::UInt32));
    }
    Ok(DbcParser::parse(&mut Cursor::new(bytes.as_slice()))?
        .with_schema(schema)?
        .parse_records()?)
}

pub(crate) fn number(row: &Record, col: usize) -> Result<u32, Box<dyn std::error::Error>> {
    match row.get_value(col) {
        Some(Value::UInt32(n)) => Ok(*n),
        _ => Err("invalid DBC column".into()),
    }
}

pub(crate) fn string(
    rows: &RecordSet,
    row: &Record,
    col: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    Ok(rows.get_string(StringRef(number(row, col)?))?.to_string())
}

pub(crate) fn localized(
    rows: &RecordSet,
    row: &Record,
    start: usize,
    slot: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let preferred = string(rows, row, start + slot)?;
    if !preferred.is_empty() {
        return Ok(preferred);
    }
    for index in start..start + 8 {
        let name = string(rows, row, index)?;
        if !name.is_empty() {
            return Ok(name);
        }
    }
    Ok(String::new())
}
