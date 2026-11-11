//! Reproducible source and VM checks, deliberately distinct from gameplay verification.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use benilla_ui::script::{ScriptValue, UiScript};
use benilla_ui::toc::Toc;
use mlua::Value;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Failure {
    pub file: String,
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct Requirement {
    pub name: String,
    /// A static reference, not proof that a branch executes or the function is correct.
    pub files: Vec<String>,
    pub available_in_stock_vm: bool,
    pub available_in_compat_vm: bool,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub addon: String,
    pub source_revision: Option<String>,
    pub manifest: String,
    pub lua_files: usize,
    pub source_errors: Vec<Failure>,
    pub stock_syntax_errors: Vec<Failure>,
    pub compat_syntax_errors: Vec<Failure>,
    pub requirements: Vec<Requirement>,
    pub stock_ui_errors: Vec<Failure>,
    pub stock_ui_attempted: bool,
    pub bootstrap_attempted: bool,
    pub bootstrap_errors: Vec<Failure>,
    /// The addons loaded between the stock UI and the audited one (`inspect_with`), such as the
    /// ClassicAPI DLL's own Lua addon, and their load errors as `<folder>/<file>`.
    pub preloads: Vec<String>,
    pub preload_errors: Vec<Failure>,
    pub session_errors: Vec<String>,
    /// An audit never proves gameplay, rendering, or working inventory operations.
    pub gameplay_verified: bool,
}

/// Check the complete ClassicAPI manifest, without editing pfUI or its normal manifest.
/// `data` optionally loads the owner's stock FrameXML from their read-only MPQ chain.
pub fn inspect(root: &Path, data: Option<&Path>) -> Result<Report, Box<dyn std::error::Error>> {
    inspect_flavor(root, data, false)
}

/// Select the Turtle manifest explicitly when testing a matching patched installation.
pub fn inspect_flavor(
    root: &Path,
    data: Option<&Path>,
    turtle: bool,
) -> Result<Report, Box<dyn std::error::Error>> {
    inspect_with(root, data, turtle, &[])
}

/// As [`inspect_flavor`], with `preload` addon folders loaded after the stock UI and before the
/// audited addon, each from its flavor manifest ([`addon_manifest`]).
pub fn inspect_with(
    root: &Path,
    data: Option<&Path>,
    turtle: bool,
    preload: &[PathBuf],
) -> Result<Report, Box<dyn std::error::Error>> {
    let root = root.canonicalize()?;
    let manifest = if turtle {
        "pfUI_Turtle.toc"
    } else {
        "pfUI_ClassicAPI.toc"
    };
    let toc = Toc::parse(&benilla_ui::source::decode(&std::fs::read(
        root.join(manifest),
    )?));
    let mut files = Vec::new();
    let mut source_errors = Vec::new();
    let mut seen = BTreeSet::new();
    for file in &toc.files {
        collect(
            &root,
            &root,
            file,
            &mut seen,
            &mut files,
            &mut source_errors,
        );
    }
    let stock = UiScript::new()?;
    let mut compat = UiScript::new()?;
    compat.set_screen_size(1024.0, 768.0);
    // No client applies the calls the UI makes; the VM keeps them, as a test VM does.
    compat.keep_calls();
    crate::install(&compat)?;
    let mut stock_syntax_errors = Vec::new();
    let mut compat_syntax_errors = Vec::new();
    let mut names: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in &files {
        let source = std::fs::read(path)?;
        let name = relative(&root, path);
        for (vm, failures) in [
            (&stock, &mut stock_syntax_errors),
            (&compat, &mut compat_syntax_errors),
        ] {
            if let Err(error) = vm
                .lua()
                .load(benilla_ui::source::chunk(&source))
                .set_name(format!("@pfUI/{name}"))
                .set_mode(mlua::ChunkMode::Text)
                .into_function()
            {
                failures.push(Failure {
                    file: name.clone(),
                    error: error.to_string(),
                });
            }
        }
        for name_ref in namespace_references(&source) {
            names.entry(name_ref).or_default().insert(name.clone());
        }
    }
    let requirements = names
        .into_iter()
        .map(|(name, files)| Requirement {
            available_in_stock_vm: available(&stock, &name),
            available_in_compat_vm: available(&compat, &name),
            name,
            files: files.into_iter().collect(),
        })
        .collect();
    let mut report = Report {
        addon: "pfUI".into(),
        source_revision: std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|revision| revision.trim().to_string()),
        manifest: manifest.into(),
        lua_files: files.len(),
        source_errors,
        stock_syntax_errors,
        compat_syntax_errors,
        requirements,
        stock_ui_errors: Vec::new(),
        stock_ui_attempted: data.is_some(),
        bootstrap_attempted: false,
        bootstrap_errors: Vec::new(),
        preloads: Vec::new(),
        preload_errors: Vec::new(),
        session_errors: Vec::new(),
        gameplay_verified: false,
    };
    if let Some(data) = data {
        let chain = benilla_formats::Chain::open(data)?;
        crate::icons::Catalog::from_paths(chain.list()?.into_iter().map(|p| p.name))
            .feed(compat.lua());
        let catalog = crate::creature::Catalog::read(
            &|path| chain.read(path).ok(),
            &compat.eval::<String>("return GetLocale()")?,
        )?;
        catalog.install(compat.lua())?;
        crate::spell::Catalog::read(
            &|path| chain.read(path).ok(),
            &compat.eval::<String>("return GetLocale()")?,
        )?
        .install(compat.lua())?;
        crate::maps::Catalog::read(&|path| chain.read(path).ok())?.install(compat.lua())?;
        crate::install_class_names(
            &compat,
            catalog
                .classes
                .iter()
                .map(|(token, name)| (token.as_str(), name.as_str())),
        )?;
        for requirement in &mut report.requirements {
            requirement.available_in_compat_vm = available(&compat, &requirement.name);
        }
        compat.set_instruction_budget(200_000_000);
        load_stock(&compat, &chain, &mut report.stock_ui_errors)?;
        // Only a clean stock boot is a valid environment for assigning errors to the addon.
        if report.stock_ui_errors.is_empty() {
            for folder in preload {
                let folder = folder.canonicalize()?;
                let name = folder
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or("an addon folder needs a name")?
                    .to_string();
                let manifest = addon_manifest(&folder, &name, turtle)?;
                let toc = Toc::parse(&benilla_ui::source::decode(&std::fs::read(
                    folder.join(&manifest),
                )?));
                // A preload keeps going past an error, so one run lists all of them.
                for failure in load_addon(&compat, &folder, &name, &toc, false)? {
                    report.preload_errors.push(Failure {
                        file: format!("{name}/{}", failure.file),
                        error: failure.error,
                    });
                }
                compat.fire_event("ADDON_LOADED", vec![ScriptValue::Str(name.clone())]);
                report.preloads.push(format!("{name}/{manifest}"));
            }
            report.bootstrap_attempted = true;
            bootstrap(&mut compat, &root, &toc, &mut report)?;
        }
    }
    Ok(report)
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn collect(
    root: &Path,
    base: &Path,
    name: &str,
    seen: &mut BTreeSet<PathBuf>,
    files: &mut Vec<PathBuf>,
    failures: &mut Vec<Failure>,
) {
    let path = base.join(name.replace('\\', "/"));
    let path = match path.canonicalize() {
        Ok(path) if path.starts_with(root) => path,
        Ok(_) => {
            failures.push(Failure {
                file: name.into(),
                error: "path outside addon folder".into(),
            });
            return;
        }
        Err(error) => {
            failures.push(Failure {
                file: name.into(),
                error: error.to_string(),
            });
            return;
        }
    };
    if !seen.insert(path.clone()) {
        return;
    }
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lua"))
    {
        files.push(path);
        return;
    }
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let bytes = std::fs::read(&path)?;
        let text = benilla_ui::source::decode(&bytes);
        let document = roxmltree::Document::parse(&text)?;
        for node in document.descendants().filter(|node| node.is_element()) {
            if matches!(node.tag_name().name(), "Include" | "Script") {
                if let Some(file) = node.attribute("file") {
                    collect(root, path.parent().unwrap(), file, seen, files, failures);
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        failures.push(Failure {
            file: relative(root, &path),
            error: error.to_string(),
        });
    }
}

fn available(script: &UiScript, name: &str) -> bool {
    let mut parts = name.split('.');
    let Ok(mut value) = script.lua().globals().get::<Value>(parts.next().unwrap()) else {
        return false;
    };
    for part in parts {
        let Value::Table(table) = value else {
            return false;
        };
        let Ok(next) = table.get::<Value>(part) else {
            return false;
        };
        value = next;
    }
    matches!(value, Value::Function(_))
}

fn load_stock(
    script: &UiScript,
    chain: &benilla_formats::Chain,
    failures: &mut Vec<Failure>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Camera defaults read in stock dropdown OnLoads, and the resolution of this audit's
    // explicit virtual viewport. There is no renderer or character fixture.
    script.register_cvars([
        ("cameraSmoothStyle", "1"),
        ("cameraSmoothTrackingStyle", "1"),
        ("gxResolution", "1024x768"),
        ("useUiScale", "0"),
        ("uiScale", "1"),
        ("statusBarText", "0"),
        ("MusicVolume", "0.4"),
        ("EnableMusic", "1"),
    ]);
    // A player identity for the files that read it at load (Turtle's shop and transmog UI). Unit
    // reads go through the client's host, which an audit has none of.
    script.run(
        "local class, race, sex = UnitClass, UnitRace, UnitSex
         UnitClass = function(u) local a, b = class(u)
           if a == nil and u == 'player' then return 'Warrior', 'WARRIOR' end return a, b end
         UnitRace = function(u) local a, b = race(u)
           if a == nil and u == 'player' then return 'Human', 'Human' end return a, b end
         UnitSex = function(u) local s = sex(u)
           if s == nil and u == 'player' then return 2 end return s end",
    )?;
    let toc = Toc::parse(&benilla_ui::source::decode(
        &chain.read("Interface/FrameXML/FrameXML.toc")?,
    ));
    for file in toc.files {
        let path = format!("Interface/FrameXML/{}", file.replace('\\', "/"));
        let result = chain
            .read(&path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                if path.to_ascii_lowercase().ends_with(".lua") {
                    script
                        .run_chunk_named(&bytes, &format!("@{path}"))
                        .map_err(|error| error.to_string())
                } else {
                    let text = benilla_ui::source::decode(&bytes);
                    let document =
                        benilla_ui::framexml::parse(&text).map_err(|error| error.to_string())?;
                    let report = benilla_ui::loader::load_in(script, &document, &path, &|name| {
                        chain.read(name).ok()
                    });
                    let mut errors = report.errors;
                    errors.extend(
                        report
                            .missing_files
                            .into_iter()
                            .map(|name| format!("missing {name}")),
                    );
                    if errors.is_empty() {
                        Ok(())
                    } else {
                        Err(errors.join("\n"))
                    }
                }
            });
        if let Err(error) = result {
            failures.push(Failure { file: path, error });
        }
    }
    for error in script.errors() {
        failures.push(Failure {
            file: "stock FrameXML handler".into(),
            error,
        });
    }
    Ok(())
}

/// The manifest the ClassicAPI DLL picks for an addon folder: its flavor TOC when it has one.
pub fn addon_manifest(
    folder: &Path,
    name: &str,
    turtle: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let flavor = if turtle { "_Turtle" } else { "_ClassicAPI" };
    [format!("{name}{flavor}.toc"), format!("{name}.toc")]
        .into_iter()
        .find(|manifest| folder.join(manifest).is_file())
        .ok_or_else(|| format!("{name}: no TOC").into())
}

/// Load one addon's manifest files in order, under its addon scope. With `stop`, the first Lua
/// error ends the load, as later files of the same addon build on it.
fn load_addon(
    script: &UiScript,
    root: &Path,
    name: &str,
    toc: &Toc,
    stop: bool,
) -> Result<Vec<Failure>, Box<dyn std::error::Error>> {
    let mut failures = Vec::new();
    let _scope = benilla_ui::script::pfui_host::addon_scope(script.lua(), name)?;
    for file in toc
        .files
        .iter()
        .filter(|file| !crate::skipped_file(name, file))
    {
        let path = root.join(file.replace('\\', "/"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                failures.push(Failure {
                    file: file.clone(),
                    error: error.to_string(),
                });
                continue;
            }
        };
        if file.to_ascii_lowercase().ends_with(".lua") {
            if let Err(error) =
                script.run_chunk_named(&bytes, &benilla_ui::script::addon_chunk_name(name, file))
            {
                failures.push(Failure {
                    file: file.clone(),
                    error: error.to_string(),
                });
                if stop {
                    break;
                }
            }
        } else {
            let document = benilla_ui::framexml::parse(&benilla_ui::source::decode(&bytes))?;
            let loaded =
                benilla_ui::loader::load_in(script, &document, &file.replace('\\', "/"), &|name| {
                    let path = root.join(name).canonicalize().ok()?;
                    path.starts_with(root)
                        .then(|| std::fs::read(path).ok())
                        .flatten()
                });
            for error in loaded.errors {
                failures.push(Failure {
                    file: file.clone(),
                    error,
                });
            }
            for name in loaded.missing_files {
                failures.push(Failure {
                    file: file.clone(),
                    error: format!("missing {name}"),
                });
            }
        }
    }
    Ok(failures)
}

fn bootstrap(
    script: &mut UiScript,
    root: &Path,
    toc: &Toc,
    report: &mut Report,
) -> Result<(), Box<dyn std::error::Error>> {
    report.bootstrap_errors = load_addon(script, root, "pfUI", toc, true)?;
    // Do not fire session events after a bad load.
    if report.bootstrap_errors.is_empty() {
        script.fire_event("ADDON_LOADED", vec![ScriptValue::Str("pfUI".into())]);
        script.fire_event("VARIABLES_LOADED", vec![]);
        script.fire_event("PLAYER_LOGIN", vec![]);
        script.fire_event("PLAYER_ENTERING_WORLD", vec![]);
        script.tick(0.1);
    }
    report.session_errors = script.errors();
    Ok(())
}

/// Collect dotted ClassicAPI references outside comments and strings. Includes references used
/// as values as well as direct calls; dynamic lookups and global/widget APIs need runtime probes.
pub fn namespace_references(source: &[u8]) -> BTreeSet<String> {
    let tokens = tokens(source);
    let mut names = BTreeSet::new();
    for parts in tokens.windows(3) {
        if (parts[0].starts_with("C_") || parts[0] == "EventUtil")
            && parts[1] == "."
            && parts[2].as_bytes()[0].is_ascii_alphabetic()
        {
            names.insert(format!("{}.{}", parts[0], parts[2]));
        }
    }
    names
}

fn long_bracket(source: &[u8], start: usize) -> Option<(usize, usize)> {
    if source.get(start) != Some(&b'[') {
        return None;
    }
    let mut end = start + 1;
    while source.get(end) == Some(&b'=') {
        end += 1;
    }
    (source.get(end) == Some(&b'[')).then_some((end - start - 1, end + 1))
}

fn skip_long(source: &[u8], equals: usize, mut at: usize) -> usize {
    while at < source.len() {
        if source[at] == b']'
            && source
                .get(at + 1..at + 1 + equals)
                .is_some_and(|s| s.iter().all(|c| *c == b'='))
            && source.get(at + 1 + equals) == Some(&b']')
        {
            return at + 2 + equals;
        }
        at += 1;
    }
    at
}

fn tokens(source: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < source.len() {
        if source.get(at..at + 2) == Some(b"--") {
            at += 2;
            if let Some((equals, begin)) = long_bracket(source, at) {
                at = skip_long(source, equals, begin);
            } else {
                while at < source.len() && source[at] != b'\n' {
                    at += 1;
                }
            }
        } else if matches!(source[at], b'\'' | b'"') {
            let quote = source[at];
            at += 1;
            while at < source.len() {
                let byte = source[at];
                at += 1;
                if byte == b'\\' {
                    at = (at + 1).min(source.len());
                } else if byte == quote {
                    break;
                }
            }
            out.push("<string>".into());
        } else if let Some((equals, begin)) = long_bracket(source, at) {
            at = skip_long(source, equals, begin);
            out.push("<string>".into());
        } else if source[at].is_ascii_alphabetic() || source[at] == b'_' {
            let start = at;
            at += 1;
            while at < source.len() && (source[at].is_ascii_alphanumeric() || source[at] == b'_') {
                at += 1;
            }
            out.push(String::from_utf8_lossy(&source[start..at]).into_owned());
        } else if !source[at].is_ascii_whitespace() {
            out.push(char::from(source[at]).to_string());
            at += 1;
        } else {
            at += 1;
        }
    }
    out
}
