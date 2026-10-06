//! **Every boot-time reader of the chain either rebuilds with it or says why not.**
//!
//! The chain's data changes at runtime: entering a realm that advertises Turtle WoW 1.18.1 mounts
//! the Turtle layer, and leaving it unmounts it. A catalog built once at startup would then answer
//! with the other flavor's data for the rest of the session, which nothing would report. So a
//! catalog loader is registered in `benilla_assets::DataLoad` (through `add_data_loaders`), which
//! runs at startup and again on every data change; a `Startup` system that reads the chain is one
//! of the allowlisted exceptions below, each with its reason, or it fails this test.
//!
//! Read off the source rather than a built `App`, as the manifest and page-twin tests are: the full
//! plugin set needs a window and an install, and the question is about registration, not runtime.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `Startup` systems allowed to read the chain once, and why each is right to. The check fails on
/// an entry that stops reading the chain too, so the list cannot outlive its reasons.
const ALLOWED: &[(&str, &str)] = &[
    (
        "setup_loading_screen",
        "spawns the cover; LoadingScreens.dbc and the bar art, which no layer overrides",
    ),
    (
        "load_ink",
        "spawns the footprint lane; FootprintTextures.dbc, which no layer overrides",
    ),
    (
        "setup_sun",
        "sun and moon sprites, which no layer overrides",
    ),
    (
        "setup_wdl",
        "the startup continent's WDL; `stream_wdl` loads each map's on entry and after \
         `forget_wdl` on a data change",
    ),
    (
        "load_expectations",
        "a dev probe's fixture, read once per probe run",
    ),
];

fn sources(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("reading a source directory") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let name = path.file_name().unwrap().to_string_lossy();
            if !(name.ends_with("_tests.rs") || name == "tests.rs" || name.starts_with("test_")) {
                out.push(path);
            }
        }
    }
}

/// A file's text up to its `#[cfg(test)] mod …`, where test-only registrations live.
fn non_test(text: &str) -> &str {
    let mut at = 0;
    while let Some(i) = text[at..].find("#[cfg(test)]") {
        let after = text[at + i + "#[cfg(test)]".len()..].trim_start();
        if after.starts_with("mod ")
            || after.starts_with("pub mod ")
            || after.starts_with("pub(crate) mod ")
        {
            return &text[..at + i];
        }
        at += i + 1;
    }
    text
}

/// The text between the `(` at `open` and its matching `)`.
fn balanced(text: &str, open: usize) -> &str {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' | '{' => depth += 1,
            ')' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open + 1..open + i];
                }
            }
            _ => {}
        }
    }
    &text[open + 1..]
}

/// The system names a `Startup`/`PostStartup`/`PreStartup` registration names: every identifier
/// in its arguments that is not a schedule-config method or a set.
fn startup_systems(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find("add_systems(") {
        let open = at + i + "add_systems".len();
        let args = balanced(text, open);
        let args_trim = args.trim_start();
        at = open + 1;
        let Some(rest) = ["Startup", "PostStartup", "PreStartup"]
            .iter()
            .find_map(|s| {
                args_trim
                    .strip_prefix(s)
                    .filter(|r| r.trim_start().starts_with(','))
            })
        else {
            continue;
        };
        let mut word = String::new();
        let mut prev_dot = false;
        for c in rest.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
                word.push(c);
                continue;
            }
            if !word.is_empty() {
                let last = word.rsplit("::").next().unwrap_or(&word).to_string();
                let is_fn_like = last.chars().next().is_some_and(|f| f.is_ascii_lowercase());
                if is_fn_like && !prev_dot {
                    out.push(last);
                }
                word.clear();
            }
            prev_dot = c == '.';
        }
    }
    out
}

/// The system names an `add_data_loaders(…)` registration names, by the same reading.
fn data_loaders(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find("add_data_loaders(") {
        let open = at + i + "add_data_loaders".len();
        let args = balanced(text, open);
        at = open + 1;
        // `Startup, …` so the one reader serves both.
        out.extend(startup_systems(&format!("add_systems(Startup, {args})")));
    }
    out
}

/// The body of `fn name(` in `text`, if it is defined there.
fn fn_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("fn {name}(");
    let i = text.find(&needle)?;
    let brace = i + text[i..].find('{')?;
    Some(balanced(text, brace))
}

fn reads_chain(body: &str) -> bool {
    body.contains("lock_recover()")
        || body.contains("chain.read")
        || body.contains("&mut chain")
        || body.contains(".chain.lock")
}

#[test]
fn every_startup_reader_of_the_chain_is_a_data_loader_or_says_why_not() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let mut files = Vec::new();
    for krate in ["crates/benilla-app/src", "crates/benilla-world/src"] {
        sources(&root.join(krate), &mut files);
    }
    let texts: Vec<(PathBuf, String)> = files
        .into_iter()
        .map(|p| {
            let t = std::fs::read_to_string(&p).expect("reading a source file");
            (p, non_test(&t).to_string())
        })
        .collect();

    let mut offenders: BTreeMap<String, String> = BTreeMap::new();
    let mut seen_allowed = std::collections::BTreeSet::new();
    for (path, text) in &texts {
        for system in startup_systems(text) {
            let Some(body) = texts.iter().find_map(|(_, t)| fn_body(t, &system)) else {
                continue;
            };
            if !reads_chain(body) {
                continue;
            }
            if ALLOWED.iter().any(|(name, _)| *name == system) {
                seen_allowed.insert(system);
                continue;
            }
            offenders.insert(system, path.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "these Startup systems read the chain once and would keep the other flavor's data after a \
         realm switch — register them with `add_data_loaders`, or allowlist them with a reason: \
         {offenders:#?}"
    );
    let stale: Vec<&str> = ALLOWED
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !seen_allowed.contains(*name))
        .collect();
    assert!(
        stale.is_empty(),
        "allowlisted Startup chain readers that no longer exist or no longer read the chain: {stale:?}"
    );
}

/// **A second run of `DataLoad` is safe**: every loader replaces what it inserts, so it must not
/// spawn, or each realm switch would leave one more of its entities behind.
#[test]
fn no_data_loader_spawns() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let mut files = Vec::new();
    for krate in ["crates/benilla-app/src", "crates/benilla-world/src"] {
        sources(&root.join(krate), &mut files);
    }
    let texts: Vec<String> = files
        .into_iter()
        .map(|p| non_test(&std::fs::read_to_string(&p).expect("reading a source file")).to_string())
        .collect();
    let mut loaders = 0;
    let mut spawning = Vec::new();
    for text in &texts {
        for system in data_loaders(text) {
            let Some(body) = texts.iter().find_map(|t| fn_body(t, &system)) else {
                continue;
            };
            loaders += 1;
            if body.contains(".spawn(") || body.contains("spawn_batch(") {
                spawning.push(system);
            }
        }
    }
    assert!(
        loaders > 40,
        "the scan found only {loaders} loaders — it has gone blind"
    );
    assert!(
        spawning.is_empty(),
        "these DataLoad loaders spawn, so a realm switch would duplicate their entities — keep the \
         spawn in Startup and the chain read in the loader: {spawning:?}"
    );
}

#[test]
fn the_scanner_finds_systems_and_skips_config_methods() {
    let text =
        "app.add_systems(Startup, (ring::setup_ring, load_x.after(AssetSet::Open)).chain())\n\
                .add_systems(Update, not_this)\n\
                .add_systems(PostStartup, (a, b));";
    assert_eq!(
        startup_systems(text),
        ["setup_ring", "load_x", "a", "b"].map(String::from)
    );
}
