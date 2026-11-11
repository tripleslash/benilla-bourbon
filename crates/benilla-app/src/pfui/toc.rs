//! The TOC conventions a ClassicAPI client reads for every addon: the flavor TOC and bindings files
//! (`<Name>_Turtle.toc` on a Turtle client, then `<Name>_ClassicAPI.toc`, over the plain one), an
//! `## Interface:` list holding 11200, per-line load conditions and path variables, and
//! `## LoadSavedVariablesFirst`.

use benilla_ui::toc::Toc;
use std::path::Path;

/// The flavor suffixes in the order they win.
fn flavors() -> &'static [&'static str] {
    if cfg!(feature = "pfui-turtle") {
        &["_Turtle", "_ClassicAPI"]
    } else {
        &["_ClassicAPI"]
    }
}

fn find_file(dir: &Path, want: &str) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let name = entry.file_name();
        name.to_str()
            .is_some_and(|f| f.eq_ignore_ascii_case(want))
            .then(|| entry.path())
    })
}

/// The flavor TOC of addon `name` in `dir`, which wins over a plain `<Name>.toc` and makes the
/// folder an addon without one.
pub(crate) fn manifest(dir: &Path, name: &str) -> Option<String> {
    flavors().iter().find_map(|flavor| {
        let path = find_file(dir, &format!("{name}{flavor}.toc"))?;
        let bytes = std::fs::read(path).ok()?;
        Some(benilla_ui::source::decode(&bytes).into_owned())
    })
}

/// The bindings file an addon loads: a flavor one, else `Bindings.xml`, as `exists` finds them.
pub(crate) fn bindings(prefix: &str, exists: impl Fn(&str) -> bool) -> Option<String> {
    flavors()
        .iter()
        .map(|flavor| benilla_ui::loader::join_ref(prefix, &format!("Bindings{flavor}.xml")))
        .find(|path| exists(path))
}

/// The version the gate compares: 11200 when an `## Interface:` list holds it, wherever it stands.
pub(crate) fn interface(toc: &Toc) -> u32 {
    let client = benilla_ui::script::addon_gate::CLIENT_INTERFACE;
    if toc.interface_versions().contains(&client) {
        client
    } else {
        toc.interface_version()
    }
}

/// Whether the addon's saved variables load before its files (`## LoadSavedVariablesFirst: 1`).
pub(crate) fn saved_variables_first(toc: &Toc) -> bool {
    toc.directives.iter().any(|d| {
        d.raw().split_once(':').is_some_and(|(key, value)| {
            key.trim().eq_ignore_ascii_case("LoadSavedVariablesFirst") && value.trim() == "1"
        })
    })
}

/// The client's text locale code (`GetLocale()`), which `[TextLocale]` and
/// `[AllowLoadTextLocale]` read.
pub(crate) fn locale(script: &benilla_ui::script::UiScript) -> String {
    script
        .lua()
        .globals()
        .get::<benilla_ui::mlua::Function>("GetLocale")
        .and_then(|f| f.call(()))
        .unwrap_or_else(|_| "enUS".to_string())
}

/// A file line of `addon`'s TOC as it loads here: `None` when a condition drops it (or the
/// ClassicAPI addon's skip list does), else the path with its variables expanded. Every condition
/// on a line must pass; an unknown one drops it.
pub(crate) fn file_line(addon: &str, line: &str, locale: &str) -> Option<String> {
    let mut path = String::new();
    let mut rest = line;
    while let Some(open) = rest.find('[') {
        let close = open + rest[open..].find(']')?;
        path.push_str(&rest[..open]);
        let token = rest[open + 1..close].trim();
        let (word, list) = token.split_once(' ').unwrap_or((token, ""));
        let values: Vec<&str> = list.split([',', ' ']).filter(|v| !v.is_empty()).collect();
        let has = |want: &str| values.iter().any(|v| v.eq_ignore_ascii_case(want));
        match word {
            w if w.eq_ignore_ascii_case("Family") => path.push_str("Classic"),
            w if w.eq_ignore_ascii_case("Game") => path.push_str("Vanilla"),
            w if w.eq_ignore_ascii_case("TextLocale") => path.push_str(locale),
            w if w.eq_ignore_ascii_case("AllowLoadGameType") && has("vanilla") => {}
            w if w.eq_ignore_ascii_case("AllowLoadTextLocale") && has(locale) => {}
            w if w.eq_ignore_ascii_case("AllowLoad") && has("game") => {}
            _ => return None,
        }
        rest = &rest[close + 1..];
    }
    path.push_str(rest);
    let path = path.trim().to_string();
    (!path.is_empty() && !benilla_pfui::skipped_file(addon, &path)).then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_lines_keep_what_passes_and_expand_the_variables() {
        let line = |l| file_line("A", l, "deDE");
        assert_eq!(line("Plain.lua").as_deref(), Some("Plain.lua"));
        assert_eq!(
            line("Vanilla.lua   [AllowLoadGameType vanilla]").as_deref(),
            Some("Vanilla.lua")
        );
        assert_eq!(line("Mainline.lua [AllowLoadGameType mainline]"), None);
        assert_eq!(
            line("Loc\\deDE.lua [AllowLoadTextLocale deDE, frFR]").as_deref(),
            Some("Loc\\deDE.lua")
        );
        assert_eq!(line("Loc\\enUS.lua [AllowLoadTextLocale enUS]"), None);
        assert_eq!(line("Glue.lua [AllowLoad glue]"), None);
        assert_eq!(line("Odd.lua [SomethingElse]"), None);
        assert_eq!(
            line("[Family]\\[Game]\\Init.lua").as_deref(),
            Some("Classic\\Vanilla\\Init.lua")
        );
        assert_eq!(
            line("Localization\\[TextLocale].lua").as_deref(),
            Some("Localization\\deDE.lua")
        );
        assert_eq!(
            file_line("!!!ClassicAPI", "Util\\TradeSkillLink.lua", "enUS").as_deref(),
            Some("Util\\TradeSkillLink.lua")
        );
    }

    #[test]
    fn saved_variables_first_reads_its_directive() {
        assert!(saved_variables_first(&Toc::parse(
            "## SavedVariables: DB\n## LoadSavedVariablesFirst: 1\n"
        )));
        assert!(!saved_variables_first(&Toc::parse(
            "## SavedVariables: DB\n"
        )));
    }

    #[test]
    fn an_interface_list_passes_with_this_clients_version_anywhere() {
        let toc = Toc::parse("## Interface: 120100, 50504, 11200\n");
        assert_eq!(interface(&toc), 11200);
        let old = Toc::parse("## Interface: 20504, 11100\n");
        assert_eq!(interface(&old), 20504);
    }
}
