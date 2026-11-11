//! Source audits must cover the same optional modules as the tested client.

/// The pfUI files only the `nampower` feature compiles.
const NAMPOWER: &[&str] = &[
    "pfui/cast_channel_tests.rs",
    "pfui/cast_reports.rs",
    "pfui/cooldowns.rs",
    "pfui/focus_session_tests.rs",
    "pfui/nampower.rs",
    "pfui/script_queue.rs",
    "pfui/spell_queue.rs",
    "pfui/spell_queue_deferred_tests.rs",
    "pfui/spell_queue_special_tests.rs",
    "pfui/spell_queue_tests.rs",
    "pfui/spell_retry.rs",
    "pfui/spell_retry_tests.rs",
];

pub(super) fn is_compiled(path: &std::path::Path) -> bool {
    let rel = super::rel_path(path);
    if NAMPOWER.contains(&rel.as_str()) {
        return cfg!(feature = "nampower");
    }
    let name = rel.rsplit('/').next().unwrap_or_default();
    cfg!(feature = "pfui")
        || !(rel.starts_with("pfui/") || matches!(name, "pfui.rs" | "pfui_tests.rs"))
}

#[test]
fn optional_sources_follow_the_build_feature() {
    for path in [
        "pfui.rs",
        "net/pfui.rs",
        "ui_social/pfui.rs",
        "ui_aura/pfui_tests.rs",
    ] {
        assert_eq!(
            is_compiled(&super::src_dir().join(path)),
            cfg!(feature = "pfui")
        );
    }
    for path in ["pfui/nampower.rs", "pfui/spell_queue_tests.rs"] {
        assert_eq!(
            is_compiled(&super::src_dir().join(path)),
            cfg!(feature = "nampower")
        );
    }
    assert!(is_compiled(&super::src_dir().join("spell/cooldowns.rs")));
}
