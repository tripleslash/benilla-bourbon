//! Seed icon choosers before addon OnLoads; providers snapshot their catalog at construction.

use std::sync::OnceLock;

pub(super) fn prepare(script: &benilla_ui::script::UiScript) -> Result<(), String> {
    static CATALOG: OnceLock<Result<benilla_pfui::icons::Catalog, String>> = OnceLock::new();
    let catalog = CATALOG.get_or_init(|| {
        let data = benilla_formats::wow_data().ok_or("no data for pfUI icon catalog")?;
        let chain = benilla_formats::open_chain(&data).map_err(|e| e.to_string())?;
        let paths = chain.list().map_err(|e| e.to_string())?;
        let catalog = benilla_pfui::icons::Catalog::from_paths(paths.into_iter().map(|p| p.name));
        #[cfg(not(target_arch = "wasm32"))]
        let catalog = {
            let entries = data
                .parent()
                .and_then(|root| std::fs::read_dir(root.join("Interface/Icons")).ok());
            catalog.with_loose_paths(entries.into_iter().flatten().flatten().filter_map(|entry| {
                entry.file_type().ok().filter(|kind| kind.is_file())?;
                Some(format!(
                    "Interface/Icons/{}",
                    entry.file_name().to_string_lossy()
                ))
            }))
        };
        Ok(catalog)
    });
    catalog.as_ref().map_err(Clone::clone)?.feed(script.lua());
    Ok(())
}
