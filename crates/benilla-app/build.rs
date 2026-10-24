//! Links this package's binaries, tests and examples with MSVC's 8 KiB PDB pages.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // A PDB of the default 4 KiB pages holds at most 4 GiB, and link.exe updates an existing PDB
    // in place, so a debug example relinked over days outgrows it (LNK1201: `addon_harness` at 4.5
    // GiB, 1.3 fresh). 8 KiB pages hold 8. Every target this package links carries the whole app.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg=/PDBPAGESIZE:8192");
    }
}
