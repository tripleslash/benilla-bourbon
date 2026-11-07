//! Stamps the commit this binary was built from, gives the main thread the stack the app needs on
//! Windows (the `benilla` launcher's build script), and on a Windows target embeds the client's
//! icon: `web/favicon.ico`, WoW.exe's own icon group, which the web build serves as its favicon.

fn main() {
    benilla_buildstamp::emit();
    let icon = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/favicon.ico");
    println!("cargo:rerun-if-changed={icon}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(icon);
        if let Err(e) = res.compile() {
            panic!("embedding the icon {icon}: {e}");
        }
    }
}
