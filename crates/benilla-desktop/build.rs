//! Stamps the commit this binary was built from, and gives the main thread the stack the app needs
//! on Windows — the `benilla` launcher's build script.

fn main() {
    benilla_buildstamp::emit();
}
