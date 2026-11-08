use std::{
    env,
    path::{Path, PathBuf},
};

/// BENILLA: the NGX SDK is not redistributable and not in the tree. Without `DLSS_SDK` (or
/// without `VULKAN_SDK`, whose headers bindgen needs) the crate builds as its mock: the same API,
/// every constructor answering `FeatureNotSupported`, so a clone and CI compile it and the client
/// falls back to FSR. `DLSS_SDK` may name the NVIDIA/DLSS repository (`lib/<platform>/x64/`) or
/// Streamline's `external/ngx-sdk` (`lib/<platform>/`).
fn main() {
    println!("cargo:rustc-check-cfg=cfg(dlss_mock)");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if cfg!(feature = "mock") {
        println!("cargo:rustc-cfg=dlss_mock");
        return;
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if !matches!(target_os.as_str(), "windows" | "linux") {
        println!("cargo:rustc-cfg=dlss_mock");
        return;
    }
    let (Ok(dlss_sdk), Ok(vulkan_sdk)) = (env::var("DLSS_SDK"), env::var("VULKAN_SDK")) else {
        println!(
            "cargo:warning=dlss_wgpu: DLSS_SDK or VULKAN_SDK unset, building the mock (no DLSS)"
        );
        println!("cargo:rustc-cfg=dlss_mock");
        return;
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let platform = if target_os == "windows" {
        format!("Windows_{target_arch}")
    } else {
        format!("Linux_{target_arch}")
    };
    println!("cargo:rustc-env=DLSS_SDK_PLATFORM={platform}");

    // BENILLA: both SDK layouts.
    let base = format!("{dlss_sdk}/lib/{platform}");
    let lib_dir = [format!("{base}/x64"), base.clone()]
        .into_iter()
        .find(|dir| {
            let lib = if target_os == "windows" {
                "nvsdk_ngx_d.lib"
            } else {
                "libnvsdk_ngx.a"
            };
            Path::new(dir).join(lib).is_file()
        })
        .unwrap_or_else(|| panic!("DLSS SDK at {dlss_sdk} has no NGX library for {platform}"));
    println!("cargo:rustc-link-search=native={lib_dir}");
    if target_os == "windows" {
        // Streamline's copy ships only the dynamic-CRT library, which is what Rust links.
        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
    } else {
        println!("cargo:rustc-link-lib=static=nvsdk_ngx");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }

    let vulkan_sdk_include = if target_os == "windows" {
        "Include"
    } else {
        "include"
    };
    bindgen::Builder::default()
        .header(format!("{}/src/wrapper.h", env!("CARGO_MANIFEST_DIR")))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_dir.join("wrap_static_fns"))
        .clang_arg(format!("-I{dlss_sdk}/include"))
        .clang_arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .allowlist_item(".*NGX.*")
        .blocklist_item("Vk.*")
        .blocklist_item("PFN_vk.*")
        .blocklist_item(".*Cuda.*")
        .blocklist_item(".*CUDA.*")
        // The D3D12 entry points are declared by hand in `d3d12.rs`, on opaque COM pointers.
        .blocklist_item(".*D3[Dd].*")
        .generate()
        .unwrap()
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap();

    cc::Build::new()
        .file(out_dir.join("wrap_static_fns.c"))
        .includes([
            format!("{dlss_sdk}/include"),
            format!("{vulkan_sdk}/{vulkan_sdk_include}"),
        ])
        .compile("wrap_static_fns");
}
