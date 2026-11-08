use std::{
    env,
    path::{Path, PathBuf},
};

/// BENILLA: the NGX SDK is vendored beside this crate (`third_party/nvidia-dlss`, DLSS SDK
/// 310.9.16) and the Vulkan headers bindgen reads too (`third_party/vulkan-headers`, 1.3.281, ash
/// 0.38's), so a clone builds DLSS with nothing installed but libclang. `$DLSS_SDK` and
/// `$VULKAN_SDK` override them. The crate builds as its mock (the same API, every constructor
/// answering `FeatureNotSupported`) where NGX has no library for the target or libclang is missing;
/// the warning says which.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(dlss_mock)");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    let mock = |why: &str| {
        println!("cargo:warning=dlss_wgpu: {why}; building the mock (no DLSS)");
        println!("cargo:rustc-cfg=dlss_mock");
    };
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
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dlss_sdk = env::var("DLSS_SDK").map_or_else(
        |_| manifest.join("../nvidia-dlss"),
        PathBuf::from,
    );
    let vulkan_include = env::var("VULKAN_SDK").map_or_else(
        |_| manifest.join("../vulkan-headers/include"),
        |sdk| {
            let upper = Path::new(&sdk).join("Include");
            if upper.is_dir() {
                upper
            } else {
                Path::new(&sdk).join("include")
            }
        },
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let platform = if target_os == "windows" {
        format!("Windows_{target_arch}")
    } else {
        format!("Linux_{target_arch}")
    };
    println!("cargo:rustc-env=DLSS_SDK_PLATFORM={platform}");

    // BENILLA: the C runtime the target links: the desktop build is `+crt-static`, which needs
    // NGX's static-CRT library (`_s`); the dynamic one (`_d`) otherwise. Both SDK layouts: the
    // DLSS repository's `lib/<platform>/x64/` and Streamline's `lib/<platform>/`.
    let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
        .is_ok_and(|f| f.split(',').any(|f| f == "crt-static"));
    let (lib_file, lib_name) = match (target_os.as_str(), crt_static) {
        ("windows", true) => ("nvsdk_ngx_s.lib", "nvsdk_ngx_s"),
        ("windows", false) => ("nvsdk_ngx_d.lib", "nvsdk_ngx_d"),
        _ => ("libnvsdk_ngx.a", "nvsdk_ngx"),
    };
    let base = dlss_sdk.join("lib").join(&platform);
    let Some(lib_dir) = [base.join("x64"), base]
        .into_iter()
        .find(|dir| dir.join(lib_file).is_file())
    else {
        mock(&format!(
            "no {lib_file} for {platform} in {}",
            dlss_sdk.display()
        ));
        return;
    };
    let include = dlss_sdk.join("include");

    // libclang is loaded at run time by bindgen and panics when absent: caught, it is the mock.
    let bindings = std::panic::catch_unwind(|| {
        bindgen::Builder::default()
            .header(manifest.join("src/wrapper.h").to_string_lossy())
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .wrap_static_fns(true)
            .wrap_static_fns_path(out_dir.join("wrap_static_fns"))
            .clang_arg(format!("-I{}", include.display()))
            .clang_arg(format!("-I{}", vulkan_include.display()))
            .allowlist_item(".*NGX.*")
            .blocklist_item("Vk.*")
            .blocklist_item("PFN_vk.*")
            .blocklist_item(".*Cuda.*")
            .blocklist_item(".*CUDA.*")
            // The D3D12 entry points are declared by hand in `d3d12.rs`, on opaque COM pointers.
            .blocklist_item(".*D3[Dd].*")
            .generate()
    });
    let bindings = match bindings {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => {
            mock(&format!("bindgen failed ({e})"));
            return;
        }
        Err(_) => {
            mock("libclang not found (install LLVM, or set LIBCLANG_PATH)");
            return;
        }
    };
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }

    // Generate and link a library for the headers' static inline functions.
    cc::Build::new()
        .file(out_dir.join("wrap_static_fns.c"))
        .includes([include, vulkan_include])
        .compile("wrap_static_fns");
}
