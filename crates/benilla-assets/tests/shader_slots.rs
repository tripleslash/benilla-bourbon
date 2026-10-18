//! Every WGSL file in the workspace gives each interface location and each bind slot to one
//! variable at a time. A struct's two `@location(n)` fields, or a file's two `@group(g)
//! @binding(n)` vars, may share `n` only in exclusive arms of one `#ifdef`/`#else`; anywhere else
//! naga and WebGPU refuse the pipeline, and every mesh drawn with it vanishes. No test compiles a
//! pipeline, so the text is checked here.

use std::path::{Path, PathBuf};

/// The shader directories of the crates that ship WGSL.
const SHADER_DIRS: [&str; 3] = [
    "../benilla-assets/src/shaders",
    "../benilla-world/src/shaders",
    "../benilla-app/src/shaders",
];

fn wgsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            wgsl_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "wgsl") {
            out.push(p);
        }
    }
}

/// One `#ifdef` group and the arm a line sits in, innermost last.
type Arms = Vec<(usize, usize)>;

/// Whether two lines sit in different arms of one group, so at most one is ever compiled.
fn exclusive(a: &Arms, b: &Arms) -> bool {
    a.iter()
        .any(|(g, arm)| b.iter().any(|(h, other)| g == h && arm != other))
}

/// Each line with the preprocessor arms it sits in.
fn with_arms(text: &str) -> Vec<(&str, Arms)> {
    let mut stack: Arms = Vec::new();
    let mut groups = 0usize;
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#ifdef") || t.starts_with("#ifndef") || t.starts_with("#if ") {
            groups += 1;
            stack.push((groups, 0));
        } else if t.starts_with("#else") {
            if let Some(top) = stack.last_mut() {
                top.1 += 1;
            }
        } else if t.starts_with("#endif") {
            stack.pop();
        } else {
            out.push((line, stack.clone()));
        }
    }
    out
}

/// The number inside `tag(` … `)` on `line`, and the text after it.
fn tagged<'a>(line: &'a str, tag: &str) -> Option<(String, &'a str)> {
    let at = line.find(tag)? + tag.len();
    let end = line[at..].find(')')? + at;
    Some((line[at..end].trim().to_string(), &line[end + 1..]))
}

fn clashes(file: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(file).unwrap();
    let lines = with_arms(&text);
    let mut out = Vec::new();
    let mut seen = |kind: &str, key: String, arms: &Arms, list: &mut Vec<(String, Arms)>| {
        for (k, a) in list.iter() {
            if *k == key && !exclusive(a, arms) {
                out.push(format!("{}: {kind} {key} given twice", file.display()));
            }
        }
        list.push((key, arms.clone()));
    };

    // Struct fields, per struct.
    let mut locations: Vec<(String, Arms)> = Vec::new();
    let mut in_struct: Option<String> = None;
    for (line, arms) in &lines {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("struct ") {
            in_struct = rest.split_whitespace().next().map(str::to_string);
            locations.clear();
        } else if line.starts_with('}') {
            in_struct = None;
        } else if let Some(name) = &in_struct {
            if let Some((n, _)) = tagged(t, "@location(") {
                seen(&format!("{name} location"), n, arms, &mut locations);
            }
        }
    }

    // Module-scope bind slots, per file.
    let mut slots: Vec<(String, Arms)> = Vec::new();
    for (line, arms) in &lines {
        let t = line.trim_start();
        if !t.starts_with("@group(") {
            continue;
        }
        if let (Some((g, _)), Some((b, _))) = (tagged(t, "@group("), tagged(t, "@binding(")) {
            seen("binding", format!("({g}, {b})"), arms, &mut slots);
        }
    }
    out
}

#[test]
fn no_two_shader_variables_share_a_location_or_a_binding() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in SHADER_DIRS {
        wgsl_files(&root.join(dir), &mut files);
    }
    assert!(
        files.iter().any(|f| f.ends_with("wow_model.wgsl")),
        "the model shader is among the files read"
    );
    let found: Vec<String> = files.iter().flat_map(|f| clashes(f)).collect();
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// The rule bites: the fork's `point_sel` and upstream's `env_uv` both at location 11, the merge
/// that left every model undrawn.
#[test]
fn a_shared_location_is_found_and_exclusive_arms_are_not() {
    let dir = std::env::temp_dir().join("benilla_shader_slots_test");
    std::fs::create_dir_all(&dir).unwrap();
    let clash = dir.join("clash.wgsl");
    std::fs::write(
        &clash,
        "struct Out {\n    @location(11) @interpolate(flat) point_sel: vec3<u32>,\n    \
         @location(11) env_uv: vec2<f32>,\n}\n",
    )
    .unwrap();
    assert_eq!(clashes(&clash).len(), 1);
    let arms = dir.join("arms.wgsl");
    std::fs::write(
        &arms,
        "#ifdef MSAA\n@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;\n#else\n\
         @group(0) @binding(1) var depth_tex: texture_depth_2d;\n#endif\n",
    )
    .unwrap();
    assert!(clashes(&arms).is_empty());
}
