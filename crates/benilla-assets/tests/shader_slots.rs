//! Every WGSL and WESL file in the workspace gives each interface location and each bind slot to
//! one variable at a time. A struct's two `@location(n)` fields, or a file's two `@group(g)
//! @binding(n)` vars, may share `n` only in exclusive arms of one condition: an `@if(X) { }` block
//! and its `@else`, or `@if(X)` and `@if(!X)`, on blocks or on single items; anywhere else naga
//! and WebGPU refuse the pipeline, and every mesh drawn with it vanishes. No test compiles every
//! pipeline, so the text is checked here.

use std::path::{Path, PathBuf};

/// The shader directories of the crates that ship shaders.
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
        } else if p.extension().is_some_and(|x| x == "wgsl" || x == "wesl") {
            out.push(p);
        }
    }
}

/// One condition and the arm a line sits in, innermost last. A condition on one flag, plain or
/// negated, is keyed by the flag, so `@if(X)` and `@if(!X)` are the two arms of one key wherever
/// they stand; any other condition is a key of its own, whose `@else` is its other arm.
type Arms = Vec<(String, usize)>;

/// Whether two lines sit in different arms of one condition, so at most one is ever compiled.
fn exclusive(a: &Arms, b: &Arms) -> bool {
    a.iter()
        .any(|(g, arm)| b.iter().any(|(h, other)| g == h && arm != other))
}

/// The condition of a leading `@if(…)` / `@elif(…)` attribute and the text after it.
fn condition<'a>(t: &'a str, attribute: &str) -> Option<(&'a str, &'a str)> {
    let rest = t.strip_prefix(attribute)?.trim_start().strip_prefix('(')?;
    let mut open = 1usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => open += 1,
            ')' => {
                open -= 1;
                if open == 0 {
                    return Some((rest[..i].trim(), rest[i + 1..].trim_start()));
                }
            }
            _ => {}
        }
    }
    None
}

/// The arm a condition puts its item in; `unique` numbers the keys of compound conditions.
fn arm_of(condition: &str, unique: &mut usize) -> (String, usize) {
    let flag = condition.strip_prefix('!').unwrap_or(condition).trim();
    if flag.chars().all(|c| c.is_alphanumeric() || c == '_') {
        (flag.to_string(), usize::from(condition.starts_with('!')))
    } else {
        *unique += 1;
        (format!("#{unique}"), 0)
    }
}

/// Each line with the conditional arms it sits in, and the line with a leading per-item `@if`
/// removed. wesl flattens an `@if(…) {` block into its parent, so its braces are no scope.
fn with_arms(text: &str) -> Vec<(&str, Arms)> {
    // Per open brace: the arm of a conditional block, or `None` for a scope of the code.
    let mut braces: Vec<Option<(String, usize)>> = Vec::new();
    let mut unique = 0usize;
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        let code = t.split("//").next().unwrap_or("").trim_end();
        if let Some((cond, "{")) = condition(code, "@if") {
            braces.push(Some(arm_of(cond, &mut unique)));
            continue;
        }
        if let Some(rest) = code.strip_prefix('}').map(str::trim_start) {
            let next = if rest == "@else {" {
                Some(())
            } else {
                condition(rest, "@elif").and_then(|(_, after)| (after == "{").then_some(()))
            };
            if next.is_some() {
                if let Some(Some((_, arm))) = braces.last_mut() {
                    *arm += 1;
                }
                continue;
            }
        }
        let mut arms: Arms = braces.iter().flatten().cloned().collect();
        let item = match condition(t, "@if") {
            Some((cond, rest)) => {
                arms.push(arm_of(cond, &mut unique));
                rest
            }
            None => t,
        };
        for c in code.chars() {
            match c {
                '{' => braces.push(None),
                '}' => {
                    braces.pop();
                }
                _ => {}
            }
        }
        out.push((item, arms));
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
    for (t, arms) in &lines {
        if let Some(rest) = t.strip_prefix("struct ") {
            in_struct = rest.split_whitespace().next().map(str::to_string);
            locations.clear();
        } else if t.starts_with('}') {
            in_struct = None;
        } else if let Some(name) = &in_struct {
            if let Some((n, _)) = tagged(t, "@location(") {
                seen(&format!("{name} location"), n, arms, &mut locations);
            }
        }
    }

    // Module-scope bind slots, per file.
    let mut slots: Vec<(String, Arms)> = Vec::new();
    for (t, arms) in &lines {
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
        files.iter().any(|f| f.ends_with("wow_model.wesl")),
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
    let clash = dir.join("clash.wesl");
    std::fs::write(
        &clash,
        "struct Out {\n    @location(11) @interpolate(flat) point_sel: vec3<u32>,\n    \
         @location(11) env_uv: vec2<f32>,\n}\n",
    )
    .unwrap();
    assert_eq!(clashes(&clash).len(), 1);
    let arms = dir.join("arms.wesl");
    std::fs::write(
        &arms,
        "@if(MSAA) {\n@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;\n\
         } @else {\n@group(0) @binding(1) var depth_tex: texture_depth_2d;\n}\n\
         struct Out {\n    @if(A) @location(3) a: f32,\n    @if(!A) @location(3) b: f32,\n}\n\
         @if(!B) {\n@group(1) @binding(0) var<uniform> c: f32;\n}\n\
         @if(B) @group(1) @binding(0) var<uniform> d: f32;\n",
    )
    .unwrap();
    assert!(clashes(&arms).is_empty(), "{:?}", clashes(&arms));
    let unrelated = dir.join("unrelated.wesl");
    std::fs::write(
        &unrelated,
        "@if(A) {\n@group(0) @binding(1) var<uniform> a: f32;\n}\n\
         @if(B) @group(0) @binding(1) var<uniform> b: f32;\n\
         fn f() {\n    if (true) {\n    }\n}\n@group(0) @binding(2) var<uniform> c: f32;\n\
         @if(A && B) @group(0) @binding(2) var<uniform> d: f32;\n",
    )
    .unwrap();
    assert_eq!(clashes(&unrelated).len(), 2, "{:?}", clashes(&unrelated));
}
