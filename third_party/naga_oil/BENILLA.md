# benilla's `naga_oil` fork — what differs from upstream, and how to check

Upstream: `naga_oil 0.22.0` (crates.io), MIT OR Apache-2.0 — the version `bevy_shader 0.19.1`
resolves to. Wired in through `[patch.crates-io]` in the workspace root. Copied from the registry
source as published (minus `.cargo_vcs_info.json`, `.cargo-ok`, `.github`, `.gitignore`,
`Cargo.lock` and `Cargo.toml.orig`).

## Why a fork exists at all

**A composed mesh or task shader names the wrong items.** When the composer builds a module, it
adds every imported module's items to the derived module first and the shader's own items after
them, so a handle into the shader's own arenas is not the same handle in the composed module. The
entry point's function is remapped (`localize_function`), but its `mesh_info` (the output
variable, the vertex and primitive output types, the override expressions), `task_payload` and
`incoming_ray_payload` were copied verbatim. A mesh entry point in a shader with any `#import`
then names some other global as its output (or one past the arena), and the validator or the
SPIR-V writer fails on it. static_gx's mesh-shader lane (`static_gx/render/meshlets.rs`) is such a
shader: it imports the bindless, lighting and wind libraries.

## The delta

| file | hunk | why |
|---|---|---|
| `src/derive.rs` | `DerivedModule::localize_mesh_info` and `localize_payload` (new): `import_type` / `import_global` / `import_global_expression` over each handle | the remapping upstream skipped |
| `src/derive.rs` | `into_module_with_entrypoints` builds `mesh_info`, `task_payload`, `incoming_ray_payload` through them | composable modules and `redirect` |
| `src/compose/mod.rs` | `make_naga_module`'s entry-point loop does the same | the top-level shader, where the lane's entry points live |

Nothing changes for a shader without mesh, task or ray-payload entry points: the new calls return
`None` for `None`.

## How to check after a bump

Diff the registry source against this tree; only the hunks above should differ:

```bash
diff -r ~/.cargo/registry/src/*/naga_oil-0.22.0/src third_party/naga_oil/src
```

If a later `naga_oil` remaps these fields itself, drop the fork and its patch entry.
