# benilla's `naga` fork — what differs from upstream, and how to check

Upstream: [`gfx-rs/wgpu`](https://github.com/gfx-rs/wgpu) (`naga`), MIT OR Apache-2.0, version
`30.0.1`, the version the workspace lock resolves to (Bevy 0.20.1's wgpu 30). Wired in through
`[patch.crates-io]` in the workspace root. The crate is copied from the cargo registry as published
(`src/`, `build.rs`, `Cargo.toml`, `README.md`, the two licence files). Every hunk is marked
`BENILLA:`. Its partners are `third_party/wgpu-core` and `third_party/wgpu-hal` (the meshlet lane
on DX12, "The draw index" in each).

## Why a fork exists at all

naga 30's HLSL backend has no `@builtin(draw_index)`: `BuiltIn::DrawIndex` maps to "Unsupported
builtin" (`back/hlsl/conv.rs`) and `supported_capabilities` lists no `DRAW_INDEX`
(`back/hlsl/mod.rs`, "No DRAW_INDEX"). static_gx's meshlet lane reads it in its task stage
(`static_gx_mesh.wesl`, `meshlet_task`: which draw of the multi-draw this workgroup belongs to), so
the lane could not run on DX12.

HLSL has no semantic for it before Shader Model 6.8, but the backend already carries the
mechanism it needs: the three-word special constants (`_NagaConstants`: `first_vertex`,
`first_instance`, `other`) that wgpu-hal sets before each direct draw and dispatch, and that
wgpu-core's indirect rewrite writes in front of each indirect draw's arguments on DX12
(gfx-rs/wgpu#7974). In a draw the third word, `other`, was unused (always 0).

## The patch

| file | hunk |
|---|---|
| `src/back/hlsl/writer.rs`, `is_draw_index_binding`, `arg_has_draw_index` | new: whether a binding is `draw_index`, and whether an entry point argument is or holds one. |
| `src/back/hlsl/writer.rs`, `write_ep_interface` | an entry point with a `draw_index` argument (or struct member) takes its inputs as one struct, as one with a subgroup built-in does. |
| `src/back/hlsl/writer.rs`, `write_interface_struct` | `draw_index` is left out of that struct, as the subgroup built-ins are: it has no semantic. |
| `src/back/hlsl/writer.rs`, `write_ep_argument_initialization` | `draw_index` is initialized as `_NagaConstants.other`; without a special constants binding the writer returns an error. |
| `src/back/hlsl/writer.rs`, `write_ep_arguments_initialization` | the input struct goes back into `entry_point_io` after the body's initialization, where upstream leaves it taken: a task or mesh entry point's outer wrapper, written after the body, takes it as its input too, and left taken it lists `draw_index` as a plain argument with no semantic. |
| `src/back/hlsl/mesh_shader.rs`, `write_nested_function_outer` | the separator after that struct, which upstream never needed (the group index follows it). |
| `src/back/hlsl/writer.rs`, `NestedEntryPointArgs`, `write_function`; `src/back/hlsl/mesh_shader.rs`, `write_call_args`, `write_nested_function_outer` | a task or mesh entry point whose input struct holds `local_invocation_index` reads it from the struct (`input.lane`) instead of taking a second `SV_GroupIndex` argument, which DXC rejects as an overlapping semantic; a group index the inner function does take as an argument of its own is declared and passed by the outer wrapper in a task stage too, where upstream declared it only for a mesh stage or workgroup zeroing and the task wrapper passed an undeclared `local_invocation_index`. Both failed the meshlet lane's task stage at pipeline creation on DX12. |
| `src/back/hlsl/mod.rs`, `supported_capabilities` | lists `Caps::DRAW_INDEX`. |

**What a shader reads.** In vertex, task and mesh stages, `draw_index` is `_NagaConstants.other`:

- a direct draw: 0 (wgpu-hal's `prepare_draw` writes `[first_vertex, first_instance, 0]`);
- an indirect draw, a multi-draw and a multi-draw-indirect-count on DX12: the draw's index in its
  call, which wgpu-core's rewrite writes as the third word in front of each draw's arguments;
- a direct `draw_mesh_tasks`: the dispatch's `z`, because wgpu-hal sets the special constants to
  the dispatch size there (`prepare_dispatch`), for a task stage's `num_workgroups`. A task stage
  that reads both `draw_index` and `num_workgroups` reads one word for two things; the meshlet
  lane draws only through counted multi-draws and reads no `num_workgroups`.

Compute's own use of the constants (`num_workgroups`, all three words) is unchanged.

## How to check

```sh
diff -ru ~/.cargo/registry/src/index.crates.io-*/naga-30.0.1 third_party/naga
```

shows exactly the hunks above, this file, and `Only in` the registry copy for cargo's packaging
files and the repository's (`.cargo/`, `.cargo-ok`, `.cargo_vcs_info.json`, `.gitattributes`,
`.gitignore`, `Cargo.lock`, `Cargo.toml.orig`, `CHANGELOG.md`). If
the registry copy is absent, fetch `https://static.crates.io/crates/naga/naga-30.0.1.crate`.

`benilla-app`'s `shaders.rs` tests write the meshlet lane's module as HLSL and check that its task
stage reads `_NagaConstants.other`, and that a compute stage's `num_workgroups` still reads all
three words.

## When to drop it

When naga's HLSL backend supports `draw_index` (to be offered upstream as a gfx-rs/wgpu issue or
PR with this mechanism), or when the workspace leaves wgpu 30: delete this directory and the `naga`
line under `[patch.crates-io]`, and check `third_party/wgpu-core` and `third_party/wgpu-hal` ("The
draw index") against the release.
