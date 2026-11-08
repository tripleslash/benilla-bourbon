# benilla's `bevy_pbr` fork — what differs from upstream, and how to check

Upstream: `bevy_pbr 0.19.1` (crates.io), MIT OR Apache-2.0. Wired in through `[patch.crates-io]`
in the workspace root. Unpacked from the registry's `.crate` (sha256
`244ae7d618b51a59c913c36b0564a295cd85ea91a5e777b542bb7bdb846a23d6`, the `Cargo.lock` checksum),
with cargo's packaging metadata (`.cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`) dropped
as the other `third_party` forks drop it.

## The first delta: occlusion culling never culls a `NoFrustumCulling` mesh

`src/render/mesh_preprocess.wgsl`, the occlusion test of the GPU preprocessing pass
(`OCCLUSION_CULLING`, both phases).

Upstream skips the *frustum* test for a mesh flagged `MESH_FLAGS_NO_FRUSTUM_CULLING_BIT` and then
runs the *occlusion* test on it anyway, on its `MeshCullingData` box. That box is one of two things
for such a mesh, and neither bounds what it draws:

- **No `Aabb`** (`calculate_bounds` leaves a `NoFrustumCulling` entity alone): `MeshCullingData::new`
  stands in a box of infinite half-extents at the origin. Its corners go through the world matrix,
  where `0 · ∞` is NaN; the screen box and the view depth are then NaN or ±∞, and with IEEE
  `max`/`min` (which return the non-NaN operand) the depth can come out as the far plane. Such a
  mesh is culled behind anything at all.
- **An `Aabb` it does not keep to**: benilla's skinned dress keeps its idle or bind box for the
  picker (`benilla-app` `entities/attach/dress.rs`), and its drawn pose (a mount, an emote, a
  raised weapon) leaves it; particles and the player-shadow proxy draw wherever their vertices are.

`NoFrustumCulling` already says "do not cull this on its bounds", which is exactly what the
occlusion test needs to hear, so the hunk makes the test say so: an exempt mesh is never culled in
either phase (never deferred to the late phase, never dropped there). Nothing else in the shader
changes; every other mesh is tested as upstream tests it.

benilla's world camera turns this pass on with `$WOW_ENTITY_GPU_CULL=1` (`benilla-world`
`entity_cull.rs`, off by default); no other camera has `OcclusionCulling`.

## The second delta: no prepass depth before the colour pipeline is ready

`src/prepass/mod.rs`, `queue_prepass_material_meshes`. A mesh's prepass pipeline and its colour
pipeline compile asynchronously and separately, and a phase item whose pipeline is not ready is
skipped. Upstream queues the prepass item as soon as the prepass pipeline is specialized, so when
it finishes first the mesh lays its depth with nothing drawn on it: what is behind it fails the
depth test and its silhouette shows the clear colour until the colour pipeline lands (seen as new
M2 doodads flashing blue-grey for about a second). The hunk holds the entity in the queue's own
pending list (the one upstream uses for a mesh whose assets have not arrived) until its colour
pipeline in that view is ready, so prepass depth never exists without the colour draw on it. A
view with no prepass returns before the hunk's one lookup.

## The third delta: shadow passes open a diagnostics span

`src/render/light.rs`, `view_shadow_pass`. Upstream opens only a `trace` info span there, so bevy's
shadow cascades, point faces and spot shadows (the `DirectionalLight` `benilla-app`'s
`player_shadow.rs` draws) show in neither `WOW_GPU_PASSES` nor `WOW_GPU_STATS`. The hunk opens a
`pass_span` named after the shadow view (`shadow_directional_light_{L}_cascade_{C}`,
`shadow_point_light_{L}_{face}`, `shadow_spot_light_{L}`), with a `_late` twin for the late
occlusion-culling pass, as `src/ssr/mod.rs` does for its own pass. Only a pass span opens a
pipeline-statistics query and those cannot nest; nothing above `view_shadow_pass` holds one.

## Verifying against upstream

```sh
mkdir -p /tmp/upstream && curl -sL https://static.crates.io/crates/bevy_pbr/bevy_pbr-0.19.1.crate | tar xz -C /tmp/upstream
diff -r third_party/bevy_pbr /tmp/upstream/bevy_pbr-0.19.1
```

That must print exactly a `diff` for `src/render/mesh_preprocess.wgsl` (the `occlusion_exempt`
binding and the condition it joins, five lines in, one changed), one for `src/prepass/mod.rs`
(two parameters, the view's colour cache lookup and the readiness check), one for
`src/render/light.rs` (the `RecordDiagnostics` import and the span around the shadow phase),
`Only in third_party/bevy_pbr:
BENILLA.md`, and `Only in <upstream>: .cargo_vcs_info.json`, `Cargo.lock`, `Cargo.toml.orig`.
