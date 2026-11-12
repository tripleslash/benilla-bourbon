//! fork: **the sun cascades' entity casters, several cascades in one multiview pass** (native).
//!
//! Every frame the cascades whose entity pass runs ([`holds_entities`]: the near two, and one of
//! the far two in turn) each record a pass over the same entity parts: a body beside the camera is
//! in c0's, c1's and c2's lists, three draws of one mesh. Where the device has multiview
//! ([`HwCaps::multiview_views`]) and `WOW_SHADOW_MULTIVIEW` is not `0`, one more pass,
//! `modern_shadow_dynamic_mv`, draws every part that two or more of those cascades take ONCE,
//! into all of their layers; each layer's own pass keeps only the parts no other cascade of the
//! pass takes. One draw per shared part instead of one per cascade.
//!
//! **The pass.** Its attachment is the working array's layers `0..n` as one `D2Array` view, its
//! mask the cascades drawn this frame ([`pick`]): `0b0111` (c0, c1, c2) on the frames c3 is held,
//! `0b1011` (c0, c1, c3) on the others where masks may skip a layer
//! ([`HwCaps::multiview_selective`]: Vulkan, DX12), else `0b0011`. The view index is the array
//! layer on Vulkan and DX12 (wgpu-hal DX12 lists a view-instance location for every layer up to
//! the mask's highest bit), so `entity_shadow.wesl`'s `MULTIVIEW` variant reads cascade `i`'s
//! matrix at row `i` of [`MultiviewViews`].
//!
//! **Per-view culling is kept.** Each part's draw carries its own view mask in its one instance
//! (`first_instance`, under its caster-table slot, [`instance_of`]): the layers of the pass whose
//! entity cull ([`entity_cull_for`]) takes it.
//! The vertex stage moves the part outside the clip volume in any other view, so every layer
//! receives exactly the triangles its own pass would have drawn, and the dirty tiles each layer
//! records for the next frame's restore are unchanged (the layer's own loop still marks them).
//!
//! **What it costs.** A multiview draw runs the vertex stage once per view of the pass on most
//! hardware, including the views a part is culled out of. A part two of three cascades take costs
//! 1.5x the vertex work of its two per-layer draws, and one all three take costs the same; a part
//! only one cascade takes stays in that layer's pass ([`plan`]'s two-view floor). So the lane
//! trades at most half again the vertex work of the shared parts for a third to a half of the
//! entity caster draws. `WOW_SHADOW_MULTIVIEW=0` is the A/B.
//!
//! **Not on the lane.** The static passes (cached; with the bindless multi-draw a cascade's static
//! casters are already a handful of calls, and the union of nested cascades' casters drawn into
//! every view would multiply their vertex work), the restores (each layer's own rectangles), the
//! sky map (no entity pass), and the lamp atlas (a cube's six faces are tiles of ONE array layer,
//! which a multiview pass cannot address: its views are layers).

use std::num::NonZeroU32;
use std::sync::OnceLock;

use super::*;
use crate::modern::caps::HwCaps;

/// The pass masks the lane draws with, one pipeline variant each: c0-c1, c0-c2, and c0, c1, c3
/// (selective). wgpu-core lets a pipeline into a multiview pass only under its own mask.
pub(super) const MASKS: [u32; 3] = [0b0011, 0b0111, 0b1011];

/// One part's multiview pipelines, by [`MASKS`] slot (`None` where the slot is not wanted).
pub(super) type MaskPipelines = [Option<CachedRenderPipelineId>; MASKS.len()];

/// The span of the multiview pass, in `WOW_GPU_PASSES` beside `modern_shadow_dynamic_c0..c3`.
const SPAN: &str = "modern_shadow_dynamic_mv";

/// The multiview pipelines' group 0: every cascade's clip matrix, row `i` for layer `i`.
#[derive(Clone, Copy, Default, ShaderType)]
pub(super) struct MultiviewViews {
    clip_from_world: [Mat4; MAX_CASCADES],
}

/// `WOW_SHADOW_MULTIVIEW=0` turns the lane off (default on where the device has multiview).
pub(crate) fn switch_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::launch_env::var("WOW_SHADOW_MULTIVIEW").as_deref() != Some("0"))
}

/// A multiview draw's one instance: the part's caster-table slot above its view mask's
/// [`MAX_CASCADES`] bits (`entity_shadow.wesl`, `MULTIVIEW`).
fn instance_of(slot: u32, views: u32) -> u32 {
    slot << MAX_CASCADES | views
}

/// The layers a pass under `mask` spans: `0..=` its highest bit.
fn span_of(mask: u32) -> u32 {
    32 - mask.leading_zeros()
}

/// The multiview pipelines' group 0 layout.
pub(super) fn view_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "modern_shadow_multiview_views",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            uniform_buffer::<MultiviewViews>(false),
        ),
    )
}

/// Which [`MASKS`] the device, the switch and `count` live cascades allow.
pub(super) fn wanted(caps: Option<&HwCaps>, count: u32) -> [bool; MASKS.len()] {
    let mut out = [false; MASKS.len()];
    let Some(caps) = caps else {
        return out;
    };
    if !switch_on() {
        return out;
    }
    for (k, &mask) in MASKS.iter().enumerate() {
        let span = span_of(mask);
        let contiguous = mask == (1u32 << span) - 1;
        out[k] = span <= count.min(MAX_CASCADES as u32)
            && span <= caps.multiview_views
            && (contiguous || caps.multiview_selective);
    }
    out
}

/// This frame's mask slot: the cascades whose entity pass runs if their layers are cached
/// ([`holds_entities`]), as one selective mask where the device takes it, else the run of them
/// from c0. A held layer redrawn whole this frame (re-centred) is outside the mask and draws all
/// of its parts in its own pass.
fn pick(wanted: &[bool; MASKS.len()], count: u32, frame_no: u32, selective: bool) -> Option<usize> {
    let mut drawn = 0u32;
    for i in 0..(count as usize).min(MAX_CASCADES) {
        if !holds_entities(i, frame_no) {
            drawn |= 1u32 << i;
        }
    }
    let run = (1u32 << (!drawn).trailing_zeros()) - 1;
    let candidates = if selective { [drawn, run] } else { [run, run] };
    candidates.into_iter().find_map(|mask| {
        if mask.count_ones() < 2 {
            return None;
        }
        MASKS.iter().position(|&m| m == mask).filter(|&k| wanted[k])
    })
}

/// The queued multiview pipelines of one part, for every wanted mask.
pub(super) fn pipelines(
    pipes: &mut EntityShadowPipelines,
    cache: &PipelineCache,
    mesh: &RenderMesh,
    skinned: bool,
    cutout: bool,
    wanted: &[bool; MASKS.len()],
) -> MaskPipelines {
    let mut out: MaskPipelines = [None; MASKS.len()];
    for (k, &mask) in MASKS.iter().enumerate() {
        if wanted[k] {
            out[k] = pipes.get_multiview(cache, mesh, skinned, cutout, mask);
        }
    }
    out
}

impl EntityShadowPipelines {
    /// [`Self::get`]'s multiview twin: the `MULTIVIEW` variant under `mask` (never the lamp
    /// atlas's fixture clip), queued on first sight through the pipeline cache's multiview entry.
    fn get_multiview(
        &mut self,
        cache: &PipelineCache,
        mesh: &RenderMesh,
        skinned: bool,
        cutout: bool,
        mask: u32,
    ) -> Option<CachedRenderPipelineId> {
        let key = (mesh.layout.clone(), skinned, cutout, mask);
        if let Some(id) = self.mv_by_key.get(&key) {
            return Some(*id);
        }
        let descriptor = self.descriptor(mesh, skinned, cutout, false, true)?;
        let id = cache.queue_multiview_render_pipeline(descriptor, NonZeroU32::new(mask)?);
        self.mv_by_key.insert(key, id);
        Some(id)
    }
}

/// `PrepareResources`, from [`prepare_modern_shadow`] once the cascade block is live: the layer
/// views (made once per working texture) and this frame's matrices and bind group. Leaves
/// `mv_bind` `None` wherever the lane cannot draw this frame.
pub(super) fn prepare(
    device: &RenderDevice,
    queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    caps: Option<&HwCaps>,
    cascades: &ModernCascades,
    texture: &Texture,
    frame: &mut ModernShadowFrame,
) {
    frame.mv_bind = None;
    if !wanted(caps, cascades.count).contains(&true) {
        return;
    }
    if frame.mv_views.is_empty() {
        let layers = texture.depth_or_array_layers().min(MAX_CASCADES as u32);
        frame.mv_views = (2..=layers)
            .map(|n| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some("modern_shadow_multiview"),
                    dimension: Some(TextureViewDimension::D2Array),
                    base_array_layer: 0,
                    array_layer_count: Some(n),
                    ..default()
                })
            })
            .collect();
    }
    // Rows past the live cascades stay the identity; no mask reaches them.
    let mut views = MultiviewViews::default();
    let live = (cascades.count as usize).min(MAX_CASCADES);
    views.clip_from_world[..live].copy_from_slice(&cascades.clip_from_world[..live]);
    frame.mv_uniform.set(views);
    frame.mv_uniform.write_buffer(device, queue);
    let Some(binding) = frame.mv_uniform.binding() else {
        return;
    };
    let bind = device.create_bind_group(
        "modern_shadow_multiview_views",
        &pipeline_cache.get_bind_group_layout(&view_layout()),
        &BindGroupEntries::single(binding),
    );
    frame.mv_bind = Some(bind);
}

/// This frame's multiview pass: which parts it draws, into which layers.
pub(super) struct Plan<'a> {
    slot: usize,
    mask: u32,
    view: &'a TextureView,
    bind: &'a BindGroup,
    /// Per entry of [`EntityShadowFrame::draws`]: the layers of `mask` the pass draws it into
    /// (two or more), or 0 where the layers' own passes draw it.
    taken: Vec<u32>,
}

impl Plan<'_> {
    /// Does the multiview pass draw entity draw `j` into layer `i`? Then that layer's own pass
    /// marks its tiles and skips the draw.
    pub(super) fn takes(&self, j: usize, i: usize) -> bool {
        self.taken.get(j).is_some_and(|m| m & (1u32 << i) != 0)
    }
}

/// Plan this frame's multiview pass, before any layer records, or `None`: the lane is off, the
/// frame has fewer than two cascades with an entity pass, or no part two of them take has its
/// pipeline compiled yet (such a part draws per layer meanwhile).
pub(super) fn plan<'a>(
    world: &World,
    frame: &'a ModernShadowFrame,
    cascades: &ModernCascades,
    pipeline_cache: &PipelineCache,
) -> Option<Plan<'a>> {
    let caps = world.get_resource::<HwCaps>()?;
    let bind = frame.mv_bind.as_ref()?;
    let entities = world.get_resource::<EntityShadowFrame>()?;
    if entities.draw_bind.is_none() || entities.draws.is_empty() {
        return None;
    }
    let wanted = wanted(Some(caps), cascades.count);
    let slot = pick(
        &wanted,
        cascades.count,
        frame.frame_no,
        caps.multiview_selective,
    )?;
    let mask = MASKS[slot];
    let view = frame.mv_views.get(span_of(mask) as usize - 2)?;
    let culls: Vec<(usize, Mat4)> = (0..MAX_CASCADES)
        .filter(|&i| mask & (1u32 << i) != 0)
        .map(|i| (i, entity_cull_for(cascades, i)))
        .collect();
    let mut any = false;
    let taken = entities
        .draws
        .iter()
        .map(|cmd| {
            // The same test each layer's own loop makes.
            if !cmd.sun {
                return 0;
            }
            let views = culls.iter().fold(0u32, |m, (i, cull)| {
                if outside_box(cull, cmd.center, Vec3::splat(cmd.radius)) {
                    m
                } else {
                    m | 1u32 << i
                }
            });
            let ready = cmd.mv[slot].and_then(|id| pipeline_cache.get_render_pipeline(id));
            if views.count_ones() >= 2 && ready.is_some() {
                any = true;
                views
            } else {
                0
            }
        })
        .collect();
    any.then_some(Plan {
        slot,
        mask,
        view,
        bind,
        taken,
    })
}

/// Record the multiview pass: after every cascade layer's own job (restore, static, its own
/// parts), so each layer's depth is final but for these parts. Depth compares `GreaterEqual`, so
/// the order of parts within the layer does not change what it holds.
pub(super) fn draw(
    render_context: &mut Recorder<'_>,
    world: &World,
    plan: &Plan<'_>,
    pipeline_cache: &PipelineCache,
    meshes: &RenderAssets<RenderMesh>,
    allocator: &MeshAllocator,
) {
    let Some(entities) = world.get_resource::<EntityShadowFrame>() else {
        return;
    };
    let Some(draw_bind) = &entities.draw_bind else {
        return;
    };
    let diagnostics = render_context.diagnostic_recorder();
    // Timed on the encoder, around the pass: a query written inside a multiview pass takes one
    // index per view on Vulkan, which the recorder's spans do not reserve. So the pass has a time
    // in `WOW_GPU_PASSES` and no pipeline statistics in `WOW_GPU_STATS`.
    let span = diagnostics.time_span(render_context.command_encoder(), SPAN);
    {
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some(SPAN),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: plan.view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: NonZeroU32::new(plan.mask),
        });
        pass.set_bind_group(0, plan.bind, &[]);
        for (cmd, &views) in entities.draws.iter().zip(&plan.taken) {
            if views == 0 {
                continue;
            }
            let Some(pipeline) =
                cmd.mv[plan.slot].and_then(|id| pipeline_cache.get_render_pipeline(id))
            else {
                continue;
            };
            let Some(mesh) = meshes.get(cmd.mesh) else {
                continue;
            };
            let Some(vslice) = allocator.mesh_vertex_slice(&cmd.mesh) else {
                continue;
            };
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(1, draw_bind, &[]);
            if let Some(cut) = &cmd.cutout {
                pass.set_bind_group(2, cut, &[]);
            }
            pass.set_vertex_buffer(0, vslice.buffer.slice(..));
            // The part's one instance packs its slot and its view mask (`entity_shadow.wesl`,
            // `MULTIVIEW`, [`instance_of`]).
            let instance = instance_of(cmd.slot, views);
            match &mesh.buffer_info {
                bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } => {
                    let Some(islice) = allocator.mesh_index_slice(&cmd.mesh) else {
                        continue;
                    };
                    pass.set_index_buffer(islice.buffer.slice(..), *index_format);
                    crate::static_gx::count_draw(1);
                    pass.draw_indexed(
                        islice.range.clone(),
                        i32::try_from(vslice.range.start).unwrap_or(0),
                        instance..instance + 1,
                    );
                }
                bevy::render::mesh::RenderMeshBufferInfo::NonIndexed => {
                    pass.draw(vslice.range.clone(), instance..instance + 1);
                }
            }
        }
    }
    span.end(render_context.command_encoder());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame's mask follows the far cascades' turns: c0-c2, then c0, c1, c3 where a mask may
    /// skip a layer, else c0-c1; one live cascade or the switch off gives none.
    #[test]
    fn the_mask_follows_the_cascades_drawn_this_frame() {
        let all = [true; MASKS.len()];
        let even = (0..2).find(|&f| !holds_entities(2, f)).unwrap();
        let odd = even + 1;
        assert_eq!(pick(&all, 4, even, true), Some(1));
        assert_eq!(pick(&all, 4, odd, true), Some(2));
        assert_eq!(pick(&all, 4, odd, false), Some(0));
        assert_eq!(pick(&all, 3, odd, true), Some(0));
        assert_eq!(pick(&all, 2, even, true), Some(0));
        assert_eq!(pick(&all, 1, even, true), None);
        assert_eq!(pick(&[true, false, false], 4, even, true), None);
    }

    /// A mask is wanted only within the live cascades, the device's views and its selectivity.
    #[test]
    fn a_mask_needs_its_views_and_selectivity() {
        let caps = HwCaps {
            multiview_views: 4,
            multiview_selective: false,
            ..HwCaps::default()
        };
        if switch_on() {
            assert_eq!(wanted(Some(&caps), 4), [true, true, false]);
            assert_eq!(wanted(Some(&caps), 2), [true, false, false]);
            let selective = HwCaps {
                multiview_selective: true,
                ..caps
            };
            assert_eq!(wanted(Some(&selective), 4), [true, true, true]);
            let three = HwCaps {
                multiview_views: 3,
                ..selective
            };
            assert_eq!(wanted(Some(&three), 4), [true, true, false]);
        }
        assert_eq!(wanted(None, 4), [false; MASKS.len()]);
    }

    /// The shader's matrix table holds every cascade.
    #[test]
    fn the_shader_table_holds_every_cascade() {
        let src = include_str!("../../../shaders/modern/entity_shadow.wesl");
        let row = format!("clip_from_world: array<mat4x4<f32>, {MAX_CASCADES}>");
        assert!(src.contains(&row));
        for &mask in &MASKS {
            assert!(span_of(mask) as usize <= MAX_CASCADES);
        }
    }

    /// The instance packs the slot above the mask's bits, and the shader unpacks it the same way.
    #[test]
    fn the_instance_packs_slot_and_mask_as_the_shader_reads_them() {
        let src = include_str!("../../../shaders/modern/entity_shadow.wesl");
        let mask_bits = (1u32 << MAX_CASCADES) - 1;
        assert!(src.contains(&format!("parts[v.instance >> {MAX_CASCADES}u]")));
        assert!(src.contains(&format!("v.instance & {mask_bits}u")));
        for &mask in &MASKS {
            let i = instance_of(677, mask);
            assert_eq!((i >> MAX_CASCADES, i & mask_bits), (677, mask));
        }
    }
}
