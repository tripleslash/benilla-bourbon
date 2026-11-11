//! A frame's explicit GUID uses the stock binder, relations and range inputs without selection.

use super::{owner_store, ArmCandidate, CastContext, CastTargeting};

impl CastTargeting<'_, '_> {
    /// A frame action's explicit `target` chain reads the selection at this ordered call.
    pub(crate) fn pfui_selected_target(&self, hops: usize) -> Option<u64> {
        let mut guid = self.selection.guid?;
        for _ in 0..hops {
            let entity = *self.index.as_ref()?.0.get(&guid)?;
            let store = self
                .stores
                .get(entity)
                .ok()
                .filter(|store| store.is_unit())?;
            guid = store.0.unit_target()?;
        }
        self.pfui_context(guid).map(|_| guid)
    }

    pub(crate) fn pfui_context(&self, guid: u64) -> Option<CastContext<'_>> {
        let entity = *self.index.as_ref()?.0.get(&guid)?;
        let store = self.stores.get(entity).ok()?;
        if !store.is_unit() {
            return None;
        }
        let mut ctx = self.context();
        // The guid passed as `TryCast`'s, which the arm reads again past its signal.
        ctx.candidate = ArmCandidate::Guid(guid);
        ctx.selection_guid = Some(guid);
        // The object manager holds it (found above), whatever the real selection holds now.
        ctx.selection_held = true;
        // An explicit frame target must never fall back to healing the caster.
        ctx.auto_self_cast = false;
        ctx.rel.target_store = Some(store);
        ctx.rel.target_owner_store = owner_store(Some(store), self.index.as_deref(), &self.stores);
        ctx.rel.group.target_guid = Some(guid);
        ctx.range.target_pos = self.transforms.get(entity).ok().map(|t| t.translation);
        ctx.range.target = self.range_units.unit(entity);
        Some(ctx)
    }
}
