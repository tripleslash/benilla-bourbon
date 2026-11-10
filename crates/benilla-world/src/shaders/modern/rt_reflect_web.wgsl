// fork: the browser's `benilla_modern::rtreflect`, with nothing in it, for the reason
// `rt_shadow_web.wgsl` gives: the cata water names this path under `RT_REFLECT`, which the web
// never sets, and an unregistered import would hold its pipeline queued for ever.
#define_import_path benilla_modern::rtreflect
