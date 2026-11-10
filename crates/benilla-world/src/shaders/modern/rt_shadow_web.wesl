// fork: the browser's `benilla_modern::rtshadow`, with nothing in it. Bevy's shader cache composes
// every `#import` a shader names, inside an `#ifdef` or not, and holds a pipeline whose import is
// unregistered queued for ever, without an error. Each receiver of the ray-traced shadows
// (`wow_model`, `terrain`, `static_gx`, `liquid_cata`, `froxel_scatter`, `gi_trace`) names this
// path under `RT_SHADOWS` / `RT_FROXEL`; the web never sets either, so it needs the name and no
// item, and the real library (`rt_shadow.wgsl`, ray queries) stays native.
