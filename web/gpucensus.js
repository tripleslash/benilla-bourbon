// gpucensus.js — `?gpucensus=1`: count the WebGPU objects the agent that renders makes, and those
// its garbage collector has since freed, by kind. An object the backend cannot `destroy()` (a bind
// group, a view, a command buffer) is freed only when its wrapper is collected, so "made − freed"
// climbing without end is memory the GPU process holds for nothing.
//
// The render worker installs it on itself (`renderworker.js`); with `?renderworker=0` the page
// installs it on the page (`threads.js`). Nothing here runs, and no prototype is touched, without
// the flag.

/// Wrap the creating methods on this agent's WebGPU prototypes. Returns `snapshot()`, the counts
/// so far.
export function installGpuCensus() {
  const made = Object.create(null);
  const freed = Object.create(null);
  const registry = new FinalizationRegistry((kind) => {
    freed[kind] = (freed[kind] || 0) + 1;
  });
  const watch = (proto, method, kind) => {
    const orig = proto?.[method];
    if (typeof orig !== 'function') return;
    proto[method] = function (...args) {
      const obj = orig.apply(this, args);
      made[kind] = (made[kind] || 0) + 1;
      if (obj && typeof obj === 'object') registry.register(obj, kind);
      return obj;
    };
  };
  watch(self.GPUDevice?.prototype, 'createBindGroup', 'bindGroup');
  watch(self.GPUDevice?.prototype, 'createBuffer', 'buffer');
  watch(self.GPUDevice?.prototype, 'createTexture', 'texture');
  watch(self.GPUDevice?.prototype, 'createSampler', 'sampler');
  watch(self.GPUDevice?.prototype, 'createCommandEncoder', 'encoder');
  watch(self.GPUTexture?.prototype, 'createView', 'view');
  watch(self.GPUCommandEncoder?.prototype, 'beginRenderPass', 'pass');
  watch(self.GPUCommandEncoder?.prototype, 'beginComputePass', 'computePass');
  watch(self.GPUCommandEncoder?.prototype, 'finish', 'commandBuffer');
  watch(self.GPUCanvasContext?.prototype, 'getCurrentTexture', 'canvasTexture');
  return () => ({ made: { ...made }, freed: { ...freed } });
}

/// One report as a console line: the shared linear memory, the JS values Rust holds on the
/// reporting agent (wasm-bindgen's externref table, which only grows), and per kind the objects
/// alive (made, less freed) with how many were made since `last`.
export function censusLine(where, { made, freed, refs, mib }, last) {
  const parts = Object.keys(made)
    .sort()
    .map((k) => `${k}=${made[k] - (freed[k] || 0)}(+${made[k] - (last?.[k] || 0)})`);
  return `[gpucensus] ${where} refs=${refs} mem=${mib}MiB alive(+made/5s): ${parts.join(' ')}`;
}
