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
  // Buffers and textures made since the last snapshot, by label: which pass churns them.
  let labels = Object.create(null);
  const registry = new FinalizationRegistry((kind) => {
    freed[kind] = (freed[kind] || 0) + 1;
  });
  const watch = (proto, method, kind) => {
    const orig = proto?.[method];
    if (typeof orig !== 'function') return;
    const labelled = kind === 'buffer' || kind === 'texture' || kind === 'bindGroup';
    proto[method] = function (...args) {
      const obj = orig.apply(this, args);
      made[kind] = (made[kind] || 0) + 1;
      if (labelled) {
        const d = args[0];
        // A bind group by its layout's label; an unlabelled buffer by its size and usage.
        const name =
          kind === 'bindGroup'
            ? `${d?.label || '?'}@${d?.layout?.label || '?'}`
            : d?.label || (kind === 'buffer' ? `?${d?.size}u${d?.usage}` : '?');
        const key = `${kind}:${name}`;
        labels[key] = (labels[key] || 0) + 1;
      }
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
  const gpu = watchGpuFrames();
  return () => {
    const top = Object.entries(labels)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    labels = Object.create(null);
    return { made: { ...made }, freed: { ...freed }, top, gpu: gpu() };
  };
}

/// When the GPU finished each frame: a frame is the work submitted between two `getCurrentTexture`
/// calls, and it is done when `onSubmittedWorkDone`, asked after its last submit, resolves. Returns
/// a summary of the window since the last call: `lat` from that submit to done, `gap` between two
/// frames' done. A `gap` that stays above the refresh while `lat` grows is a GPU that cannot keep up.
function watchGpuFrames() {
  const submit = self.GPUQueue?.prototype?.submit;
  const current = self.GPUCanvasContext?.prototype?.getCurrentTexture;
  if (typeof submit !== 'function' || typeof current !== 'function') return () => '';
  let last = null;
  let lastDone = 0;
  let lat = [];
  let gap = [];
  self.GPUQueue.prototype.submit = function (...args) {
    const r = submit.apply(this, args);
    last = { queue: this, at: performance.now() };
    return r;
  };
  self.GPUCanvasContext.prototype.getCurrentTexture = function (...args) {
    if (last) {
      const { queue, at } = last;
      queue.onSubmittedWorkDone().then(() => {
        const now = performance.now();
        lat.push(now - at);
        if (lastDone) gap.push(now - lastDone);
        lastDone = now;
      });
      last = null;
    }
    return current.apply(this, args);
  };
  const q = (xs, p) => (xs.length ? xs[Math.min(xs.length - 1, Math.floor(p * xs.length))].toFixed(2) : '-');
  return () => {
    const l = lat.sort((a, b) => a - b);
    const g = gap.sort((a, b) => a - b);
    lat = [];
    gap = [];
    return `frames=${l.length} lat p50=${q(l, 0.5)} p90=${q(l, 0.9)} max=${q(l, 1)} gap p50=${q(g, 0.5)} p90=${q(g, 0.9)} p99=${q(g, 0.99)}`;
  };
}

/// One report as a console line: the shared linear memory, the JS values Rust holds on the
/// reporting agent (wasm-bindgen's externref table, which only grows), and per kind the objects
/// alive (made, less freed) with how many were made since `last`.
export function censusLine(where, { made, freed, refs, mib, top, handshake, gpu }, last) {
  const parts = Object.keys(made)
    .sort()
    .map((k) => `${k}=${made[k] - (freed[k] || 0)}(+${made[k] - (last?.[k] || 0)})`);
  const churn = (top || []).map(([label, n]) => `${label}×${n}`).join(' ');
  const hs = handshake ? ` | handshake: ${handshake}` : '';
  const done = gpu ? ` | gpu: ${gpu}` : '';
  return `[gpucensus] ${where} refs=${refs} mem=${mib}MiB alive(+made/5s): ${parts.join(' ')} | made by label: ${churn}${hs}${done}`;
}
