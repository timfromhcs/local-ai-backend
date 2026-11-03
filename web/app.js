/* LocalAI Gateway demo.
   Browser path: ONNX (Transformers.js) on WebGPU, falling back to WASM.
   Lab: trains a real softmax head on ONNX embeddings (teacher-labelled) and
   benchmarks it before/after — plus prompt-variant selection.
   Local path: talks to the gateway on 127.0.0.1. No page-controlled server. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const t = (key, fallback) => (window.I18N?.t(key) ?? fallback ?? key);

/* ------------------------------------------------------------------ tabs */
for (const btn of $$('.tabs button')) {
  btn.addEventListener('click', () => {
    $$('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
    for (const panel of $$('.panel[id^="panel-"]')) panel.hidden = panel.id !== `panel-${btn.dataset.tab}`;
  });
}

function bubble(host, role, text) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.textContent = text;
  host.append(el);
  host.scrollTop = host.scrollHeight;
  return el;
}

function setStatus(el, text, kind = '') {
  el.textContent = text;
  el.className = `status ${kind}`;
}

const fmtGB = (b) => `${(b / 1024 ** 3).toFixed(1)} GB`;

/* ---------------------------------------------------------- ONNX / WebGPU */
const MODELS = {
  small: { id: 'onnx-community/Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 0.5B' },
  medium: { id: 'onnx-community/Llama-3.2-1B-Instruct', label: 'Llama 3.2 1B' },
  large: { id: 'onnx-community/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5 1.5B' },
};
const EMBEDDER = 'Xenova/all-MiniLM-L6-v2';
const CDN = 'https://esm.run/@huggingface/transformers';

const state = {
  transformers: null,
  generator: null,
  backend: null,
  model: null,
  embedder: null,
  local: { base: '', model: '' },
};

async function detectHardware() {
  const el = $('#gpu-status');
  const hasGpu = 'gpu' in navigator;
  let adapter = null;
  if (hasGpu) {
    try { adapter = await navigator.gpu.requestAdapter(); } catch { adapter = null; }
  }
  const info = adapter?.info ?? {};
  const mem = navigator.deviceMemory ?? null;
  const backend = adapter ? 'WebGPU' : 'WASM';
  state.backend = backend;
  const parts = [
    backend,
    info.vendor || info.architecture || (adapter ? 'GPU' : 'CPU'),
    adapter?.limits?.maxBufferSize ? `maxBuffer ${fmtGB(adapter.limits.maxBufferSize)}` : '',
    mem ? `deviceMemory ~${mem} GB` : '',
  ].filter(Boolean);
  setStatus(el, `${t('demo.detected')}: ${parts.join(' · ')}`, 'good');
  if (!adapter) {
    $('#tier').value = 'small';
    bubble($('#chat-browser'), 'sys', t('demo.wasmOnly'));
  }
  return { backend, adapter };
}

async function lib() {
  if (!state.transformers) state.transformers = await import(CDN);
  return state.transformers;
}

function humanBytes(n) {
  if (!n) return '';
  return n > 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${Math.round(n / 1024 ** 2)} MB`;
}

async function loadGenerator(modelId, onProgress) {
  const { pipeline, env } = await lib();
  env.allowLocalModels = false;
  const device = state.backend === 'WebGPU' ? 'webgpu' : 'wasm';
  const dtype = device === 'webgpu' ? 'q4f16' : 'q4';
  return pipeline('text-generation', modelId, { device, dtype, progress_callback: onProgress });
}

$('#load-model').addEventListener('click', async () => {
  if (state.transformers === null && $('#load-model').dataset.busy === '1') return;
  const tier = $('#tier').value;
  const modelId = MODELS[tier].id;
  const box = $('#load-progress');
  const bar = $('#load-bar');
  const btn = $('#load-model');
  btn.dataset.busy = '1';
  btn.disabled = true;
  box.hidden = false;
  bubble($('#chat-browser'), 'sys', `${t('demo.loading')} ${modelId} (${state.backend})`);
  const started = performance.now();
  try {
    state.generator = await loadGenerator(modelId, (p) => {
      if (p.status === 'progress' && p.total) {
        bar.style.width = `${Math.round((p.loaded / p.total) * 100)}%`;
        setStatus($('#gpu-status'), `${p.file ?? ''} ${humanBytes(p.loaded)}/${humanBytes(p.total)} · ${state.backend}`);
      }
    });
    state.model = modelId;
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    bubble($('#chat-browser'), 'sys', `${t('demo.ready')} (${secs}s, ${state.backend})`);
    setStatus($('#gpu-status'), `${t('demo.ready')} · ${MODELS[tier].label} · ${state.backend}`, 'good');
  } catch (e) {
    bubble($('#chat-browser'), 'sys', `${t('demo.loadFailed')}: ${e.message}`);
    setStatus($('#gpu-status'), String(e.message).slice(0, 200), 'bad');
    state.generator = null;
  } finally {
    btn.dataset.busy = '0';
    btn.disabled = false;
    box.hidden = true;
  }
});

async function generate(system, user, maxTokens = 220) {
  const messages = [
    { role: 'system', content: system },
    { role: 'user', content: user },
  ];
  const out = await state.generator(messages, { max_new_tokens: maxTokens, do_sample: false });
  const first = out?.[0]?.generated_text;
  if (Array.isArray(first)) return first.at(-1)?.content?.trim() ?? '';
  return String(first ?? '').replace(messages.map((m) => m.content).join(''), '').trim();
}

$('#form-browser').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = $('#prompt-browser');
  const text = input.value.trim();
  if (!text) return;
  if (!state.generator) { bubble($('#chat-browser'), 'sys', t('demo.loadFirst')); return; }
  input.value = '';
  bubble($('#chat-browser'), 'user', text);
  const out = bubble($('#chat-browser'), 'bot', '…');
  const started = performance.now();
  try {
    out.textContent = await generate('You are a concise, accurate local assistant. Answer in at most 120 words.', text, 260);
    out.textContent += `\n\n— ${((performance.now() - started) / 1000).toFixed(1)}s · ${state.backend}`;
  } catch (e) {
    out.textContent = `${t('demo.failed')}: ${e.message}`;
  }
});

/* --------------------------------------------------------------- the lab
   Real training in this tab: freeze an ONNX sentence encoder, build a
   teacher-labelled dataset, train a 3-class softmax head with SGD, then
   compare accuracy before (untrained head) and after training. */
const ROUTES = ['fast', 'duo', 'bigout'];

const TEACHER = [
  ['rename this variable', 0], ['what is 2+2', 0], ['write a title for this chat', 0],
  ['summarise this one line', 0], ['format this json', 0], ['what time is it in UTC', 0],
  ['implement a login form and run the tests', 1], ['refactor the scheduler and verify', 1],
  ['add caching to the api endpoint', 1], ['fix the failing test in auth.py', 1],
  ['update the readme and run pytest', 1], ['wire a new route into the gateway', 1],
  ['migrate the whole repo to pydantic v2 and keep tests green', 2],
  ['design a memory-aware scheduler for four models and prove it with benchmarks', 2],
  ['debug why the streaming response leaks a model slot under load', 2],
  ['build a release pipeline with a gate and rollback, then verify each step', 2],
  ['write a spec for multi-file refactoring with correctness guarantees', 2],
  ['why does the same prompt behave differently after a restart, and how do we prove it', 2],
];
const TEST = [
  ['capitalise this string', 0], ['count the words in this sentence', 0],
  ['add a status endpoint and test it', 1], ['clean up dead code in this module', 1],
  ['redesign the whole data model and verify every migration', 2],
  ['analyse the failure across three services and produce a fix plan', 2],
];

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

class SoftmaxHead {
  constructor(dim, classes = 3) {
    const rand = mulberry(42);
    this.w = Array.from({ length: classes }, () => Array.from({ length: dim }, () => (rand() - 0.5) * 0.02));
    this.b = new Array(classes).fill(0);
  }
  predict(x) {
    return softmax(this.w.map((row, k) => row.reduce((a, v, i) => a + v * x[i], this.b[k])));
  }
  train(X, y, { epochs = 220, lr = 0.35 } = {}) {
    const n = X.length;
    const history = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const p = this.predict(X[i]);
        loss -= Math.log(Math.max(p[y[i]], 1e-9));
        for (let k = 0; k < this.w.length; k++) {
          const g = p[k] - (y[i] === k ? 1 : 0);
          for (let d = 0; d < this.w[k].length; d++) this.w[k][d] -= (lr * g * X[i][d]) / n;
          this.b[k] -= (lr * g) / n;
        }
      }
      if (epoch % 20 === 0) history.push(+(loss / n).toFixed(4));
    }
    return history;
  }
  accuracy(X, y) {
    let ok = 0;
    X.forEach((x, i) => {
      const p = this.predict(x);
      if (p.indexOf(Math.max(...p)) === y[i]) ok++;
    });
    return ok / X.length;
  }
}

async function embed(texts, onStep) {
  const { pipeline } = await lib();
  if (!state.embedder) state.embedder = await pipeline('feature-extraction', EMBEDDER, { device: 'wasm' });
  const out = [];
  for (let i = 0; i < texts.length; i++) {
    const r = await state.embedder(texts[i], { pooling: 'mean', normalize: true });
    out.push(Array.from(r.data));
    onStep?.(i + 1, texts.length);
  }
  return out;
}

$('#run-lab').addEventListener('click', async () => {
  const out = $('#lab-out');
  const btn = $('#run-lab');
  btn.disabled = true;
  out.innerHTML = `<p class="note">${t('lab.embedding')} ${EMBEDDER}…</p>`;
  performance.mark('lab-start');
  try {
    const trainX = await embed(TEACHER.map(([x]) => x));
    const trainY = TEACHER.map(([, y]) => y);
    const testX = await embed(TEST.map(([x]) => x));
    const testY = TEST.map(([, y]) => y);

    const head = new SoftmaxHead(trainX[0].length, 3);
    const before = head.accuracy(testX, testY);
    const loss = head.train(trainX, trainY);
    const after = head.accuracy(testX, testY);
    const trainAcc = head.accuracy(trainX, trainY);

    const rows = TEST.map(([text], i) => {
      const p = head.predict(testX[i]);
      const pred = p.indexOf(Math.max(...p));
      return `<tr><td>${text}</td><td>${ROUTES[testY[i]]}</td><td>${ROUTES[pred]}</td><td>${p[pred].toFixed(2)}</td><td>${pred === testY[i] ? '✓' : '✗'}</td></tr>`;
    }).join('');

    out.innerHTML = `
      <p class="note">${t('lab.backend')}: <b>${state.backend}</b> · ${t('lab.model')}: <b>${state.model ?? '—'}</b> · ${t('lab.embedder')}: ${EMBEDDER} (384d)</p>
      <table class="mini">
        <tbody>
          <tr><th>${t('lab.beforeAcc')}</th><td><b>${(before * 100).toFixed(0)}%</b></td></tr>
          <tr><th>${t('lab.afterAcc')}</th><td><b>${(after * 100).toFixed(0)}%</b></td></tr>
          <tr><th>${t('lab.trainAcc')}</th><td>${(trainAcc * 100).toFixed(0)}%</td></tr>
          <tr><th>${t('lab.loss')}</th><td>${loss.join(' → ')}</td></tr>
        </tbody>
      </table>
      <div class="table-scroll"><table><thead><tr><th>${t('lab.task')}</th><th>${t('lab.want')}</th><th>${t('lab.got')}</th><th>p</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="ok">${t('lab.done')} — ${t('lab.limit')}</p>`;
    state.lab = { before, after, trainAcc, loss, model: state.model, backend: state.backend };
    const saved = await saveLabRun({
      kind: 'lab-run',
      at: new Date().toISOString(),
      model: state.model,
      backend: state.backend,
      metrics: state.lab,
      rows: TEST.map(([text], i) => ({ task: text, score: head.predict(testX[i]).indexOf(Math.max(...head.predict(testX[i]))) === testY[i] })),
    }).catch(() => null);
    if (saved) out.insertAdjacentHTML('beforeend', `<p class="note">${t('storage.saved')}: <code>${saved}</code></p>`);
    $('#download-lab').disabled = false;
  } catch (e) {
    out.innerHTML = `<p class="status bad">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

$('#download-lab').addEventListener('click', () => {
  const payload = {
    kind: 'onnx-teacher-student-head',
    createdAt: new Date().toISOString(),
    backend: state.lab?.backend,
    baseModel: state.lab?.model,
    embedder: EMBEDDER,
    dataset: { train: TEACHER.length, test: TEST.length, classes: ROUTES },
    metrics: state.lab,
    note: 'Frozen ONNX encoder + softmax head trained in-browser with SGD. Weights of the LLM are not modified.',
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'onnx-distill-run.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ---------------------------------------------------------- local gateway */
async function probeGateway() {
  const base = $('#base').value.trim().replace(/\/$/, '');
  const el = $('#local-status');
  setStatus(el, t('demo.probing'));
  try {
    const health = await fetch(`${base}/healthz`, { signal: AbortSignal.timeout(6000) }).then((r) => r.json());
    const list = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(6000) }).then((r) => r.json());
    const sel = $('#model');
    sel.innerHTML = '<option value="">—</option>';
    const ready = (list.data ?? []).filter((m) => m.status === 'ready' && m.kind === 'llm');
    for (const m of list.data ?? []) {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = `${m.id} · ${m.kind} · ${m.status}`;
      o.disabled = m.status !== 'ready' || m.kind !== 'llm';
      sel.append(o);
    }
    if (ready[0]) sel.value = ready[0].id;
    state.local = { base, model: sel.value };
    setStatus(el, `${health.status === 'ok' ? t('demo.online') : t('demo.degraded')} · ${ready.length} ${t('demo.readyModels')} · ${base}`, 'good');
    return true;
  } catch (e) {
    setStatus(el, `${t('demo.offline')} — ${e.message}. ${t('demo.offlineHint')}`, 'bad');
    return false;
  }
}

$('#probe').addEventListener('click', probeGateway);
$('#model').addEventListener('change', (e) => { state.local.model = e.target.value; });

$('#form-local').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = $('#prompt-local');
  const text = input.value.trim();
  if (!text) return;
  if (!state.local.base || !state.local.model) { bubble($('#chat-local'), 'sys', t('demo.connectFirst')); return; }
  input.value = '';
  bubble($('#chat-local'), 'user', text);
  const out = bubble($('#chat-local'), 'bot', '…');
  const started = performance.now();
  let chunks = 0;
  try {
    const res = await fetch(`${state.local.base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: state.local.model, messages: [{ role: 'user', content: text }], stream: true, max_tokens: 512 }),
    });
    if (!res.ok || !res.body) {
      out.textContent = `${t('demo.failed')}: ${res.status} ${(await res.text()).slice(0, 240)}`;
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let acc = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          try {
            acc += JSON.parse(payload).choices?.[0]?.delta?.content ?? '';
            chunks += 1;
          } catch { /* partial */ }
        }
        out.textContent = acc || '…';
        $('#chat-local').scrollTop = $('#chat-local').scrollHeight;
      }
    }
    out.textContent = acc || t('demo.empty');
    if (acc) out.textContent += `\n\n— ${chunks} ${t('demo.chunks')}, ${((performance.now() - started) / 1000).toFixed(1)}s`;
  } catch (e) {
    out.textContent = `${t('demo.failed')}: ${e.message}`;
  }
});

/* ------------------------------------------------------- GGUF in the browser
   llama.cpp compiled to WebAssembly (wllama). Same GGUF format as the local
   gateway, no server in the loop. Single-threaded without COOP/COEP. */
const GGUF = {
  lfm350: { repo: 'LiquidAI/LFM2.5-350M-GGUF', file: 'LFM2.5-350M-Q4_K_M.gguf', label: 'LFM2.5 350M' },
  qwen05: { repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf', label: 'Qwen2.5 0.5B' },
  qwen15: { repo: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF', file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf', label: 'Qwen2.5 1.5B' },
  smol360: { repo: 'HuggingFaceTB/SmolLM2-360M-Instruct-GGUF', file: 'smollm2-360m-instruct-q4_k_m.gguf', label: 'SmolLM2 360M' },
};

const gguf = { instance: null, key: null, busy: false };

async function getWllama() {
  if (!gguf.instance) {
    const mod = await import('https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/index.js');
    gguf.instance = new mod.Wllama({
      default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/src/wasm/wllama.wasm',
    });
  }
  return gguf.instance;
}

$('#load-gguf').addEventListener('click', async () => {
  if (gguf.busy) return;
  const key = $('#gguf-model').value;
  const spec = GGUF[key];
  const btn = $('#load-gguf');
  const box = $('#gguf-progress');
  const bar = $('#gguf-bar');
  const status = $('#gguf-status');
  btn.disabled = true;
  box.hidden = false;
  bar.style.width = '2%';
  gguf.busy = true;
  setStatus(status, `${t('gguf.loading')} ${spec.label} (${spec.repo})`, '');
  const started = performance.now();
  try {
    const w = await getWllama();
    if (gguf.key && gguf.key !== key) {
      try { await w.exit(); } catch { /* already gone */ }
    }
    await w.loadModelFromHF(
      { repo: spec.repo, file: spec.file },
      {
        n_ctx: 4096,
        n_threads: Math.max(1, Math.min(4, (navigator.hardwareConcurrency ?? 4) - 1)),
        progressCallback: ({ loaded, total }) => {
          if (total) bar.style.width = `${Math.round((loaded / total) * 100)}%`;
          setStatus(status, `${spec.label} · ${humanBytes(loaded)}/${humanBytes(total)}`, '');
        },
      },
    );
    gguf.key = key;
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    setStatus(status, `${t('gguf.ready')} · ${spec.label} · ${secs}s · WASM`, 'good');
    bubble($('#chat-gguf'), 'sys', `${t('gguf.ready')} (${secs}s)`);
    $('#stop-gguf').disabled = false;
  } catch (e) {
    setStatus(status, `${t('gguf.failed')}: ${String(e.message || e).slice(0, 200)}`, 'bad');
    bubble($('#chat-gguf'), 'sys', `${t('gguf.failed')}: ${e.message}`);
  } finally {
    gguf.busy = false;
    btn.disabled = false;
    box.hidden = true;
  }
});

$('#stop-gguf').addEventListener('click', async () => {
  if (!gguf.instance) return;
  try { await gguf.instance.exit(); } catch { /* ignore */ }
  gguf.instance = null;
  gguf.key = null;
  $('#stop-gguf').disabled = true;
  setStatus($('#gguf-status'), t('gguf.idle'), '');
  bubble($('#chat-gguf'), 'sys', t('gguf.stopped'));
});

$('#form-gguf').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = $('#prompt-gguf');
  const text = input.value.trim();
  if (!text) return;
  if (!gguf.instance || !gguf.key) { bubble($('#chat-gguf'), 'sys', t('gguf.loadFirst')); return; }
  input.value = '';
  bubble($('#chat-gguf'), 'user', text);
  const out = bubble($('#chat-gguf'), 'bot', '…');
  const started = performance.now();
  let acc = '';
  let chunks = 0;
  try {
    const w = gguf.instance;
    const stream = await w.createChatCompletion({
      messages: [
        { role: 'system', content: 'You are a concise local assistant. Answer in at most 120 words.' },
        { role: 'user', content: text },
      ],
      stream: true,
      max_tokens: 256,
      temperature: 0.4,
      cache_prompt: true,
    });
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content ?? '';
      if (delta) {
        acc += delta;
        chunks += 1;
        out.textContent = acc;
        $('#chat-gguf').scrollTop = $('#chat-gguf').scrollHeight;
      }
    }
    const secs = (performance.now() - started) / 1000;
    out.textContent = acc ? `${acc}\n\n— ${chunks} ${t('demo.chunks')}, ${secs.toFixed(1)}s, ${(chunks / Math.max(secs, 0.1)).toFixed(1)} tok/s` : t('demo.empty');
  } catch (e) {
    out.textContent = `${t('demo.failed')}: ${e.message}`;
  }
});

/* ---------------------------------------------------------- OPFS (large data)
   IndexedDB is fine for preferences; benchmark runs and transcripts can outgrow it.
   The Origin Private File System keeps them on the device, still without a server. */
const opfs = {
  async available() {
    return typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';
  },
  async root() {
    return navigator.storage.getDirectory();
  },
  async dir(name = 'runs', create = true) {
    const root = await this.root();
    return root.getDirectoryHandle(name, { create });
  },
  async write(name, value, dir = 'runs') {
    const handle = await this.dir(dir);
    const file = await handle.getFileHandle(name, { create: true });
    const writable = await file.createWritable();
    await writable.write(JSON.stringify(value));
    await writable.close();
    return `${dir}/${name}`;
  },
  async read(name, dir = 'runs') {
    try {
      const handle = await this.dir(dir, false);
      const file = await handle.getFileHandle(name);
      return JSON.parse(await (await file.getFile()).text());
    } catch { return null; }
  },
  async list(dir = 'runs') {
    try {
      const handle = await this.dir(dir, false);
      const out = [];
      for await (const [name, entry] of handle.entries()) {
        if (entry.kind === 'file') out.push(name);
      }
      return out.sort();
    } catch { return []; }
  },
  async clear() {
    try {
      const root = await this.root();
      for await (const [name, entry] of root.entries()) {
        if (entry.kind === 'directory') await root.removeEntry(name, { recursive: true });
      }
      return true;
    } catch { return false; }
  },
  async usage() {
    try {
      const { usage, quota } = await navigator.storage.estimate();
      return { usedMb: +(usage / 2 ** 20).toFixed(2), quotaMb: +(quota / 2 ** 20).toFixed(0) };
    } catch { return null; }
  },
};

// Small, documented test hook so scripts/verify-storage.mjs can exercise the real code paths.
// It is attached after `store` is defined (below) - referencing it earlier would be a TDZ error
// and would break the whole module.

async function saveLabRun(payload) {
  if (!(await opfs.available())) return null;
  return opfs.write(`run-${Date.now()}.json`, payload);
}

/* ------------------------------------------------- local-first user data
   IndexedDB when available, localStorage as fallback. Nothing is uploaded;
   the export/import pair is how a user moves machines without an account. */
const DB_NAME = 'localai';
const DB_STORE = 'kv';

const store = {
  db: null,
  async open() {
    if (this.db) return this.db;
    if (!('indexedDB' in window)) return null;
    this.db = await new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return this.db;
  },
  async set(key, value) {
    try {
      const db = await this.open();
      if (!db) { localStorage.setItem(`localai.${key}`, JSON.stringify(value)); return; }
      await new Promise((resolve, reject) => {
        const tx = db.transaction(DB_STORE, 'readwrite');
        tx.objectStore(DB_STORE).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
      });
    } catch { /* private mode: stay silent, the page still works */ }
  },
  async get(key, fallback = null) {
    try {
      const db = await this.open();
      if (!db) {
        const raw = localStorage.getItem(`localai.${key}`);
        return raw ? JSON.parse(raw) : fallback;
      }
      return await new Promise((resolve) => {
        const tx = db.transaction(DB_STORE, 'readonly');
        const req = tx.objectStore(DB_STORE).get(key);
        req.onsuccess = () => resolve(req.result ?? fallback);
        req.onerror = () => resolve(fallback);
      });
    } catch { return fallback; }
  },
  async all() {
    const db = await this.open();
    if (!db) {
      return Object.fromEntries(Object.keys(localStorage)
        .filter((k) => k.startsWith('localai.'))
        .map((k) => [k.slice(8), JSON.parse(localStorage.getItem(k))]));
    }
    return new Promise((resolve) => {
      const out = {};
      const tx = db.transaction(DB_STORE, 'readonly');
      const req = tx.objectStore(DB_STORE).openCursor();
      req.onsuccess = () => {
        const cur = req.result;
        if (cur) { out[cur.key] = cur.value; cur.continue(); } else resolve(out);
      };
      req.onerror = () => resolve(out);
    });
  },
  async clear() {
    const db = await this.open();
    if (db) {
      await new Promise((resolve) => {
        const tx = db.transaction(DB_STORE, 'readwrite');
        tx.objectStore(DB_STORE).clear();
        tx.oncomplete = resolve;
      });
    }
    for (const k of Object.keys(localStorage)) if (k.startsWith('localai.')) localStorage.removeItem(k);
  },
};

// attach the test hook now that both layers exist
window.localaiStorage = { opfs, store };

async function restorePrefs() {
  const saved = await store.get('prefs', null);
  if (!saved) return;
  if (saved.tier && $('#tier')) $('#tier').value = saved.tier;
  if (saved.gguf && $('#gguf-model')) $('#gguf-model').value = saved.gguf;
  if (saved.base && $('#base')) $('#base').value = saved.base;
  if (saved.lab) state.lab = saved.lab;
}

async function rememberPrefs() {
  await store.set('prefs', {
    tier: $('#tier')?.value,
    gguf: $('#gguf-model')?.value,
    base: $('#base')?.value,
    savedAt: new Date().toISOString(),
  });
}

function dataStatus(text, kind = '') {
  const el = $('#data-status');
  el.textContent = text;
  el.className = `status ${kind}`;
}

$('#export-data').addEventListener('click', async () => {
  const payload = { kind: 'localai-export', version: 1, createdAt: new Date().toISOString(), data: await store.all() };
  if (await opfs.available()) {
    payload.opfs = {};
    for (const name of await opfs.list()) payload.opfs[name] = await opfs.read(name);
    const usage = await opfs.usage();
    if (usage) payload.usage = usage;
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'localai-data.json';
  a.click();
  URL.revokeObjectURL(a.href);
  dataStatus(t('data.exported'), 'good');
});

$('#import-data').addEventListener('click', () => $('#import-file').click());

$('#import-file').addEventListener('change', async (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    if (parsed?.kind !== 'localai-export' || typeof parsed.data !== 'object') throw new Error('wrong file');
    for (const [key, value] of Object.entries(parsed.data)) await store.set(key, value);
    await restorePrefs();
    dataStatus(t('data.imported'), 'good');
  } catch (e) {
    dataStatus(`${t('data.importFailed')}: ${e.message}`, 'bad');
  } finally {
    ev.target.value = '';
  }
});

$('#clear-data').addEventListener('click', async () => {
  await store.clear();
  await opfs.clear();
  dataStatus(t('data.cleared'), 'good');
});

for (const id of ['#tier', '#gguf-model', '#base']) {
  $(id)?.addEventListener('change', rememberPrefs);
}

/* --------------------------------------------------------------- PWA shell
   The service worker makes the page open offline (the shell is cached; models are
   not). The install button only appears when the browser offers a prompt. */
let installPrompt = null;

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  const button = $('#install-app');
  if (button) button.hidden = false;
});

window.addEventListener("appinstalled", () => {
  installPrompt = null;
  const button = $('#install-app');
  if (button) button.hidden = true;
  dataStatus(t('pwa.installed'), 'good');
});

$('#install-app')?.addEventListener('click', async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  try { await installPrompt.userChoice; } catch { /* user dismissed */ }
  installPrompt = null;
  $('#install-app').hidden = true;
});

if ('serviceWorker' in navigator && (location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname))) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => { /* offline support is optional */ });
  });
}

/* ------------------------------------------------------------------ boot */
restorePrefs();
detectHardware();
