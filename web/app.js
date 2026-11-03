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
    $('#panel-browser').hidden = btn.dataset.tab !== 'browser';
    $('#panel-local').hidden = btn.dataset.tab !== 'local';
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

/* ------------------------------------------------------------------ boot */
detectHardware();
