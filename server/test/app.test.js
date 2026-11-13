// The real Express app on a random port, talking to a fake Python service
// (a tiny HTTP server that streams scripted SSE) and an in-memory store.
import assert from 'node:assert/strict';
import http from 'node:http';
import { after, before, beforeEach, test } from 'node:test';
import { createApp } from '../src/app.js';
import { SseParser, sse } from '../src/sse.js';
import { MemoryHistoryStore } from '../src/store.js';

const SOURCES = [1, 2, 3].map((n) => ({ n, url: `https://ex.com/${n}`, title: `Page ${n}`, chunk_id: `d${n}:0`, text: `passage ${n}` }));
const DONE = { status: 'answered', text: 'Fact [1]. More [3].', cited: [1, 3], coverage: 1, uncited: [], invalid: [], model: 'fake', input_tokens: 10, output_tokens: 5, timings_ms: { total: 12 } };

let script; // what the fake Python service does on /answer, set per test
let lastQuery;
let upstreamClosed;
const fake = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  lastQuery = url.searchParams.get('q');
  if (url.pathname === '/health') return res.end(JSON.stringify({ status: 'ok', chunks: 3 }));
  if (url.pathname === '/search') {
    res.setHeader('content-type', 'application/json');
    return res.end(JSON.stringify({ query: lastQuery, results: SOURCES }));
  }
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  req.on('close', () => (upstreamClosed = true));
  script(res);
});

let store;
let backend;
let base;
const silent = { error() {} };

before(async () => {
  await new Promise((r) => fake.listen(0, '127.0.0.1', r));
  const retrievalUrl = `http://127.0.0.1:${fake.address().port}`;
  store = new MemoryHistoryStore();
  const app = createApp({ store, retrievalUrl, log: silent });
  backend = app.listen(0, '127.0.0.1');
  await new Promise((r) => backend.once('listening', r));
  base = `http://127.0.0.1:${backend.address().port}`;
});

after(() => {
  backend.closeAllConnections();
  backend.close();
  fake.closeAllConnections();
  fake.close();
});

beforeEach(() => {
  store.docs.clear();
  upstreamClosed = false;
  script = (res) => {
    res.write(sse('sources', SOURCES));
    for (const piece of ['Fact [1]. ', 'More [3].']) res.write(sse('token', { text: piece }));
    res.end(sse('done', DONE));
  };
});

async function ask(q) {
  const r = await fetch(`${base}/api/ask?q=${encodeURIComponent(q)}`);
  if (!r.headers.get('content-type')?.startsWith('text/event-stream')) return { status: r.status, body: await r.json() };
  return { status: r.status, events: new SseParser().push(await r.text()) };
}

test('ask relays the stream in order, then saves and reports the history id', async () => {
  const { status, events } = await ask('  what facts?  ');
  assert.equal(status, 200);
  assert.deepEqual(events.map((e) => e.event), ['sources', 'token', 'token', 'done', 'saved']);
  assert.equal(lastQuery, 'what facts?'); // trimmed before proxying

  const doc = await store.get(events.at(-1).data.id);
  assert.equal(doc.question, 'what facts?');
  assert.equal(doc.status, 'answered');
  assert.equal(doc.answer, DONE.text);
  assert.deepEqual(doc.cited, [1, 3]);
  assert.equal(doc.sources.length, 3);
  assert.equal(doc.sources[0].text, undefined); // passage text is not duplicated into history
});

test('an error event from the Python service is relayed and saved as an error', async () => {
  script = (res) => {
    res.write(sse('sources', SOURCES));
    res.end(sse('error', { message: 'Gemini API error 503' }));
  };
  const { events } = await ask('q');
  assert.deepEqual(events.map((e) => e.event), ['sources', 'error', 'saved']);
  const doc = await store.get(events.at(-1).data.id);
  assert.equal(doc.status, 'error');
  assert.equal(doc.error, 'Gemini API error 503');
});

test('a stream that ends without done gets an error event', async () => {
  script = (res) => res.end(sse('sources', SOURCES));
  const { events } = await ask('q');
  assert.deepEqual(events.map((e) => e.event), ['sources', 'error', 'saved']);
});

test('a failing history store does not lose the answer', async () => {
  const save = store.save;
  store.save = async () => {
    throw new Error('mongo down');
  };
  try {
    const { events } = await ask('q');
    assert.ok(events.some((e) => e.event === 'done'));
    assert.deepEqual(events.at(-1).data, { id: null, error: 'answer not saved to history' });
  } finally {
    store.save = save;
  }
});

test('client disconnect cancels the upstream request and saves nothing', async () => {
  script = (res) => res.write(sse('sources', SOURCES)); // never finishes, like a slow LLM
  const ctrl = new AbortController();
  const r = await fetch(`${base}/api/ask?q=slow`, { signal: ctrl.signal });
  const reader = r.body.getReader();
  await reader.read(); // the sources event arrived
  ctrl.abort();
  for (let i = 0; i < 50 && !upstreamClosed; i++) await new Promise((res) => setTimeout(res, 20));
  assert.ok(upstreamClosed, 'the fake Python service saw the connection close');
  assert.equal(store.docs.size, 0);
});

test('bad questions are rejected before anything is proxied', async () => {
  lastQuery = undefined;
  for (const q of ['', '   ', 'x'.repeat(501)]) {
    const { status, body } = await ask(q);
    assert.equal(status, 400);
    assert.match(body.error, /q/);
  }
  const twice = await fetch(`${base}/api/ask?q=a&q=b`);
  assert.equal(twice.status, 400);
  assert.equal(lastQuery, undefined);
});

test('python service down: 502 with a JSON error', async () => {
  const app = createApp({ store, retrievalUrl: 'http://127.0.0.1:1', log: silent });
  const srv = app.listen(0, '127.0.0.1');
  await new Promise((r) => srv.once('listening', r));
  try {
    const r = await fetch(`http://127.0.0.1:${srv.address().port}/api/ask?q=x`);
    assert.equal(r.status, 502);
    assert.match((await r.json()).error, /unreachable/);
  } finally {
    srv.close();
  }
});

test('history: newest first, fetch one, delete, 404s', async () => {
  const first = (await ask('first')).events.at(-1).data.id;
  await new Promise((r) => setTimeout(r, 5)); // distinct createdAt
  await ask('second');
  const list = await (await fetch(`${base}/api/history?limit=10`)).json();
  assert.deepEqual(list.map((h) => h.question), ['second', 'first']);
  assert.equal(list[0].answer, undefined); // list rows stay small

  assert.equal((await (await fetch(`${base}/api/history/${first}`)).json()).question, 'first');
  assert.equal((await fetch(`${base}/api/history/${first}`, { method: 'DELETE' })).status, 204);
  assert.equal((await fetch(`${base}/api/history/${first}`)).status, 404);
  assert.equal((await fetch(`${base}/api/history/${first}`, { method: 'DELETE' })).status, 404);
  assert.equal((await fetch(`${base}/api/history?limit=0`)).status, 400);
});

test('search passes through; health reports both dependencies; unknown api path is JSON 404', async () => {
  const s = await (await fetch(`${base}/api/search?q=csv`)).json();
  assert.equal(s.results.length, 3);
  const h = await (await fetch(`${base}/api/health`)).json();
  assert.deepEqual([h.status, h.history, h.retrieval.status], ['ok', 'ok', 'ok']);
  const r = await fetch(`${base}/api/nope`);
  assert.equal(r.status, 404);
  assert.equal((await r.json()).error, 'no such endpoint');
});
