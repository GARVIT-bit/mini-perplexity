import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SseParser, sse } from '../src/sse.js';

test('events split across chunks are reassembled', () => {
  const p = new SseParser();
  const full = sse('token', { text: 'hello' }) + sse('done', { status: 'answered' });
  const out = [];
  for (const ch of full) out.push(...p.push(ch)); // worst case: one character per chunk
  assert.deepEqual(out.map((e) => [e.event, e.data]), [
    ['token', { text: 'hello' }],
    ['done', { status: 'answered' }],
  ]);
  assert.equal(out.map((e) => e.raw).join(''), full); // forwarded bytes == received bytes
});

test('an unfinished event waits for the rest', () => {
  const p = new SseParser();
  assert.deepEqual(p.push('event: token\ndata: {"text"'), []);
  assert.equal(p.push(': "x"}\n\n')[0].data.text, 'x');
});

test('CRLF line endings, comments and missing event names', () => {
  const p = new SseParser();
  const [a, b] = p.push(': keep-alive\r\nevent: done\r\ndata: {"ok":true}\r\n\r\ndata: plain text\n\n');
  assert.equal(a.event, 'done');
  assert.deepEqual(a.data, { ok: true });
  assert.equal(b.event, 'message'); // the SSE default
  assert.equal(b.data, 'plain text'); // not JSON: kept as a string
});

test('blank lines inside answer text cannot split an event', () => {
  const out = new SseParser().push(sse('token', { text: 'a\n\nevent: done' }));
  assert.equal(out.length, 1);
  assert.equal(out[0].data.text, 'a\n\nevent: done');
});
