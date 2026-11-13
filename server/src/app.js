// Express app: the public backend the React UI talks to.
//
//   GET    /api/ask?q=...        answer stream (SSE), relayed live from the Python
//                                service; saved to history when it ends, then a
//                                final `saved` event carries the history id
//   GET    /api/search?q=&k=     passages only (JSON), passed through
//   GET    /api/history?limit=   newest questions first (id, question, status, createdAt)
//   GET    /api/history/:id      one full history document
//   DELETE /api/history/:id      remove one
//   GET    /api/health           Express, history store and Python service status
//
// createApp() takes its dependencies (store, retrieval URL) as arguments so
// tests can hand it an in-memory store and a fake Python service.

import express from 'express';
import { SseParser, sse } from './sse.js';
import { historyDoc } from './store.js';

const MAX_QUESTION_CHARS = 500; // same limit as the Python service: fail here, before any proxying

export function createApp({ store, retrievalUrl, log = console }) {
  const app = express();
  app.disable('x-powered-by');

  app.get('/api/health', async (_req, res) => {
    const [history, retrieval] = await Promise.all([
      store.ping().then(() => 'ok', (e) => `error: ${e.message}`),
      fetch(`${retrievalUrl}/health`, { signal: AbortSignal.timeout(3000) })
        .then((r) => r.json())
        .catch((e) => ({ status: `unreachable: ${e.message}` })),
    ]);
    const ok = history === 'ok' && retrieval.status === 'ok';
    res.status(ok ? 200 : 503).json({ status: ok ? 'ok' : 'degraded', history, retrieval });
  });

  app.get('/api/search', async (req, res) => {
    const q = question(req, res);
    if (q === null) return;
    const k = req.query.k ?? '10';
    try {
      const r = await fetch(`${retrievalUrl}/search?q=${encodeURIComponent(q)}&k=${encodeURIComponent(k)}`);
      res.status(r.status).json(await r.json()); // pass the Python service's answer (or its 422) through
    } catch (e) {
      res.status(502).json({ error: `retrieval service unreachable: ${e.message}` });
    }
  });

  app.get('/api/ask', async (req, res) => {
    const q = question(req, res);
    if (q === null) return;

    // If the browser goes away, cancel the upstream request: FastAPI then stops
    // the LLM, so nobody pays for tokens no one will read.
    const upstreamAbort = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) upstreamAbort.abort();
    });

    let upstream;
    try {
      upstream = await fetch(`${retrievalUrl}/answer?q=${encodeURIComponent(q)}`, { signal: upstreamAbort.signal });
    } catch (e) {
      return res.status(502).json({ error: `retrieval service unreachable: ${e.message}` });
    }
    if (!upstream.ok) {
      return res.status(502).json({ error: `retrieval service returned ${upstream.status}`, detail: await upstream.text() });
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no', // tell proxies like nginx not to buffer the stream
    });
    res.flushHeaders();

    const parser = new SseParser();
    const decoder = new TextDecoder(); // stream: true below keeps a multi-byte character split across chunks intact
    let sources = null;
    let done = null;
    let error = null;
    try {
      for await (const chunk of upstream.body) {
        for (const ev of parser.push(decoder.decode(chunk, { stream: true }))) {
          if (ev.event === 'sources') sources = ev.data;
          else if (ev.event === 'done') done = ev.data;
          else if (ev.event === 'error') error = ev.data?.message ?? 'unknown error';
          res.write(ev.raw); // forward byte for byte, as soon as it is complete
        }
      }
    } catch (e) {
      if (upstreamAbort.signal.aborted) return; // the user left: nothing to deliver, nothing worth saving
      error = `retrieval stream broke: ${e.message}`;
      res.write(sse('error', { message: error }));
    }
    if (!done && !error) {
      error = 'retrieval service closed the stream without an answer';
      res.write(sse('error', { message: error }));
    }

    // Save after the answer is delivered: a slow or down database must never delay or block the answer.
    try {
      const id = await store.save(historyDoc(q, sources, done, error));
      res.write(sse('saved', { id }));
    } catch (e) {
      log.error(`history save failed: ${e.message}`);
      res.write(sse('saved', { id: null, error: 'answer not saved to history' }));
    }
    res.end();
  });

  app.get('/api/history', async (req, res) => {
    const limit = Number(req.query.limit ?? 20);
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
      return res.status(400).json({ error: 'limit must be an integer from 1 to 100' });
    }
    res.json(await store.list(limit));
  });

  app.get('/api/history/:id', async (req, res) => {
    const doc = await store.get(req.params.id);
    if (!doc) return res.status(404).json({ error: 'not found' });
    res.json(doc);
  });

  app.delete('/api/history/:id', async (req, res) => {
    const removed = await store.remove(req.params.id);
    res.status(removed ? 204 : 404).end();
  });

  app.use('/api', (_req, res) => res.status(404).json({ error: 'no such endpoint' }));

  // Express 5 sends errors thrown in async handlers here; answer in JSON, keep details in the log
  app.use((err, _req, res, _next) => {
    log.error(err);
    if (res.headersSent) return res.end();
    res.status(500).json({ error: 'internal error' });
  });

  return app;
}

/** The validated question, or null after sending a 400. */
function question(req, res) {
  const q = req.query.q;
  // ?q=a&q=b arrives as an array: reject anything that isn't one string
  if (typeof q !== 'string' || !q.trim()) {
    res.status(400).json({ error: 'q (the question) is required' });
    return null;
  }
  if (q.length > MAX_QUESTION_CHARS) {
    res.status(400).json({ error: `q is longer than ${MAX_QUESTION_CHARS} characters` });
    return null;
  }
  return q.trim();
}
