// useAsk: ask a question and follow the answer stream from /api/ask.
//
// Events from the server, in order:
//   sources -> token* -> done | error -> saved
// `saved` is always last, so that's where the stream is closed.
//
// The EventSource trap: when a stream ends, EventSource assumes the
// connection dropped and RECONNECTS, which would ask the same question again
// (a second LLM call and a duplicate history entry). So we close() it
// ourselves on `saved`, and on any connection failure.
//
// Two different things arrive as "error":
//   - our server's `event: error` (a MessageEvent with data): the LLM failed;
//     the stream continues with `saved`
//   - the browser's own connection error (no data): Express is down, or it
//     answered 400/502 with JSON instead of a stream

import { useCallback, useEffect, useRef, useState } from 'react';

export const IDLE = {
  phase: 'idle', // idle | streaming | done | error | stopped
  question: '',
  sources: [],
  text: '',
  done: null, // the server's `done` payload: status, cited, coverage, ...
  error: null,
  savedId: null,
};

export function useAsk({ EventSourceImpl = globalThis.EventSource, onSaved } = {}) {
  const [state, setState] = useState(IDLE);
  const esRef = useRef(null);
  const onSavedRef = useRef(onSaved);
  onSavedRef.current = onSaved;

  const close = () => {
    esRef.current?.close();
    esRef.current = null;
  };

  const ask = useCallback(
    (question) => {
      close(); // a new question replaces any answer still streaming
      setState({ ...IDLE, phase: 'streaming', question });
      const es = new EventSourceImpl(`/api/ask?q=${encodeURIComponent(question)}`);
      esRef.current = es;
      const data = (e) => JSON.parse(e.data);

      es.addEventListener('sources', (e) => setState((s) => ({ ...s, sources: data(e) })));
      es.addEventListener('token', (e) => setState((s) => ({ ...s, text: s.text + data(e).text })));
      es.addEventListener('done', (e) => setState((s) => ({ ...s, done: data(e) })));
      es.addEventListener('saved', (e) => {
        close();
        const { id } = data(e);
        setState((s) => ({ ...s, phase: s.done ? 'done' : 'error', savedId: id }));
        if (id) onSavedRef.current?.(id);
      });
      es.addEventListener('error', (e) => {
        if (e.data) {
          setState((s) => ({ ...s, error: data(e).message })); // our server's error event: `saved` follows
          return;
        }
        close(); // connection-level failure: stop EventSource from retrying forever
        setState((s) =>
          s.phase === 'streaming'
            ? { ...s, phase: 'error', error: s.error ?? 'Could not reach the server. Is the backend running?' }
            : s,
        );
      });
    },
    [EventSourceImpl],
  );

  const stop = useCallback(() => {
    close(); // the server sees the disconnect and stops the LLM
    setState((s) => (s.phase === 'streaming' ? { ...s, phase: 'stopped' } : s));
  }, []);

  useEffect(() => close, []); // leaving the page closes the stream

  return { state, ask, stop };
}
