// Past questions from /api/history (MongoDB). Reloads when `refreshKey` changes,
// which App bumps each time a new answer is saved.

import { useEffect, useState } from 'react';

export function History({ refreshKey, activeId, onOpen, onDeleted, fetchImpl = globalThis.fetch }) {
  const [items, setItems] = useState(null); // null = loading
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false; // a slow older request must not overwrite a newer one
    fetchImpl('/api/history?limit=30')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((rows) => !cancelled && (setItems(rows), setError(null)))
      .catch((e) => !cancelled && setError(`History unavailable (${e.message})`));
    return () => {
      cancelled = true;
    };
  }, [refreshKey, fetchImpl]);

  async function remove(id) {
    const r = await fetchImpl(`/api/history/${id}`, { method: 'DELETE' });
    if (r.ok || r.status === 404) {
      setItems((rows) => rows.filter((h) => h.id !== id));
      onDeleted?.(id);
    }
  }

  return (
    <nav className="history" aria-label="History">
      <h2>History</h2>
      {error && <p className="muted small">{error}</p>}
      {items?.length === 0 && <p className="muted small">Questions you ask appear here.</p>}
      <ul>
        {items?.map((h) => (
          <li key={h.id} className={h.id === activeId ? 'active' : ''}>
            <button type="button" className="history-item" onClick={() => onOpen(h.id)} title={`${h.question} (${h.status})`} aria-label={h.question}>
              <span className={`dot ${h.status}`} aria-hidden="true" />
              <span className="history-q">{h.question}</span>
            </button>
            <button type="button" className="icon-button" onClick={() => remove(h.id)} aria-label={`Delete "${h.question}"`}>
              ×
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
