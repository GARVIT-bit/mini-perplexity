// The passage behind a citation chip: exactly the text the model was given as [n].
// This is what lets a reader check a claim instead of trusting it.

import { safeHref } from './sources.js';

export function SourcePanel({ source, onClose }) {
  if (!source) return null;
  const href = safeHref(source.url);
  return (
    <aside className="source-panel" aria-label={`Source ${source.n}`}>
      <header>
        <span className="chip active static">{source.n}</span>
        <div className="source-panel-title">
          <strong>{source.title || '(untitled)'}</strong>
          {href ? (
            <a href={href} target="_blank" rel="noopener noreferrer">{source.url}</a>
          ) : (
            <span className="muted">{source.url}</span>
          )}
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close source">×</button>
      </header>
      {source.text ? (
        <blockquote>{source.text}</blockquote>
      ) : (
        // history keeps only url/title/chunk_id; the text lives in the index
        <p className="muted">Passage text isn't stored in history. Open the page to read it.</p>
      )}
    </aside>
  );
}
