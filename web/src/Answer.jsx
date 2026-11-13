// The answer view: status, answer text with citation chips, the citation
// check, and source cards. Used for a live answer and for one opened from history.

import { parseBlocks, parseInline } from './answerText.js';
import { groupSources, hostOf } from './sources.js';

export function Answer({ view, activeSource, onCite }) {
  const { phase, question, sources, text, done, error } = view;
  const numbers = new Set(sources.map((s) => s.n));
  const status = done?.status;

  return (
    <article className="answer" aria-busy={phase === 'streaming'}>
      <h2 className="question">{question}</h2>
      <Progress phase={phase} hasSources={sources.length > 0} hasText={text.length > 0} />

      {status === 'not_found' ? (
        <p className="notice">
          The indexed pages don't answer this. The engine is told to say so rather than guess, and it did.
        </p>
      ) : (
        <div className="answer-body">
          {parseBlocks(text).map((b, i) => (
            <Block key={i} block={b} numbers={numbers} activeSource={activeSource} onCite={onCite} />
          ))}
          {phase === 'streaming' && text && <span className="caret" aria-hidden="true" />}
        </div>
      )}

      {status === 'refused' && <p className="notice warn">The model declined to answer this question.</p>}
      {status === 'truncated' && (
        <p className="notice warn">The answer hit the length limit and was cut off; its last sentences may be missing citations.</p>
      )}
      {phase === 'stopped' && <p className="notice">Stopped.</p>}
      {error && <p className="notice error" role="alert">{error}</p>}

      {done && status !== 'not_found' && <CitationCheck done={done} />}
      {sources.length > 0 && (
        <Sources sources={sources} cited={done?.cited ?? []} activeSource={activeSource} onCite={onCite} />
      )}
    </article>
  );
}

function Progress({ phase, hasSources, hasText }) {
  if (phase !== 'streaming') return null;
  const label = !hasSources ? 'Searching the index…' : !hasText ? 'Reading the top passages…' : 'Writing…';
  return (
    <p className="progress" role="status">
      <span className="spinner" aria-hidden="true" /> {label}
    </p>
  );
}

function Block({ block, ...chip }) {
  switch (block.type) {
    case 'h':
      return <h3><Inline text={block.text} {...chip} /></h3>;
    case 'code':
      return <pre><code>{block.text}</code></pre>;
    case 'ul':
      return (
        <ul>
          {block.items.map((item, i) => (
            <li key={i}><Inline text={item} {...chip} /></li>
          ))}
        </ul>
      );
    default:
      return <p><Inline text={block.text} {...chip} /></p>;
  }
}

function Inline({ text, numbers, activeSource, onCite }) {
  return parseInline(text).map((piece, i) => {
    if (piece.type === 'code') return <code key={i}>{piece.value}</code>;
    if (piece.type === 'bold') return <strong key={i}>{piece.value}</strong>;
    if (piece.type === 'text') return piece.value;
    return (
      <span key={i} className="cites">
        {piece.nums.map((n) => (
          <CiteChip key={n} n={n} valid={numbers.has(n)} active={activeSource === n} onCite={onCite} />
        ))}
      </span>
    );
  });
}

export function CiteChip({ n, valid, active, onCite }) {
  if (!valid) {
    // the model cited a passage it was never given: show it, but as a problem
    return <span className="chip invalid" title={`There is no passage ${n}: the model made this citation up`}>{n}</span>;
  }
  return (
    <button
      type="button"
      className={`chip${active ? ' active' : ''}`}
      onClick={() => onCite(active ? null : n)}
      aria-label={`Show source ${n}`}
      aria-pressed={active}
    >
      {n}
    </button>
  );
}

function CitationCheck({ done }) {
  const pct = Math.round((done.coverage ?? 0) * 100);
  const t = done.timings_ms ?? {};
  const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
  return (
    <details className="check">
      <summary>
        <span className={pct === 100 ? 'ok' : 'warn'}>{pct}% of claims cited</span>
        {done.invalid?.length > 0 && <span className="bad"> · {done.invalid.length} made-up citation(s)</span>}
        {done.model && <span className="muted"> · {done.model}</span>}
        {t.total != null && <span className="muted"> · {secs(t.total)}</span>}
      </summary>
      {done.uncited?.length > 0 && (
        <>
          <p>Sentences without a citation:</p>
          <ul>{done.uncited.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </>
      )}
      {done.invalid?.length > 0 && <p>Citations to passages that don't exist: {done.invalid.join(', ')}</p>}
      <p className="muted">
        {t.retrieval != null && <>search {Math.round(t.retrieval)} ms</>}
        {t.first_token != null && <> · first word {secs(t.first_token)}</>}
        {done.input_tokens > 0 && <> · tokens {done.input_tokens} in / {done.output_tokens} out</>}
      </p>
    </details>
  );
}

function Sources({ sources, cited, activeSource, onCite }) {
  const groups = groupSources(sources, cited);
  return (
    <section className="sources" aria-label="Sources">
      <h3>Sources</h3>
      <ol className="source-cards">
        {groups.map((g) => (
          <li key={g.url} className={`source-card${g.cited ? ' cited' : ''}`}>
            <div className="source-title">{g.title || g.url}</div>
            <div className="source-meta">
              <span className="muted">{hostOf(g.url)}</span>
              <span className="cites">
                {g.numbers.map((n) => (
                  <CiteChip key={n} n={n} valid active={activeSource === n} onCite={onCite} />
                ))}
              </span>
            </div>
            {!g.cited && cited.length > 0 && <div className="muted small">retrieved, not cited</div>}
          </li>
        ))}
      </ol>
    </section>
  );
}
