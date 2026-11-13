import { useState } from 'react';
import { Answer } from './Answer.jsx';
import { History } from './History.jsx';
import { SourcePanel } from './SourcePanel.jsx';
import { useAsk } from './useAsk.js';

// Questions the current corpus (Python docs + martinfowler.com) can answer
const EXAMPLES = [
  'How do I read a CSV file in Python?',
  'What is a feature toggle?',
  'How do I parse command-line arguments?',
];

/** A stored history document, in the same shape as a live answer. */
function viewFromHistory(doc) {
  return {
    phase: doc.status === 'error' ? 'error' : 'done',
    question: doc.question,
    sources: doc.sources,
    text: doc.answer,
    done: doc.status === 'error' ? null : { ...doc, text: doc.answer },
    error: doc.error,
  };
}

export default function App({ EventSourceImpl, fetchImpl = globalThis.fetch }) {
  const [input, setInput] = useState('');
  const [historyDoc, setHistoryDoc] = useState(null); // shown instead of the live answer when set
  const [activeSource, setActiveSource] = useState(null);
  const [historyKey, setHistoryKey] = useState(0);
  const { state, ask, stop } = useAsk({ EventSourceImpl, onSaved: () => setHistoryKey((k) => k + 1) });

  const view = historyDoc ? viewFromHistory(historyDoc) : state;
  const source = view.sources.find((s) => s.n === activeSource) ?? null;

  function submit(question) {
    const q = question.trim();
    if (!q) return;
    setInput(q);
    setHistoryDoc(null);
    setActiveSource(null);
    ask(q);
  }

  async function openHistory(id) {
    const r = await fetchImpl(`/api/history/${id}`);
    if (!r.ok) return;
    if (state.phase === 'streaming') stop();
    setHistoryDoc(await r.json());
    setActiveSource(null);
  }

  return (
    <div className="layout">
      <History
        refreshKey={historyKey}
        activeId={historyDoc?.id ?? state.savedId}
        onOpen={openHistory}
        onDeleted={(id) => historyDoc?.id === id && setHistoryDoc(null)}
        fetchImpl={fetchImpl}
      />

      <main>
        <header className="brand">
          <h1>Mini Perplexity</h1>
          <p className="muted">Answers from my own crawler and index. Every sentence cites its source.</p>
        </header>

        <form
          className="ask"
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
        >
          <input
            type="search"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the Python docs or software design…"
            aria-label="Question"
            maxLength={500}
            autoFocus
          />
          {state.phase === 'streaming' && !historyDoc ? (
            <button type="button" onClick={stop}>Stop</button>
          ) : (
            <button type="submit" disabled={!input.trim()}>Ask</button>
          )}
        </form>

        {view.phase === 'idle' ? (
          <div className="examples">
            <p className="muted">Try:</p>
            {EXAMPLES.map((q) => (
              <button key={q} type="button" className="example" onClick={() => submit(q)}>{q}</button>
            ))}
          </div>
        ) : (
          <div className="result">
            <Answer view={view} activeSource={activeSource} onCite={setActiveSource} />
            <SourcePanel source={source} onClose={() => setActiveSource(null)} />
          </div>
        )}
      </main>
    </div>
  );
}
