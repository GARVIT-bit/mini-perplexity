// The whole UI in jsdom, with a fake EventSource the test drives event by event
// and a fake fetch for /api/history.
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test } from 'vitest';
import App from './App.jsx';

class FakeEventSource {
  static last = null;
  constructor(url) {
    this.url = url;
    this.closed = false;
    this.listeners = {};
    FakeEventSource.last = this;
  }
  addEventListener(name, fn) {
    (this.listeners[name] ??= []).push(fn);
  }
  close() {
    this.closed = true;
  }
  /** Deliver one event; data undefined = a connection-level error with no payload */
  emit(name, data) {
    if (this.closed) return;
    const event = data === undefined ? { type: name } : { type: name, data: JSON.stringify(data) };
    act(() => (this.listeners[name] ?? []).forEach((fn) => fn(event)));
  }
}

const SOURCES = [
  { n: 1, url: 'https://docs.python.org/3/library/csv.html', title: 'csv — CSV File Reading', chunk_id: 'c:2', text: 'The csv module’s reader and writer objects read and write sequences.' },
  { n: 2, url: 'https://docs.python.org/3/library/csv.html', title: 'csv — CSV File Reading', chunk_id: 'c:1', text: 'other passage' },
  { n: 3, url: 'https://martinfowler.com/bliki/X.html', title: 'X', chunk_id: 'm:0', text: 'unrelated' },
];
const DONE = { status: 'answered', text: '', cited: [1], coverage: 1, uncited: [], invalid: [], model: 'gemini-3.1-flash-lite', timings_ms: { retrieval: 23, first_token: 3861, total: 4337 } };

let history;
let fetchCalls;
const fakeFetch = async (url, opts = {}) => {
  fetchCalls.push([url, opts.method ?? 'GET']);
  if (opts.method === 'DELETE') {
    history = history.filter((h) => !url.endsWith(h.id));
    return { ok: true, status: 204 };
  }
  if (url.startsWith('/api/history?')) return { ok: true, json: async () => history };
  const doc = history.find((h) => url.endsWith(h.id));
  return doc ? { ok: true, json: async () => doc } : { ok: false, status: 404 };
};

beforeEach(() => {
  history = [];
  fetchCalls = [];
});
afterEach(cleanup);

function setup() {
  render(<App EventSourceImpl={FakeEventSource} fetchImpl={fakeFetch} />);
  fireEvent.change(screen.getByLabelText('Question'), { target: { value: '  how do I read a csv?  ' } });
  fireEvent.submit(screen.getByLabelText('Question').closest('form'));
  return FakeEventSource.last;
}

function answerWith(es, text, done = DONE) {
  es.emit('sources', SOURCES);
  for (const piece of text.match(/\S+\s*/g)) es.emit('token', { text: piece });
  es.emit('done', { ...done, text });
  es.emit('saved', { id: 'h1' });
}

test('asking opens one stream for the trimmed question', () => {
  const es = setup();
  expect(es.url).toBe('/api/ask?q=how%20do%20I%20read%20a%20csv%3F');
  expect(screen.getByRole('status').textContent).toMatch(/Searching/);
});

test('tokens appear as they stream, citations become chips', () => {
  const es = setup();
  es.emit('sources', SOURCES);
  es.emit('token', { text: 'Use a reader ' });
  expect(screen.getByText(/Use a reader/)).toBeTruthy();
  expect(screen.getByRole('status').textContent).toMatch(/Writing/);
  es.emit('token', { text: '[1].' });
  expect(screen.getAllByRole('button', { name: 'Show source 1' }).length).toBeGreaterThan(0);
});

test('the stream is closed on `saved`, so EventSource never re-asks', () => {
  const es = setup();
  answerWith(es, 'Use a reader [1].');
  expect(es.closed).toBe(true);
  expect(screen.queryByRole('status')).toBe(null); // no spinner once done
  expect(screen.getByText(/100% of claims cited/)).toBeTruthy();
});

test('clicking a chip shows exactly the passage the model was given', () => {
  const es = setup();
  answerWith(es, 'Use a reader [1].');
  fireEvent.click(screen.getAllByRole('button', { name: 'Show source 1' })[0]);
  const panel = screen.getByRole('complementary', { name: 'Source 1' });
  expect(panel.textContent).toContain('reader and writer objects read and write sequences');
  expect(panel.querySelector('a').getAttribute('rel')).toBe('noopener noreferrer');
  fireEvent.click(screen.getByRole('button', { name: 'Close source' }));
  expect(screen.queryByRole('complementary', { name: 'Source 1' })).toBe(null);
});

test('sources are grouped by page: two csv passages make one card', () => {
  const es = setup();
  answerWith(es, 'Use a reader [1].');
  const cards = screen.getByRole('region', { name: 'Sources' }).querySelectorAll('.source-card');
  expect(cards.length).toBe(2);
  expect(cards[0].textContent).toContain('csv');
  expect(cards[1].textContent).toContain('retrieved, not cited');
});

test('a made-up citation is shown as a problem, not a link', () => {
  const es = setup();
  answerWith(es, 'Claim [9].', { ...DONE, cited: [], coverage: 0, invalid: [9] });
  expect(screen.queryByRole('button', { name: 'Show source 9' })).toBe(null);
  expect(screen.getByTitle(/no passage 9/)).toBeTruthy();
  expect(screen.getByText(/1 made-up citation/)).toBeTruthy();
});

test('model output is text, never HTML', () => {
  const es = setup();
  answerWith(es, '<img src=x onerror="alert(1)"> [1].');
  expect(document.querySelector('img')).toBe(null);
  expect(screen.getByText(/<img src=x/)).toBeTruthy();
});

test('NOT_FOUND shows a clear message instead of an answer', () => {
  const es = setup();
  answerWith(es, 'NOT_FOUND', { ...DONE, status: 'not_found', cited: [] });
  expect(screen.getByText(/indexed pages don't answer this/)).toBeTruthy();
  expect(screen.queryByText('NOT_FOUND')).toBe(null);
});

test('server error event: message shown, stream still closes on saved', () => {
  const es = setup();
  es.emit('sources', SOURCES);
  es.emit('error', { message: 'Gemini API error 503: high demand' });
  expect(es.closed).toBe(false); // `saved` is still coming
  es.emit('saved', { id: 'h2' });
  expect(es.closed).toBe(true);
  expect(screen.getByRole('alert').textContent).toContain('503');
});

test('connection failure (backend down) stops EventSource from retrying', () => {
  const es = setup();
  es.emit('error'); // no data: the browser's own connection error
  expect(es.closed).toBe(true);
  expect(screen.getByRole('alert').textContent).toMatch(/backend running/);
});

test('Stop closes the stream (the server then stops the LLM)', () => {
  const es = setup();
  es.emit('sources', SOURCES);
  fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
  expect(es.closed).toBe(true);
  expect(screen.getByText('Stopped.')).toBeTruthy();
});

test('history: refreshes after save, opens a stored answer, deletes', async () => {
  const es = setup();
  history = [{ id: 'h1', question: 'how do I read a csv?', status: 'answered', answer: 'Use a reader [1].',
    sources: SOURCES.map(({ text, ...s }) => s), cited: [1], coverage: 1, uncited: [], invalid: [], timings_ms: {} }];
  answerWith(es, 'Use a reader [1].');
  const item = await screen.findByRole('button', { name: 'how do I read a csv?' });
  fireEvent.click(item);
  await screen.findByText(/100% of claims cited/);
  fireEvent.click(screen.getAllByRole('button', { name: 'Show source 1' })[0]);
  expect(screen.getByText(/isn't stored in history/)).toBeTruthy(); // history keeps no passage text

  fireEvent.click(screen.getByRole('button', { name: 'Delete "how do I read a csv?"' }));
  await act(async () => {});
  expect(fetchCalls).toContainEqual(['/api/history/h1', 'DELETE']);
  expect(screen.queryByRole('button', { name: 'how do I read a csv?' })).toBe(null);
});

test('example questions ask immediately', () => {
  render(<App EventSourceImpl={FakeEventSource} fetchImpl={fakeFetch} />);
  fireEvent.click(screen.getByRole('button', { name: 'What is a feature toggle?' }));
  expect(FakeEventSource.last.url).toBe('/api/ask?q=What%20is%20a%20feature%20toggle%3F');
});
