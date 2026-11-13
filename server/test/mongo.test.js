// MongoHistoryStore against a real MongoDB (skipped when none is reachable).
// Uses its own database, dropped afterwards, so real history is never touched.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { MongoHistoryStore, historyDoc } from '../src/store.js';

const URL = process.env.MONGO_URL ?? 'mongodb://127.0.0.1:27017';
let store = null;
try {
  store = await MongoHistoryStore.connect(URL, 'miniperplexity_test');
} catch {
  // no MongoDB running: the tests below report as skipped
}
const opts = { skip: store ? false : `no MongoDB at ${URL}` };

before(async () => {
  if (store) await store.col.deleteMany({});
});
after(async () => {
  if (store) {
    await store.col.db.dropDatabase();
    await store.close();
  }
});

test('save, list newest first, get, remove', opts, async () => {
  const done = { status: 'answered', text: 'A [1].', cited: [1], coverage: 1 };
  const a = await store.save(historyDoc('first?', [{ n: 1, url: 'u', title: 't', chunk_id: 'c', text: 'long' }], done));
  await new Promise((r) => setTimeout(r, 5));
  const b = await store.save(historyDoc('second?', [], null, 'Gemini API error 503'));
  assert.match(a, /^[0-9a-f]{24}$/);

  const list = await store.list(10);
  assert.deepEqual(list.map((h) => [h.question, h.status]), [['second?', 'error'], ['first?', 'answered']]);
  assert.deepEqual(Object.keys(list[0]).sort(), ['createdAt', 'id', 'question', 'status']);

  const doc = await store.get(a);
  assert.equal(doc.id, a);
  assert.equal(doc._id, undefined);
  assert.equal(doc.sources[0].text, undefined);
  assert.ok(doc.createdAt instanceof Date);

  assert.equal(await store.remove(b), true);
  assert.equal(await store.get(b), null);
  assert.equal(await store.remove(b), false);
});

test('malformed ids are not found, not errors', opts, async () => {
  for (const id of ['nope', '123', 'zzzzzzzzzzzzzzzzzzzzzzzz', '{"$ne": null}']) {
    assert.equal(await store.get(id), null);
    assert.equal(await store.remove(id), false);
  }
});

test('the createdAt index exists', opts, async () => {
  const names = (await store.col.indexes()).map((i) => i.name);
  assert.ok(names.includes('createdAt_-1'));
});
