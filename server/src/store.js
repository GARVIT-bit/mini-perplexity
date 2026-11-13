// Question history: one document per asked question.
//
// Two stores with the same four methods, so the app doesn't know which it has:
//   MongoHistoryStore   the real one (MongoDB)
//   MemoryHistoryStore  tests, and running without Docker (lost on restart)
//
// Document shape (what /api/history/:id returns):
//   { id, question, status, answer, sources: [{n, url, title, chunk_id}],
//     cited, coverage, uncited, invalid, model, input_tokens, output_tokens,
//     timings_ms, error, createdAt }
// Passage text is NOT stored: it is already in the index (chunk_id finds it),
// and 8 x ~120 words per question would make history documents ~10x bigger.

import { randomUUID } from 'node:crypto';
import { MongoClient, ObjectId } from 'mongodb';

/** Build the stored document from what the proxy saw in the stream. */
export function historyDoc(question, sources, done, error) {
  return {
    question,
    status: error ? 'error' : done?.status ?? 'incomplete',
    answer: done?.text ?? '',
    sources: (sources ?? []).map(({ n, url, title, chunk_id }) => ({ n, url, title, chunk_id })),
    cited: done?.cited ?? [],
    coverage: done?.coverage ?? null,
    uncited: done?.uncited ?? [],
    invalid: done?.invalid ?? [],
    model: done?.model ?? null,
    input_tokens: done?.input_tokens ?? 0,
    output_tokens: done?.output_tokens ?? 0,
    timings_ms: done?.timings_ms ?? {},
    error: error ?? null,
    createdAt: new Date(),
  };
}

/** List rows are small: the UI's sidebar needs no answer text. */
function summary(doc) {
  return { id: doc.id, question: doc.question, status: doc.status, createdAt: doc.createdAt };
}

export class MemoryHistoryStore {
  constructor() {
    this.docs = new Map();
  }
  async save(doc) {
    const id = randomUUID();
    this.docs.set(id, { ...doc, id });
    return id;
  }
  async list(limit) {
    return [...this.docs.values()]
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, limit)
      .map(summary);
  }
  async get(id) {
    return this.docs.get(id) ?? null;
  }
  async remove(id) {
    return this.docs.delete(id);
  }
  async ping() {
    return true;
  }
  async close() {}
}

export class MongoHistoryStore {
  static async connect(url, dbName = 'miniperplexity') {
    // fail fast (3 s) if MongoDB isn't running, instead of the driver's 30 s default
    const client = new MongoClient(url, { serverSelectionTimeoutMS: 3000 });
    await client.connect();
    const store = new MongoHistoryStore(client, client.db(dbName).collection('history'));
    // newest-first listing is the only query we run on many documents: index it
    await store.col.createIndex({ createdAt: -1 });
    return store;
  }

  constructor(client, col) {
    this.client = client;
    this.col = col;
  }

  async save(doc) {
    const { insertedId } = await this.col.insertOne({ ...doc });
    return insertedId.toHexString();
  }
  async list(limit) {
    const rows = await this.col
      .find({}, { projection: { question: 1, status: 1, createdAt: 1 } })
      .sort({ createdAt: -1 })
      .limit(limit)
      .toArray();
    return rows.map((r) => summary(withId(r)));
  }
  async get(id) {
    const oid = toObjectId(id);
    return oid ? withId(await this.col.findOne({ _id: oid })) : null;
  }
  async remove(id) {
    const oid = toObjectId(id);
    return oid ? (await this.col.deleteOne({ _id: oid })).deletedCount === 1 : false;
  }
  async ping() {
    await this.client.db('admin').command({ ping: 1 });
    return true;
  }
  async close() {
    await this.client.close();
  }
}

// a malformed id is "not found", not a server error
function toObjectId(id) {
  return ObjectId.isValid(id) && String(new ObjectId(id)) === id ? new ObjectId(id) : null;
}

function withId(doc) {
  if (!doc) return null;
  const { _id, ...rest } = doc;
  return { id: _id.toHexString(), ...rest };
}
