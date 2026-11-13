// Start the backend. Settings (environment variables):
//   PORT           3001
//   HOST           127.0.0.1   (docker-compose will set 0.0.0.0)
//   RETRIEVAL_URL  http://127.0.0.1:8000   the Python service (python -m api.app)
//   HISTORY_STORE  mongo | memory          memory = no MongoDB needed, history lost on restart
//   MONGO_URL      mongodb://127.0.0.1:27017
//   MONGO_DB       miniperplexity

import { createApp } from './app.js';
import { MemoryHistoryStore, MongoHistoryStore } from './store.js';

const env = process.env;
const port = Number(env.PORT ?? 3001);
const host = env.HOST ?? '127.0.0.1';
const retrievalUrl = (env.RETRIEVAL_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const storeKind = env.HISTORY_STORE ?? 'mongo';

let store;
if (storeKind === 'memory') {
  store = new MemoryHistoryStore();
  console.warn('HISTORY_STORE=memory: history is kept in memory and lost on restart');
} else {
  const url = env.MONGO_URL ?? 'mongodb://127.0.0.1:27017';
  try {
    store = await MongoHistoryStore.connect(url, env.MONGO_DB ?? 'miniperplexity');
  } catch (e) {
    console.error(`Cannot reach MongoDB at ${url} (${e.message}).`);
    console.error('Start it (Docker Desktop, then: docker start mp-mongo) or run with HISTORY_STORE=memory.');
    process.exit(1);
  }
}

const server = createApp({ store, retrievalUrl }).listen(port, host, () => {
  console.log(`backend on http://${host}:${port}  (retrieval: ${retrievalUrl}, history: ${storeKind})`);
});

// Ctrl-C: stop accepting requests, then close the database connection cleanly
process.on('SIGINT', () => {
  server.close(() => store.close().finally(() => process.exit(0)));
  server.closeAllConnections(); // don't wait for open answer streams
});
