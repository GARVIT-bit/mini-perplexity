# Mini Perplexity

An answer engine built from scratch: my own crawler and search index find the passages, and an LLM answers **only** from them, with a citation on every sentence that links to the exact passage it came from.

No Elasticsearch, no hosted vector database, no LangChain retriever. Retrieval is my own code: inverted index, BM25, compressed postings, embeddings with HNSW, rank fusion.

```
crawler ──▶ parser + chunker ──▶ inverted index + BM25 ─┐
                               └▶ MiniLM embeddings + HNSW ─┴▶ RRF fusion ──▶ top 8 passages
                                                                                   │
React UI ◀── SSE ── Express + MongoDB (history) ◀── SSE ── FastAPI ◀── LLM answer with [n] citations
```

## Results

Every number below comes from a script in `bench/` that writes the linked file; rerun the script to reproduce it. Measured on a Windows laptop CPU.

**Corpus** ([crawl_stats.md](bench/results/crawl_stats.md)): 1,027 pages crawled from docs.python.org and martinfowler.com (99.7% of attempted URLs fetched); 971 kept after parsing, split into 25,864 chunks of ~120 words.

**Retrieval quality**, 53 labelled questions ([retrieval_eval.md](bench/results/retrieval_eval.md)):

| method | Recall@10 | nDCG@10 | Answer@8 | p50 latency |
|---|---:|---:|---:|---:|
| BM25 | 0.769 | 0.730 | 0.774 | 19 ms |
| vector (MiniLM, exact) | 0.884 | 0.825 | 0.868 | 34 ms |
| **hybrid (BM25 + vector, RRF)**: used | **0.890** | 0.789 | **0.868** | 61 ms |
| hybrid + cross-encoder re-rank (L6, 50 → 8) | 0.876 | 0.813 | 0.849 | 3,172 ms |

Re-ranking did not beat hybrid on Answer@8 (does the top 8 contain the answer?) and cost ~3 s per query on CPU, so the engine sends hybrid's top 8 to the LLM without it. Paraphrased questions are the weak spot for every method (nDCG@10 0.32 for BM25 up to 0.51 for vector).

**Vector search** ([hnsw.md](bench/results/hnsw.md)): HNSW (M=16, ef=100) finds 98.2% of the exact top 10 at 0.92 ms median, vs 2.8 ms for brute force over 25,864 × 384-dim vectors.

**Answers**, gemini-3.1-flash-lite on the same question set ([answer_eval.md](bench/results/answer_eval.md)). 48 of 59 questions got a response (41 answers, 7 `NOT_FOUND`); 11 hit Gemini free-tier overload (HTTP 503) and are excluded.

| metric | value |
|---|---:|
| answer sentences with a valid citation | 100% |
| answers citing a passage that doesn't exist | 0% |
| answers citing a page the labels say answers the question | 92.7% |
| unanswerable questions correctly refused (`NOT_FOUND`) | 5 of 6 |
| claims judged supported by their cited passage (NLI) | 73.2% (lower bound, see below) |
| retrieval latency, p50 | 73 ms |

**Postings compression** ([compression.md](bench/results/compression.md)), 1.32 M postings: delta + varint encoding stores a posting in 2.16 bytes, 27% of fixed int32 (8 bytes); in memory the index's postings take 9.9 MB instead of 99.5 MB as Python objects (10× less). The price is BM25 speed, which drops from 2.5 ms to 7.3 ms mean per query because postings are decoded on the fly in pure Python. (Timings measured with other containers keeping the CPU ~70% busy; byte counts don't depend on load.)

### How to read these numbers honestly

- **Labels:** the question set (`bench/eval/questions.jsonl`) was drafted with an LLM's help while reading the corpus and has not been fully human-reviewed.
- **Faithfulness is a lower bound.** A local NLI model (`nli-deberta-v3-xsmall`) checks each claim against 1–2 sentence windows of its cited passage. Controls run on every report: it accepts 100% of sentences copied from a passage and wrongly accepts 3.8% of claims checked against an unrelated question's passages. But a hand check of its 10 lowest scores found ~7 that are actually supported (paraphrase, facts combined across passages, negation), so the true rate is higher than 73.2%. Not yet calibrated against a human-labelled sample.
- **End-to-end latency is mostly the LLM queue.** Answers took 5.4 s at p50 end to end on Gemini's free tier, nearly all of it waiting for the first token; my retrieval is ~73 ms of that.

## Run it

**Docker** (everything, one command). Needs a built index in `data/index` (see below):

```bash
docker compose up --build
```

Open http://localhost:8080. Without settings it uses a fake LLM that needs no key; for real answers copy `.env.example` to `.env` and set `MP_LLM=gemini`, `MP_MODEL=gemini-3.1-flash-lite` and `GEMINI_API_KEY` (free tier from aistudio.google.com).

**Build the index** (Python 3.12, from the repo root):

```bash
python -m pip install -r requirements.txt
python -m crawler.crawl --seeds seeds.example.txt --max-pages 300 --workers 20 --delay 1 --max-per-host 150
python -m indexer.build
python -m indexer.embed --index data/index
python -m retrieval.hnsw --index data/index
```

**Without Docker**: three terminals, `python -m api.app --llm gemini --model gemini-3.1-flash-lite` (port 8000), `cd server; npm start` (port 3001, MongoDB on 27017), `cd web; npm run dev` (http://localhost:5173).

**Tests**: `python -m pytest -q`, and `npm test` in `server/` and in `web/`.

## Layout

| folder | what |
|---|---|
| `crawler/` | async crawler: SQLite frontier, robots.txt, per-host politeness, allow/deny focus rules, retries |
| `indexer/` | trafilatura parsing + link-density filter, chunking, analyzer (Snowball stemmer), inverted index, varint postings, embeddings |
| `retrieval/` | BM25, brute-force and HNSW vector search, RRF fusion, cross-encoder re-ranker |
| `answer/` | prompt with numbered passages, Claude / Gemini / fake LLM clients, per-sentence citation checker, answer engine |
| `api/` | FastAPI: `/search`, `/answer` as Server-Sent Events |
| `server/` | Express + MongoDB: relays the answer stream live, saves question history |
| `web/` | React UI: streaming answer, `[n]` chips that open the cited passage, sources grouped by page, history |
| `bench/` | every benchmark and eval behind the numbers above |

## Design decisions

- **Hybrid retrieval with Reciprocal Rank Fusion (k=60)**: BM25 nails exact identifiers (`csv.DictReader`), vectors catch meaning; RRF merges their rankings without tuning score scales.
- **No re-ranker in the answer path**: measured above; it cost seconds and didn't find more answers.
- **A fixed `NOT_FOUND` token** instead of free-form "I couldn't find…", so refusal is detectable and measurable.
- **Citations are checked, not trusted**: every sentence is parsed for `[n]`, numbers outside 1..8 are flagged as made up, and the UI shows exactly the passage each chip points to.
- **Untrusted text stays text**: crawled pages and LLM output are rendered as React text nodes, never as HTML.
- **Streaming end to end**: the answer streams from the LLM through FastAPI, Express and nginx (buffering off) to the browser; closing the tab cancels the LLM call.

## License

[MIT](LICENSE). The crawled pages themselves are not included in this repository; their content belongs to their authors.
