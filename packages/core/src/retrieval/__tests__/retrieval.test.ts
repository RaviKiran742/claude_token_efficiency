import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

// ---------------------------------------------------------------------------
// Stateful in-memory mock for better-sqlite3
// ---------------------------------------------------------------------------
// The native binding cannot be loaded on this machine (no VS Build Tools),
// so we mock the entire module.  The mock stores rows in Maps keyed by
// primary key so that INSERT OR REPLACE works correctly.
// ---------------------------------------------------------------------------

const _rows: Record<string, Map<string, Record<string, unknown>>> = {
  files:   new Map(),
  nodes:   new Map(),
  edges:   new Map(),
  vectors: new Map(),
};

const _reset = () => {
  for (const m of Object.values(_rows)) m.clear();
};

vi.mock('better-sqlite3', () => {
  return {
    default: vi.fn(() => {
      const db = {
        pragma: vi.fn(),
        exec:    vi.fn((_sql: string) => { _reset(); }),
        close:   vi.fn(),
        prepare: vi.fn((sql: string) => {
          const isInsert = (t: string) => sql.includes(`INSERT OR REPLACE INTO ${t}`);
          const isSelectFrom = (t: string) => /^SELECT\b/.test(sql.trim()) && sql.includes(` FROM ${t}`);

          // -- run -----------------------------------------------------------------
          const run = (...args: unknown[]) => {
            if (isInsert('files')) {
              const r = args[0] as Record<string, unknown>;
              _rows.files.set(r.id as string, { id: r.id, path: r.path, mtime: r.mtime, ast_hash: r.astHash });
            } else if (isInsert('nodes')) {
              const r = args[0] as Record<string, unknown>;
              _rows.nodes.set(r.id as string, {
                id: r.id, type: r.type, file_id: r.fileId,
                start_line: r.startLine, end_line: r.endLine,
                name: r.name, parent_id: r.parentId, body: r.body,
              });
            } else if (isInsert('edges')) {
              const r = args[0] as Record<string, unknown>;
              const k = `${r.source}|${r.target}|${r.type}`;
              _rows.edges.set(k, { source: r.source, target: r.target, type: r.type, weight: r.weight });
            } else if (isInsert('vectors')) {
              _rows.vectors.set(args[0] as string, { node_id: args[0], embedding: args[1] });
            }
          };

          // -- get -----------------------------------------------------------------
          const get = () => {
            if (sql.includes('COUNT(*)') && sql.includes(' FROM files'))  return { cnt: _rows.files.size };
            if (sql.includes('COUNT(*)') && sql.includes(' FROM nodes'))  return { cnt: _rows.nodes.size };
            if (sql.includes('COUNT(*)') && sql.includes(' FROM edges'))  return { cnt: _rows.edges.size };
            if (sql.includes('MAX(mtime)') && sql.includes(' FROM files')) {
              let mt: number | null = null;
              for (const f of _rows.files.values()) {
                const v = f.mtime as number;
                if (mt === null || v > mt) mt = v;
              }
              return { mt };
            }
            if (sql.includes('path, mtime') && sql.includes(' FROM files')) {
              const first = _rows.files.values().next().value;
              return first ? { path: first.path, mtime: first.mtime } : null;
            }
            return null;
          };

          // -- all ----------------------------------------------------------------
          const all = () => {
            if (isSelectFrom('files')) {
              return Array.from(_rows.files.values()).map(f => ({
                id: f.id, path: f.path, mtime: f.mtime,
                astHash: f.ast_hash as string,
              }));
            }
            if (isSelectFrom('nodes')) {
              return Array.from(_rows.nodes.values()).map(n => ({
                id: n.id, type: n.type,
                fileId:     n.file_id as string,
                startLine:  n.start_line as number,
                endLine:    n.end_line as number,
                name:       n.name as string | null,
                parentId:   n.parent_id as string | null,
                body:       n.body as string,
              }));
            }
            if (isSelectFrom('edges')) {
              return Array.from(_rows.edges.values());
            }
            if (isSelectFrom('vectors')) {
              return Array.from(_rows.vectors.values());
            }
            if (sql.includes('path, mtime') && sql.includes(' FROM files')) {
              return Array.from(_rows.files.values()).map(f => ({
                path: f.path as string,
                mtime: f.mtime as number,
              }));
            }
            return [];
          };

          return { run, get, all };
        }),
        transaction: vi.fn(<T extends (...a: unknown[]) => unknown>(fn: T): T => fn),
      };
      return db;
    }),
  };
});

import { Retriever } from '../index.js';
import { Store } from '../../store/index.js';
import { FakeEmbeddingAdapter } from '../../embeddings/fake.js';
import { DEFAULT_CONFIG } from '../../config/index.js';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdirSync, rmSync } from 'node:fs';

function tmpDir(): string {
  const dir = join(tmpdir(), 'cc-opts-retrieval-' + Math.random().toString(36).slice(2));
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** Seed vectors explicitly so ordering is deterministic regardless of embedding adapter. */
function explicitVectors(dim: number): {
  loginVec: number[];
  logoutVec: number[];
  dbVec: number[];
} {
  // loginVec is close to [1,0,0,...], dbVec is close to [0,1,0,...], logoutVec is [0,0,1,...]
  // This makes cosine similarity with query [1,0,0,...] rank login > db > logout
  const loginVec = new Array(dim).fill(0);
  loginVec[0] = 1;
  const dbVec = new Array(dim).fill(0);
  dbVec[1] = 1;
  const logoutVec = new Array(dim).fill(0);
  logoutVec[2] = 1;
  return { loginVec, logoutVec, dbVec };
}

describe('Retriever', () => {
  let store: Store;
  let tmp: string;

  beforeEach(() => {
    tmp = tmpDir();
    store = Store.open(join(tmp, 'graph.db'));
  });

  afterEach(() => {
    store.close();
    rmSync(tmp, { recursive: true, force: true });
  });

  it('returns ranked results for a query', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/auth.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/auth.ts::login', type: 'function', fileId: 'f1', startLine: 1, endLine: 20,
        name: 'login', parentId: null, body: 'function login() { ... }' },
      { id: 'src/auth.ts::logout', type: 'function', fileId: 'f1', startLine: 22, endLine: 35,
        name: 'logout', parentId: null, body: 'function logout() { ... }' },
      { id: 'src/db.ts::connect', type: 'function', fileId: 'f1', startLine: 1, endLine: 10,
        name: 'connect', parentId: null, body: 'function connect() { ... }' },
    ]);
    store.writeEdges([
      { source: 'src/auth.ts::login', target: 'src/db.ts::connect', type: 'calls', weight: 1.0 },
    ]);

    // Use explicit vectors so ranking is deterministic
    const dim = 16;
    const vecs = explicitVectors(dim);
    store.writeVectors(new Map([
      ['src/auth.ts::login', vecs.loginVec],
      ['src/auth.ts::logout', vecs.logoutVec],
      ['src/db.ts::connect', vecs.dbVec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('fix login authentication bug', { tokens: 1000 });

    expect(results.length).toBeGreaterThan(0);
    // Results should be well-structured
    for (const r of results) {
      expect(r.file).toBeTruthy();
      expect(r.startLine).toBeGreaterThan(0);
      expect(r.endLine).toBeGreaterThan(0);
      expect(typeof r.score).toBe('number');
      expect(r.body).toBeTruthy();
    }
  });

  it('returns empty results when store has no nodes', async () => {
    const embedder = FakeEmbeddingAdapter(16);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('fix bug');
    expect(results).toEqual([]);
  });

  it('applies token budget correctly', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/a.ts::big', type: 'function', fileId: 'f1',
        startLine: 1, endLine: 200, name: 'big', parentId: null, body: '...'.repeat(200) },
      { id: 'src/a.ts::small', type: 'function', fileId: 'f1',
        startLine: 201, endLine: 202, name: 'small', parentId: null, body: 'fn small' },
    ]);

    // small node has higher vector score (closer to unit vector), big node has lower
    // but both will be considered — budget only fits small (10 estimated tokens vs 1000)
    const dim = 16;
    const smallVec = new Array(dim).fill(0);
    smallVec[0] = 1; // very close to query
    const bigVec = new Array(dim).fill(0);
    bigVec[dim - 1] = 0.1; // far from query

    store.writeVectors(new Map([
      ['src/a.ts::big', bigVec],
      ['src/a.ts::small', smallVec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('small', { tokens: 100 });
    // big node is ~1000 estimated tokens, so it won't fit even if ranked high;
    // small node is ~10 estimated tokens and will fit
    expect(results.length).toBeGreaterThan(0);
    expect(results.every((r) => r.name === 'small')).toBe(true);
  });

  it('ranks results by score in descending order', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/a.ts::fn1', type: 'function', fileId: 'f1',
        startLine: 1, endLine: 5, name: 'fn1', parentId: null, body: 'fn1 body' },
      { id: 'src/a.ts::fn2', type: 'function', fileId: 'f1',
        startLine: 10, endLine: 14, name: 'fn2', parentId: null, body: 'fn2 body' },
      { id: 'src/a.ts::fn3', type: 'function', fileId: 'f1',
        startLine: 20, endLine: 24, name: 'fn3', parentId: null, body: 'fn3 body' },
    ]);

    const dim = 16;
    // Vectors are designed so fn2 has highest cosine with the hash of "query text"
    // query text hash first dims: [-0.08, 0.69, 0.63, 0.08, ...]
    // fn2 = [0, 0.9, 0.1, 0, ...] → cosine = 0.32
    // fn1 = [0.5, 0.5, 0, 0, ...] → cosine = 0.18
    // fn3 = [0, 0, 0, 1, 0, ...] → cosine = 0.04
    const fn1Vec = new Array(dim).fill(0);
    fn1Vec[0] = 0.5; fn1Vec[1] = 0.5;
    const fn2Vec = new Array(dim).fill(0);
    fn2Vec[1] = 0.9; fn2Vec[2] = 0.1;
    const fn3Vec = new Array(dim).fill(0);
    fn3Vec[3] = 1;

    store.writeVectors(new Map([
      ['src/a.ts::fn1', fn1Vec],
      ['src/a.ts::fn2', fn2Vec],
      ['src/a.ts::fn3', fn3Vec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('query text', { tokens: 1000 });

    // fn2 should rank first (highest cosine similarity with query hash)
    if (results.length >= 2) {
      expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    }
    expect(results[0].name).toBe('fn2');
  });
});
