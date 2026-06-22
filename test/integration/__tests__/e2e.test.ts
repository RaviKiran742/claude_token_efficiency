import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { resolve } from 'node:path';
import { rmSync, existsSync } from 'node:fs';
import {
  buildIndexAtomic,
  Store,
  Retriever,
  FakeEmbeddingAdapter,
  DEFAULT_CONFIG,
} from '@cc-optimize/core';

const FIXTURE_ROOT = resolve(import.meta.dirname!, '../../fixtures/sample-repo');

describe('end-to-end', () => {
  beforeAll(async () => {
    // Build index over fixture repo
    const adapter = FakeEmbeddingAdapter(16);
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    config.index.ignore = [];
    await buildIndexAtomic(FIXTURE_ROOT, adapter, config);
  });

  afterAll(() => {
    const optDir = resolve(FIXTURE_ROOT, '.context-optimizer');
    if (existsSync(optDir)) {
      rmSync(optDir, { recursive: true, force: true });
    }
  });

  it('retrieves auth-related nodes for auth query', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('fix auth bug', { tokens: 5000 });

      const names = results.map((r) => r.name).filter(Boolean);
      // Ground truth: login, verifyToken, findById should be in results
      const hasAuthFunction = names.some((n) => n === 'login' || n === 'verifyToken');
      expect(hasAuthFunction).toBe(true);
    } finally {
      store.close();
    }
  });

  it('retrieves db-related nodes for database query', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('add database logging', { tokens: 5000 });

      const names = results.map((r) => r.name).filter(Boolean);
      const hasDbFunction = names.some((n) => n === 'query' || n === 'log' || n === 'connect');
      expect(hasDbFunction).toBe(true);
    } finally {
      store.close();
    }
  });

  it('returns results in correct format with scores', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('refactor user module');

      for (const r of results) {
        expect(r.file).toBeTruthy();
        expect(r.startLine).toBeGreaterThan(0);
        expect(r.endLine).toBeGreaterThan(0);
        expect(typeof r.score).toBe('number');
        expect(r.body).toBeTruthy();
      }
    } finally {
      store.close();
    }
  });
});
