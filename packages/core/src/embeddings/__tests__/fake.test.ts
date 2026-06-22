import { describe, it, expect } from 'vitest';
import { FakeEmbeddingAdapter, createHashEmbedding } from '../fake.js';

describe('FakeEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = FakeEmbeddingAdapter(768);
    expect(adapter.dimension()).toBe(768);
  });

  it('produces deterministic embeddings', async () => {
    const adapter = FakeEmbeddingAdapter(16);
    const a = await adapter.embed(['hello world']);
    const b = await adapter.embed(['hello world']);
    expect(a[0]).toEqual(b[0]);
  });

  it('produces different embeddings for different texts', async () => {
    const adapter = FakeEmbeddingAdapter(16);
    const [a] = await adapter.embed(['function auth()']);
    const [b] = await adapter.embed(['function db()']);
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    // Extremely unlikely to collide; should be different
    expect(Math.abs(dot)).toBeLessThan(5);
  });

  it('batches multiple texts', async () => {
    const adapter = FakeEmbeddingAdapter(64);
    const results = await adapter.embed(['a', 'b', 'c']);
    expect(results).toHaveLength(3);
    expect(results[0]).toHaveLength(64);
  });
});
