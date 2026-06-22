import { describe, it, expect, vi } from 'vitest';
import { OllamaEmbeddingAdapter } from '../ollama.js';

describe('OllamaEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = OllamaEmbeddingAdapter();
    expect(adapter.dimension()).toBe(768);
  });

  it('calls Ollama API with correct payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ embedding: [0.1, 0.2] }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OllamaEmbeddingAdapter('http://localhost:11434');
    const results = await adapter.embed(['test']);

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual([0.1, 0.2]);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/embeddings',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('nomic-embed-text'),
      })
    );
  });

  it('throws on non-ok response', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('internal error'),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OllamaEmbeddingAdapter();
    await expect(adapter.embed(['test'])).rejects.toThrow('Ollama API error 500');
  });
});
