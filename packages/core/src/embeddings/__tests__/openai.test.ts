import { describe, it, expect, vi } from 'vitest';
import { OpenAIEmbeddingAdapter } from '../openai.js';

describe('OpenAIEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = OpenAIEmbeddingAdapter('sk-fake');
    expect(adapter.dimension()).toBe(1536);
  });

  it('calls the OpenAI API with correct payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
      }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OpenAIEmbeddingAdapter('sk-test');
    const results = await adapter.embed(['test input']);

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual([0.1, 0.2, 0.3]);

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/embeddings',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer sk-test',
        }),
        body: expect.stringContaining('test input'),
      })
    );
  });

  it('retries on failure and throws after 3 attempts', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('network error'));
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OpenAIEmbeddingAdapter('sk-test');
    await expect(adapter.embed(['test'])).rejects.toThrow('Embedding API failed after 3 attempts');
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('returns empty array for empty input', async () => {
    const adapter = OpenAIEmbeddingAdapter('sk-test');
    const results = await adapter.embed([]);
    expect(results).toEqual([]);
  });
});
