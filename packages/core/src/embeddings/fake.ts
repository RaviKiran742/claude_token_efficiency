import { createHash } from 'node:crypto';
import { EmbeddingAdapter } from '../types.js';

/**
 * Deterministic hash-based embedding. Seed is SHA256 of text, then
 * expanded into `dimension` floats using the hash bytes cyclically.
 */
export function createHashEmbedding(text: string, dimension: number): number[] {
  const hash = createHash('sha256').update(text).digest();
  const vec: number[] = [];
  for (let i = 0; i < dimension; i++) {
    const byte = hash[i % hash.length];
    vec.push((byte / 255) * 2 - 1); // Map to [-1, 1]
  }
  return vec;
}

export function FakeEmbeddingAdapter(dimension = 1536): EmbeddingAdapter {
  return {
    dimension: () => dimension,
    async embed(texts: string[]): Promise<number[][]> {
      return texts.map((t) => createHashEmbedding(t, dimension));
    },
  };
}
