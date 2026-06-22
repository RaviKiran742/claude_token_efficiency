import { EmbeddingAdapter } from '../types.js';

export function OllamaEmbeddingAdapter(
  baseUrl = 'http://localhost:11434',
  model = 'nomic-embed-text'
): EmbeddingAdapter {
  return {
    dimension: () => 768,
    async embed(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];

      const results: number[][] = [];
      for (const text of texts) {
        const response = await fetch(`${baseUrl}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model, prompt: text }),
        });

        if (!response.ok) {
          const body = await response.text();
          throw new Error(`Ollama API error ${response.status}: ${body}`);
        }

        const data = await response.json() as { embedding: number[] };
        results.push(data.embedding);
      }

      return results;
    },
  };
}
