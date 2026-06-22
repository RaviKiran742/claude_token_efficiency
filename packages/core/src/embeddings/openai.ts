import { EmbeddingAdapter } from '../types.js';

const MAX_INPUTS_PER_REQUEST = 100;
const RETRY_DELAYS = [1000, 2000, 4000];

async function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export function OpenAIEmbeddingAdapter(
  apiKey: string,
  model = 'text-embedding-3-small'
): EmbeddingAdapter {
  const MODEL_DIMENSIONS: Record<string, number> = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
  };
  const dimension = MODEL_DIMENSIONS[model] ?? 1536;

  return {
    dimension: () => dimension,

    async embed(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];

      const allEmbeddings: number[][] = [];

      // Chunk into batches of 100
      for (let i = 0; i < texts.length; i += MAX_INPUTS_PER_REQUEST) {
        const batch = texts.slice(i, i + MAX_INPUTS_PER_REQUEST);

        let lastError: Error | null = null;
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const response = await fetch('https://api.openai.com/v1/embeddings', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model,
                input: batch,
                encoding_format: 'float',
              }),
            });

            if (!response.ok) {
              const body = await response.text();
              throw new Error(`OpenAI API error ${response.status}: ${body}`);
            }

            const data = await response.json() as {
              data: { embedding: number[]; index: number }[];
            };

            // Restore original ordering
            data.data.sort((a, b) => a.index - b.index);
            allEmbeddings.push(...data.data.map((d) => d.embedding));
            lastError = null;
            break;
          } catch (err: any) {
            lastError = err;
            if (attempt < 2) {
              await sleep(RETRY_DELAYS[attempt]);
            }
          }
        }

        if (lastError) {
          throw new Error(
            `Embedding API failed after 3 attempts for batch starting at index ${i}: ${lastError.message}`
          );
        }
      }

      return allEmbeddings;
    },
  };
}
