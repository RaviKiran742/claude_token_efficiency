import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import {
  loadConfig,
  Store,
  Retriever,
  OpenAIEmbeddingAdapter,
  OllamaEmbeddingAdapter,
  EmbeddingAdapter,
} from '@cc-optimize/core';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

function createEmbedder(config: any): EmbeddingAdapter {
  if (config.embeddings.provider === 'ollama') {
    return OllamaEmbeddingAdapter();
  }
  if (config.embeddings.provider === 'openai') {
    if (!config.embeddings.apiKey) {
      throw new Error('OpenAI API key not set.');
    }
    return OpenAIEmbeddingAdapter(config.embeddings.apiKey, config.embeddings.model);
  }
  throw new Error(`Unknown embedding provider: ${config.embeddings.provider}`);
}

export async function start(): Promise<void> {
  const server = new Server(
    { name: 'context-optimizer', version: '0.1.0' },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'retrieve_context',
        description: 'Retrieve the most relevant code regions for a given task from the indexed codebase.',
        inputSchema: {
          type: 'object',
          properties: {
            task:   { type: 'string', description: 'Natural language task description' },
            file:   { type: 'string', description: 'Current file path (optional, biases retrieval)' },
            line:   { type: 'number', description: 'Current line number (optional, biases retrieval)' },
            tokens: { type: 'number', description: 'Token budget (default: 8000)' },
            format: { type: 'string', enum: ['context', 'manifest', 'json'], default: 'context' },
          },
          required: ['task'],
        },
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params.name !== 'retrieve_context') {
      throw new Error(`Unknown tool: ${request.params.name}`);
    }

    const args = (request.params.arguments ?? {}) as Record<string, any>;
    const root = process.cwd();
    const optDir = resolve(root, '.context-optimizer');

    if (!existsSync(optDir)) {
      return {
        content: [{ type: 'text', text: 'Error: No index found. Run cc-optimize build first.' }],
        isError: true,
      };
    }

    const config = loadConfig(root);
    const embedder = createEmbedder(config);
    const store = Store.open(resolve(optDir, 'graph.db'));

    try {
      const retriever = new Retriever(store, embedder, config.index);
      const results = await retriever.query(args.task, {
        file: args.file,
        line: args.line,
        tokens: args.tokens ?? config.index.defaultTokenBudget,
      });

      let text: string;
      switch (args.format ?? 'context') {
        case 'manifest':
          text = results.map((r: any) => `${r.file}:${r.startLine}-${r.endLine}`).join('\n');
          break;
        case 'json':
          text = JSON.stringify(results, null, 2);
          break;
        default:
          text = results.map((r: any) =>
            `// ${r.file}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.body}`
          ).join('\n\n');
          break;
      }

      return { content: [{ type: 'text', text }] };
    } finally {
      store.close();
    }
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}
