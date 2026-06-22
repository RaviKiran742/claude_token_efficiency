import { existsSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  loadConfig,
  buildIndexAtomic,
  Store,
  Retriever,
  OpenAIEmbeddingAdapter,
  OllamaEmbeddingAdapter,
  EmbeddingAdapter,
  CcOptimizeConfig,
} from '@cc-optimize/core';

function getRoot(opts: Record<string, string>): string {
  return resolve(opts['--root'] ?? process.cwd());
}

function createEmbedder(config: CcOptimizeConfig): EmbeddingAdapter {
  if (config.embeddings.provider === 'ollama') {
    return OllamaEmbeddingAdapter();
  }
  if (config.embeddings.provider === 'openai') {
    if (!config.embeddings.apiKey) {
      throw new Error('OpenAI API key not set. Provide it via .cc-optimize.json or OPENAI_API_KEY env var.');
    }
    return OpenAIEmbeddingAdapter(config.embeddings.apiKey, config.embeddings.model);
  }
  throw new Error(`Unknown embedding provider: ${config.embeddings.provider}`);
}

export async function run(argv: string[]): Promise<void> {
  const command = argv[0];
  const args = parseArgs(argv.slice(1));

  try {
    switch (command) {
      case 'build':
        await cmdBuild(args);
        break;
      case 'rebuild':
        await cmdRebuild(args);
        break;
      case 'query':
        await cmdQuery(args);
        break;
      case 'status':
        await cmdStatus(args);
        break;
      case 'mcp':
        await cmdMcp(args);
        break;
      default:
        console.error(`Unknown command: ${command}\nUsage: cc-optimize <build|rebuild|query|status|mcp>`);
        process.exit(1);
    }
  } catch (err: any) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }
}

function parseArgs(argv: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  let positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i];
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      result[key] = val;
    } else {
      positional.push(argv[i]);
    }
  }
  result.__positional = positional.join(' ');
  return result;
}

async function cmdBuild(args: Record<string, string>): Promise<void> {
  const root = getRoot(args);
  const config = loadConfig(root);
  const embedder = createEmbedder(config);
  const stats = await buildIndexAtomic(root, embedder, config);
  console.log(`Index built: ${stats.nodeCount} nodes, ${stats.edgeCount} edges`);
}

async function cmdRebuild(args: Record<string, string>): Promise<void> {
  const root = getRoot(args);
  const optDir = resolve(root, '.context-optimizer');
  if (existsSync(optDir)) {
    rmSync(optDir, { recursive: true, force: true });
  }
  const config = loadConfig(root);
  const embedder = createEmbedder(config);
  const stats = await buildIndexAtomic(root, embedder, config);
  console.log(`Index rebuilt: ${stats.nodeCount} nodes, ${stats.edgeCount} edges`);
}

async function cmdQuery(args: Record<string, string>): Promise<void> {
  const root = getRoot(args);
  const task = args.__positional;

  if (!task) {
    console.error('Usage: cc-optimize query <task description> [--file <path>] [--line <n>] [--tokens <n>] [--format context|manifest|json]');
    process.exit(1);
  }

  const optDir = resolve(root, '.context-optimizer');
  if (!existsSync(optDir)) {
    console.error('No index found. Run cc-optimize build first.');
    process.exit(1);
  }

  const config = loadConfig(root);
  const embedder = createEmbedder(config);
  const store = Store.open(resolve(optDir, 'graph.db'));
  // Check staleness
  if (store.hasStaleFiles(root)) {
    console.warn('⚠  Some files have been modified since the last build. Run cc-optimize rebuild to update.');
  }

  try {
    const retriever = new Retriever(store, embedder, config.index);
    const results = await retriever.query(task, {
      file: args['--file'],
      line: args['--line'] ? parseInt(args['--line']) : undefined,
      tokens: args['--tokens'] ? parseInt(args['--tokens']) : undefined,
    });

    const format = args['--format'] ?? 'context';
    outputResults(results, format);
  } finally {
    store.close();
  }
}

function outputResults(results: any[], format: string): void {
  switch (format) {
    case 'manifest':
      for (const r of results) {
        console.log(`${r.file}:${r.startLine}-${r.endLine}`);
      }
      break;
    case 'json':
      console.log(JSON.stringify(results, null, 2));
      break;
    case 'context':
    default:
      for (const r of results) {
        console.log(`// ${r.file}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})`);
        console.log(r.body);
        console.log('');
      }
      break;
  }
}

async function cmdStatus(args: Record<string, string>): Promise<void> {
  const root = getRoot(args);
  const optDir = resolve(root, '.context-optimizer');

  if (!existsSync(optDir)) {
    console.log('No index found. Run cc-optimize build to create one.');
    return;
  }

  const store = Store.open(resolve(optDir, 'graph.db'));
  try {
    const stats = store.getStats();
    console.log(`Files:  ${stats.fileCount}`);
    console.log(`Nodes:  ${stats.nodeCount}`);
    console.log(`Edges:  ${stats.edgeCount}`);
    console.log(`Built:  ${stats.lastBuilt ? new Date(stats.lastBuilt).toISOString() : 'unknown'}`);
    if (store.hasStaleFiles(root)) {
      console.log('Status: STALE — some files modified since build');
    } else {
      console.log('Status: current');
    }
  } finally {
    store.close();
  }
}

async function cmdMcp(_args: Record<string, string>): Promise<void> {
  // Dynamic import to avoid loading MCP SDK when not needed
  const { start } = await import('@cc-optimize/mcp');
  await start();
}
