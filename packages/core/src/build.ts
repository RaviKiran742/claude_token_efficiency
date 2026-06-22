import { mkdirSync, renameSync, rmSync, existsSync, writeFileSync, readFileSync, appendFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { Indexer } from './indexer/index.js';
import { Store } from './store/index.js';
import { buildGraph, diffuseEmbeddings } from './graph/index.js';
import { EmbeddingAdapter, CcOptimizeConfig, CodeNode, IndexStats } from './types.js';

/** Number of files to parse + embed per batch. */
const BATCH_SIZE = 50;

/** Maximum texts per embedding API call. */
const EMBED_BATCH_SIZE = 100;

/** OpenAI embedding models have a hard 8192-token input limit.
 *  We conservatively estimate 1 token ≈ 4 characters for code,
 *  which gives ~32768 chars. Using 30000 as a safe margin. */
const MAX_EMBEDDING_CHARS = 30000;

/**
 * Truncate embedding input text to fit within the model's token limit.
 * Preserves the function-name prefix and keeps a meaningful suffix.
 */
function truncateEmbeddingText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;

  // Try to find the colon separator (from "name: body" format)
  const colonIdx = text.indexOf(':');
  const prefix = colonIdx > 0 ? text.slice(0, colonIdx + 2) : '';
  const bodyStart = colonIdx > 0 ? colonIdx + 2 : 0;
  const availableForBody = maxChars - prefix.length - 20; // 20 chars for truncation marker

  const head = text.slice(bodyStart, bodyStart + Math.floor(availableForBody * 0.7));
  const tail = text.slice(-Math.floor(availableForBody * 0.3));

  return prefix + head + '\n// ... [truncated] ...\n' + tail;
}

export async function buildIndex(
  rootPath: string,
  store: Store,
  embedder: EmbeddingAdapter,
  config: CcOptimizeConfig,
): Promise<IndexStats> {
  const indexer = new Indexer(config);

  // ── Phase 0: Walk files ──────────────────────────────────────
  console.log('Walking files...');
  const files = indexer.walkFiles(rootPath);
  console.log(`  ${files.length} files found`);

  store.clear();
  store.writeFiles(files);

  // ── Phase 1: Parse + embed in batches ────────────────────────
  console.log('Parsing and embedding (phase 1)...');
  let totalNodes = 0;

  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    const batch = files.slice(i, i + BATCH_SIZE);

    // Parse every file in this batch, collecting all nodes
    const batchNodes: CodeNode[] = [];
    for (const file of batch) {
      const nodes = indexer.parseFile(rootPath, file);
      batchNodes.push(...nodes);
    }

    if (batchNodes.length === 0) continue;

    // Write nodes to store immediately (frees node body strings from memory)
    store.writeNodes(batchNodes);
    totalNodes += batchNodes.length;

    // Embed nodes in sub-batches (API max inputs = 100)
    // Truncate any single node that exceeds the model's token limit
    const texts = batchNodes.map((n) => {
      const raw = n.name ? `${n.name}: ${n.body}` : n.body;
      if (raw.length > MAX_EMBEDDING_CHARS) {
        const truncated = truncateEmbeddingText(raw, MAX_EMBEDDING_CHARS);
        console.warn(`  ⚠  Truncated oversized node: ${n.id} (${raw.length} → ${truncated.length} chars, ~${Math.round(raw.length / 4)} tokens)`);
        return truncated;
      }
      return raw;
    });

    for (let e = 0; e < texts.length; e += EMBED_BATCH_SIZE) {
      const textBatch = texts.slice(e, e + EMBED_BATCH_SIZE);
      const nodeBatch = batchNodes.slice(e, e + EMBED_BATCH_SIZE);
      const vecs = await embedder.embed(textBatch);

      const vecMap = new Map<string, number[]>();
      for (let v = 0; v < nodeBatch.length; v++) {
        vecMap.set(nodeBatch[v].id, vecs[v]);
      }
      store.writeVectors(vecMap);
    }

    console.log(`  batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(files.length / BATCH_SIZE)} — ${totalNodes} nodes so far`);
  }

  console.log(`  total: ${totalNodes} nodes`);

  // ── Phase 2: Build cross-file edges ──────────────────────────
  console.log('Building call graph edges (phase 2)...');

  // Build a name→node-id map from stored nodes (load once, O(1) lookups)
  const storedNodes = store.getNodes();
  const nodeNameMap = new Map<string, string[]>();
  for (const n of storedNodes) {
    if (n.type !== 'function' || !n.name) continue;
    const existing = nodeNameMap.get(n.name);
    if (existing) {
      existing.push(n.id);
    } else {
      nodeNameMap.set(n.name, [n.id]);
    }
  }

  let totalEdges = 0;
  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    const batch = files.slice(i, i + BATCH_SIZE);
    for (const file of batch) {
      const edges = indexer.buildEdgesForFile(rootPath, file, nodeNameMap);
      if (edges.length > 0) {
        store.writeEdges(edges);
        totalEdges += edges.length;
      }
    }
    console.log(`  batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(files.length / BATCH_SIZE)} — ${totalEdges} edges so far`);
  }

  console.log(`  total: ${totalEdges} edges`);

  // ── Phase 3: Graph diffusion (needs all embeddings at once) ───
  console.log('Building graph and diffusing (phase 3)...');
  const allNodes = store.getNodes();
  const allEdges = store.getEdges();
  const allVectors = store.getAllVectors();

  const graph = buildGraph(allNodes, allEdges);
  graph.setEmbeddings(allVectors);
  const diffused = diffuseEmbeddings(graph, config.index.diffusionAlpha);

  // Rewrite vectors with diffused versions
  store.writeVectors(diffused);
  console.log('  diffusion complete');

  const stats = store.getStats();
  console.log(`Build complete: ${stats.fileCount} files, ${stats.nodeCount} nodes, ${stats.edgeCount} edges`);
  return stats;
}

/** Build and persist index into <rootPath>/.context-optimizer/ atomically. */
export async function buildIndexAtomic(
  rootPath: string,
  embedder: EmbeddingAdapter,
  config: CcOptimizeConfig,
): Promise<IndexStats> {
  const optDir = resolve(rootPath, '.context-optimizer');
  const tmpDir = resolve(rootPath, '.context-optimizer-tmp');

  // Clean up any leftover temp
  if (existsSync(tmpDir)) {
    rmSync(tmpDir, { recursive: true, force: true });
  }

  mkdirSync(tmpDir, { recursive: true });
  const tmpDbPath = join(tmpDir, 'graph.db');

  const store = Store.open(tmpDbPath);
  let stats: IndexStats;
  try {
    stats = await buildIndex(rootPath, store, embedder, config);
  } finally {
    store.close();
  }

  // Write config snapshot
  writeFileSync(
    join(tmpDir, 'config.json'),
    JSON.stringify(config, null, 2),
  );

  // Atomic swap
  if (existsSync(optDir)) {
    rmSync(optDir, { recursive: true, force: true });
  }
  renameSync(tmpDir, optDir);

  // Auto-add to .gitignore
  const gitignorePath = resolve(rootPath, '.gitignore');
  const gitignoreLine = '.context-optimizer/';
  try {
    const contents = readFileSync(gitignorePath, 'utf-8');
    if (!contents.includes(gitignoreLine)) {
      appendFileSync(gitignorePath, '\n' + gitignoreLine + '\n');
    }
  } catch {
    appendFileSync(gitignorePath, gitignoreLine + '\n');
  }

  return stats;
}
