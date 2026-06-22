import { mkdirSync, renameSync, rmSync, existsSync, writeFileSync, readFileSync, appendFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { Indexer } from './indexer/index.js';
import { Store } from './store/index.js';
import { buildGraph, diffuseEmbeddings } from './graph/index.js';
import { EmbeddingAdapter, CcOptimizeConfig, IndexStats } from './types.js';

export async function buildIndex(
  rootPath: string,
  store: Store,
  embedder: EmbeddingAdapter,
  config: CcOptimizeConfig,
): Promise<IndexStats> {
  console.log('Walking files...');
  const indexer = new Indexer(config);
  const { files, nodes, edges } = indexer.build(rootPath);

  console.log(`  ${files.length} files, ${nodes.length} nodes, ${edges.length} edges`);

  // Embed all nodes
  console.log('Embedding nodes...');
  const bodies = nodes.map((n) => (n.name ? `${n.name}: ${n.body}` : n.body));
  const embeddings = await embedder.embed(bodies);

  const embeddingMap = new Map<string, number[]>();
  for (let i = 0; i < nodes.length; i++) {
    embeddingMap.set(nodes[i].id, embeddings[i]);
  }

  // Build graph and diffuse
  console.log('Building graph and diffusing...');
  const graph = buildGraph(nodes, edges);
  graph.setEmbeddings(embeddingMap);
  const diffused = diffuseEmbeddings(graph, config.index.diffusionAlpha);
  graph.setEmbeddings(diffused);

  // Write to store (clear first for rebuild)
  store.clear();
  store.writeFiles(files);
  store.writeNodes(nodes);
  store.writeEdges(edges);
  store.writeVectors(diffused);

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
