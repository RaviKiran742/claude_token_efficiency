import { Graph, buildGraph } from '../graph/index.js';
import { Store } from '../store/index.js';
import { EmbeddingAdapter, QueryResult, CodeNode, CcOptimizeConfig } from '../types.js';

export class Retriever {
  private store: Store;
  private embedder: EmbeddingAdapter;
  private config: CcOptimizeConfig['index'];

  constructor(store: Store, embedder: EmbeddingAdapter, config: CcOptimizeConfig['index']) {
    this.store = store;
    this.embedder = embedder;
    this.config = config;
  }

  async query(
    task: string,
    opts?: { file?: string; line?: number; tokens?: number },
  ): Promise<QueryResult[]> {
    const tokenBudget = opts?.tokens ?? this.config.defaultTokenBudget;

    // 1. Embed the query
    const [queryVec] = await this.embedder.embed([task]);

    // 2. Vector search — top 20 candidates
    const vectorCandidates = this.store.searchVectors(queryVec, 20);

    // 3. Load graph from store
    const nodes = this.store.getNodes();
    const edges = this.store.getEdges();
    const graph = buildGraph(nodes, edges);

    // 4. Build candidate set
    const candidateIds = new Set<string>(vectorCandidates.map((c) => c.nodeId));
    const semanticScores = new Map<string, number>(
      vectorCandidates.map((c) => [c.nodeId, c.score]),
    );

    // 5. Cursor anchor
    let cursorNodeId: string | null = null;
    if (opts?.file && opts?.line !== undefined) {
      cursorNodeId = this.resolveCursorNode(nodes, opts.file, opts.line);
      if (cursorNodeId) {
        candidateIds.add(cursorNodeId);
        // Depth-1 neighbors of cursor
        for (const edge of graph.getNeighbors(cursorNodeId, 0.5)) {
          candidateIds.add(edge.target);
        }
      }
    }

    // 6. Graph expansion — for each candidate, walk edges with weight >= 0.5
    const expandedIds = new Set<string>(candidateIds);
    for (const id of candidateIds) {
      for (const edge of graph.getNeighbors(id, 0.5)) {
        expandedIds.add(edge.target);
      }
    }

    // 7. Score fusion
    const items: Array<{ node: CodeNode; score: number }> = [];
    const nodeMap = new Map<string, CodeNode>(nodes.map((n) => [n.id, n]));

    for (const id of expandedIds) {
      const node = nodeMap.get(id);
      if (!node) continue;

      const semantic = semanticScores.get(id) ?? 0;

      let graphProximity = 0;
      if (cursorNodeId) {
        const dist = graph.bfsDistance(cursorNodeId, id, 0.5);
        if (dist !== Infinity) {
          graphProximity = 1 / (1 + dist);
        }
      }

      const finalScore = cursorNodeId
        ? 0.6 * semantic + 0.4 * graphProximity
        : semantic;
      items.push({ node, score: finalScore });
    }

    // Sort by score descending
    items.sort((a, b) => b.score - a.score);

    // 8. Apply token budget (greedy)
    const results: QueryResult[] = [];
    let tokensUsed = 0;

    for (const item of items) {
      const estimatedTokens = (item.node.endLine - item.node.startLine + 1) * 5;
      if (tokensUsed + estimatedTokens > tokenBudget) continue;
      tokensUsed += estimatedTokens;
      results.push({
        file: item.node.id.split('::')[0] ?? '',
        startLine: item.node.startLine,
        endLine: item.node.endLine,
        name: item.node.name,
        score: item.score,
        body: item.node.body,
      });
    }

    // 9. Deduplicate overlapping ranges (same file + adjacent lines), merge them
    return this.deduplicate(results);
  }

  private resolveCursorNode(nodes: CodeNode[], file: string, line: number): string | null {
    let best: CodeNode | null = null;
    let bestDist = Infinity;

    for (const n of nodes) {
      const nFile = n.id.split('::')[0] ?? '';
      if (nFile !== file) continue;
      // Find closest node that contains this line or is near it
      if (line >= n.startLine && line <= n.endLine) {
        // Exact containment
        const mid = (n.startLine + n.endLine) / 2;
        const dist = Math.abs(line - mid);
        if (dist < bestDist) {
          bestDist = dist;
          best = n;
        }
      }
    }

    return best?.id ?? null;
  }

  private deduplicate(results: QueryResult[]): QueryResult[] {
    if (results.length <= 1) return results;

    // Sort by file then line
    results.sort((a, b) => a.file.localeCompare(b.file) || a.startLine - b.startLine);

    const merged: QueryResult[] = [];
    let current = { ...results[0] };

    for (let i = 1; i < results.length; i++) {
      const next = results[i];
      if (next.file === current.file && next.startLine <= current.endLine + 3) {
        // Merge — extend range, keep max score, join bodies
        current.endLine = Math.max(current.endLine, next.endLine);
        current.score = Math.max(current.score, next.score);
        current.body += '\n' + next.body;
      } else {
        merged.push(current);
        current = { ...next };
      }
    }
    merged.push(current);

    // Re-sort by score descending
    merged.sort((a, b) => b.score - a.score);
    return merged;
  }
}
