import { CodeNode, GraphEdge } from '../types.js';

export class Graph {
  private nodeMap: Map<string, CodeNode>;
  private adjOut: Map<string, GraphEdge[]>; // outgoing edges
  private adjIn: Map<string, GraphEdge[]>;  // incoming edges
  private embeddings: Map<string, number[]>;

  constructor(nodes: CodeNode[], edges: GraphEdge[], embeddings: Map<string, number[]>) {
    this.nodeMap = new Map(nodes.map((n) => [n.id, n]));
    this.embeddings = new Map(embeddings);
    this.adjOut = new Map();
    this.adjIn = new Map();

    for (const n of nodes) {
      this.adjOut.set(n.id, []);
      this.adjIn.set(n.id, []);
    }

    for (const e of edges) {
      this.adjOut.get(e.source)?.push(e);
      this.adjIn.get(e.target)?.push(e);
    }
  }

  getNode(id: string): CodeNode | undefined {
    return this.nodeMap.get(id);
  }

  getEmbedding(id: string): number[] | undefined {
    return this.embeddings.get(id);
  }

  getEmbeddings(): Map<string, number[]> {
    return new Map(this.embeddings);
  }

  setEmbeddings(embeddings: Map<string, number[]>): void {
    this.embeddings = new Map(embeddings);
  }

  getAllNodes(): CodeNode[] {
    return Array.from(this.nodeMap.values());
  }

  getNeighbors(id: string, minWeight = 0): GraphEdge[] {
    return this.adjOut.get(id)?.filter((e) => e.weight >= minWeight) ?? [];
  }

  nodeCount(): number {
    return this.nodeMap.size;
  }

  /** BFS distance on directed graph, respecting min weight threshold. */
  bfsDistance(fromId: string, toId: string, minWeight = 0.5): number {
    if (fromId === toId) return 0;

    const visited = new Set<string>();
    const queue: Array<[string, number]> = [[fromId, 0]];
    visited.add(fromId);

    while (queue.length > 0) {
      const [current, dist] = queue.shift()!;
      const neighbors = this.getNeighbors(current, minWeight);
      for (const edge of neighbors) {
        if (visited.has(edge.target)) continue;
        if (edge.target === toId) return dist + 1;
        visited.add(edge.target);
        queue.push([edge.target, dist + 1]);
      }
    }

    return Infinity;
  }
}

/**
 * Build graph from nodes and edges. Assumes embeddings have been computed
 * and will be set separately via setEmbeddings.
 */
export function buildGraph(
  nodes: CodeNode[],
  edges: GraphEdge[],
): Graph {
  const emptyEmbeddings = new Map<string, number[]>();
  for (const n of nodes) {
    emptyEmbeddings.set(n.id, []);
  }
  return new Graph(nodes, edges, emptyEmbeddings);
}

/**
 * One-pass diffusion: E_final = (1 - alpha) * E_semantic + alpha * A_normalized * E_semantic
 */
export function diffuseEmbeddings(
  graph: Graph,
  alpha = 0.3,
): Map<string, number[]> {
  const original = graph.getEmbeddings();
  const result = new Map<string, number[]>();

  for (const [id, emb] of original.entries()) {
    const neighbors = graph.getNeighbors(id, 0);
    const dim = emb.length;
    const neighborMean = new Array(dim).fill(0);

    if (neighbors.length > 0) {
      for (const edge of neighbors) {
        const neighborEmb = original.get(edge.target);
        if (neighborEmb) {
          for (let i = 0; i < dim; i++) {
            neighborMean[i] += (neighborEmb[i] * edge.weight) / neighbors.length;
          }
        }
      }
    }

    const diffused = new Array(dim);
    for (let i = 0; i < dim; i++) {
      diffused[i] = (1 - alpha) * emb[i] + alpha * neighborMean[i];
    }

    result.set(id, diffused);
  }

  graph.setEmbeddings(result);
  return result;
}
