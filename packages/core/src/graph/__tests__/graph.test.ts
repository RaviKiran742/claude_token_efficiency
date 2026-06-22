import { describe, it, expect } from 'vitest';
import { buildGraph, diffuseEmbeddings } from '../index.js';
import { CodeNode, GraphEdge } from '../../types.js';

function makeNodes(): CodeNode[] {
  return [
    { id: 'a', type: 'function', fileId: 'f1', startLine: 1, endLine: 5, name: 'a', parentId: null, body: 'fn a' },
    { id: 'b', type: 'function', fileId: 'f1', startLine: 6, endLine: 10, name: 'b', parentId: null, body: 'fn b' },
    { id: 'c', type: 'function', fileId: 'f1', startLine: 11, endLine: 15, name: 'c', parentId: null, body: 'fn c' },
  ];
}

function makeEdges(): GraphEdge[] {
  return [
    { source: 'a', target: 'b', type: 'calls', weight: 1.0 },
    { source: 'b', target: 'c', type: 'calls', weight: 1.0 },
  ];
}

describe('Graph', () => {
  it('builds adjacency correctly', () => {
    const nodes = makeNodes();
    const edges = makeEdges();
    const graph = buildGraph(nodes, edges);

    const neighbors = graph.getNeighbors('a');
    expect(neighbors).toHaveLength(1);
    expect(neighbors[0].target).toBe('b');
  });

  it('bfsDistance returns correct distances', () => {
    const nodes = makeNodes();
    const edges = makeEdges();
    const graph = buildGraph(nodes, edges);

    expect(graph.bfsDistance('a', 'a', 0.5)).toBe(0);
    expect(graph.bfsDistance('a', 'b', 0.5)).toBe(1);
    expect(graph.bfsDistance('a', 'c', 0.5)).toBe(2);
  });

  it('bfsDistance returns Infinity when unreachable', () => {
    const nodes = makeNodes();
    const graph = buildGraph(nodes, []); // no edges
    expect(graph.bfsDistance('a', 'c', 0.5)).toBe(Infinity);
  });

  it('bfsDistance respects minWeight threshold', () => {
    const nodes = makeNodes();
    const edges = [{ source: 'a', target: 'b', type: 'calls' as const, weight: 0.3 }];
    const graph = buildGraph(nodes, edges);
    expect(graph.bfsDistance('a', 'b', 0.5)).toBe(Infinity);
    expect(graph.bfsDistance('a', 'b', 0)).toBe(1);
  });

  it('getNeighbors respects minWeight filter', () => {
    const nodes = makeNodes();
    const edges = [
      { source: 'a', target: 'b', type: 'calls' as const, weight: 0.9 },
      { source: 'a', target: 'c', type: 'uses_type' as const, weight: 0.3 },
    ];
    const graph = buildGraph(nodes, edges);
    expect(graph.getNeighbors('a', 0.5)).toHaveLength(1);
    expect(graph.getNeighbors('a', 0)).toHaveLength(2);
  });
});

describe('diffuseEmbeddings', () => {
  it('diffuses embeddings along edges', () => {
    const nodes = makeNodes();
    const edges = makeEdges();
    const graph = buildGraph(nodes, edges);

    // Set embeddings before diffusion
    const embeddings = new Map<string, number[]>();
    embeddings.set('a', [1, 0, 0]);
    embeddings.set('b', [0, 1, 0]);
    embeddings.set('c', [0, 0, 1]);
    graph.setEmbeddings(embeddings);

    const result = diffuseEmbeddings(graph, 0.5);

    // a has neighbor b=[0,1,0], so a's diffused = 0.5*[1,0,0] + 0.5*[0,1,0] = [0.5, 0.5, 0]
    expect(result.get('a')![0]).toBeCloseTo(0.5, 4);
    expect(result.get('a')![1]).toBeCloseTo(0.5, 4);
    expect(result.get('a')![2]).toBeCloseTo(0, 4);
  });

  it('no-op when alpha is 0', () => {
    const nodes = makeNodes();
    const edges = makeEdges();
    const graph = buildGraph(nodes, edges);
    const embeddings = new Map<string, number[]>();
    embeddings.set('a', [1, 0]);
    graph.setEmbeddings(embeddings);

    const result = diffuseEmbeddings(graph, 0);
    expect(result.get('a')![0]).toBeCloseTo(1, 4);
    expect(result.get('a')![1]).toBeCloseTo(0, 4);
  });
});
