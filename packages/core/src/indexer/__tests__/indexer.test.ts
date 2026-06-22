import { describe, it, expect } from 'vitest';
import { Indexer } from '../index.js';
import { DEFAULT_CONFIG } from '../../config/index.js';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CodeNode } from '../../types.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const fixturesDir = resolve(__dirname, 'fixtures');

function buildNameMap(nodes: CodeNode[]): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const n of nodes) {
    if (n.type !== 'function' || !n.name) continue;
    const existing = map.get(n.name);
    if (existing) {
      existing.push(n.id);
    } else {
      map.set(n.name, [n.id]);
    }
  }
  return map;
}

describe('Indexer', () => {
  it('extracts function nodes from a TypeScript file', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = [];
    const indexer = new Indexer(config);
    const files = indexer.walkFiles(fixturesDir);

    const sampleFile = files.find((f) => f.path.includes('sample.ts'));
    expect(sampleFile).toBeDefined();

    const nodes = indexer.parseFile(fixturesDir, sampleFile!);
    const addNode = nodes.find((n) => n.name === 'add');
    expect(addNode).toBeDefined();
    expect(addNode!.type).toBe('function');
    expect(addNode!.startLine).toBeGreaterThan(0);
    expect(addNode!.endLine).toBeGreaterThan(addNode!.startLine);
    expect(addNode!.body).toContain('function add');
  });

  it('finds call edges between functions', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = [];
    const indexer = new Indexer(config);
    const files = indexer.walkFiles(fixturesDir);

    // Collect all nodes first to build name map
    const allNodes: CodeNode[] = [];
    for (const file of files) {
      allNodes.push(...indexer.parseFile(fixturesDir, file));
    }
    const nameMap = buildNameMap(allNodes);

    // Build edges using the name map
    const sampleFile = files.find((f) => f.path.includes('sample.ts'))!;
    const edges = indexer.buildEdgesForFile(fixturesDir, sampleFile, nameMap);

    const callsEdge = edges.find(
      (e) => e.type === 'calls' && e.target.includes('multiply')
    );
    expect(callsEdge).toBeDefined();
    expect(callsEdge!.weight).toBe(1.0);
  });

  it('creates file entries for each source file', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = [];
    const indexer = new Indexer(config);
    const files = indexer.walkFiles(fixturesDir);

    const sampleFile = files.find((f) => f.path.includes('sample.ts'));
    expect(sampleFile).toBeDefined();
    expect(sampleFile!.astHash).toBeTruthy();
  });

  it('respects ignore patterns', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = ['**/sample.ts'];
    const indexer = new Indexer(config);
    const files = indexer.walkFiles(fixturesDir);

    const sampleFile = files.find((f) => f.path.includes('sample.ts'));
    expect(sampleFile).toBeUndefined();
  });
});
