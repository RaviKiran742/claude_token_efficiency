import { describe, it, expect } from 'vitest';
import { Indexer } from '../index.js';
import { DEFAULT_CONFIG } from '../../config/index.js';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const fixturesDir = resolve(__dirname, 'fixtures');

describe('Indexer', () => {
  it('extracts function nodes from a TypeScript file', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = []; // don't ignore fixtures
    const indexer = new Indexer(config);
    const result = indexer.build(fixturesDir);

    const addNode = result.nodes.find((n) => n.name === 'add');
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
    const result = indexer.build(fixturesDir);

    const callsEdge = result.edges.find(
      (e) => e.type === 'calls' && e.target.includes('multiply')
    );
    expect(callsEdge).toBeDefined();
    expect(callsEdge!.weight).toBe(1.0);
  });

  it('creates file entries for each source file', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = [];
    const indexer = new Indexer(config);
    const result = indexer.build(fixturesDir);

    const sampleFile = result.files.find((f) => f.path.includes('sample.ts'));
    expect(sampleFile).toBeDefined();
    expect(sampleFile!.astHash).toBeTruthy();
  });

  it('respects ignore patterns', () => {
    const config = { ...DEFAULT_CONFIG };
    config.index.ignore = ['**/sample.ts'];
    const indexer = new Indexer(config);
    const result = indexer.build(fixturesDir);

    const sampleFile = result.files.find((f) => f.path.includes('sample.ts'));
    expect(sampleFile).toBeUndefined();
  });
});
