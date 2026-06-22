import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdirSync, rmSync, writeFileSync, statSync } from 'node:fs';
import { Store } from '../index.js';

function tmpStore(): { store: Store; dbPath: string } {
  const dir = join(tmpdir(), 'cc-opts-test-' + Math.random().toString(36).slice(2));
  mkdirSync(dir, { recursive: true });
  const dbPath = join(dir, 'graph.db');
  return { store: Store.open(dbPath), dbPath: dir };
}

describe('Store', () => {
  let store: Store;
  let dbDir: string;

  beforeEach(() => {
    const s = tmpStore();
    store = s.store;
    dbDir = s.dbPath;
  });

  afterEach(() => {
    store.close();
    rmSync(dbDir, { recursive: true, force: true });
  });

  it('writes and reads files', () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    const files = store.getFiles();
    expect(files).toHaveLength(1);
    expect(files[0].path).toBe('src/a.ts');
  });

  it('writes and reads nodes', () => {
    store.writeNodes([{
      id: 'src/a.ts::login',
      type: 'function',
      fileId: 'f1',
      startLine: 10,
      endLine: 30,
      name: 'login',
      parentId: null,
      body: 'function login() { ... }',
    }]);
    const nodes = store.getNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].name).toBe('login');
  });

  it('writes and reads edges', () => {
    store.writeEdges([{
      source: 'src/a.ts::login',
      target: 'src/b.ts::hash',
      type: 'calls',
      weight: 1.0,
    }]);
    const edges = store.getEdges();
    expect(edges).toHaveLength(1);
    expect(edges[0].type).toBe('calls');
  });

  it('writes and searches vectors', () => {
    store.writeVectors(new Map([
      ['n1', [1, 0, 0]],
      ['n2', [0, 1, 0]],
      ['n3', [0.7, 0.7, 0]],
    ]));

    const results = store.searchVectors([1, 0, 0], 2);
    expect(results).toHaveLength(2);
    expect(results[0].nodeId).toBe('n1');
    expect(results[0].score).toBeCloseTo(1.0, 4);
  });

  it('clear removes all data', () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([{
      id: 'n1', type: 'function', fileId: 'f1',
      startLine: 1, endLine: 5, name: 'f', parentId: null, body: '...',
    }]);
    store.clear();
    expect(store.getFiles()).toHaveLength(0);
    expect(store.getNodes()).toHaveLength(0);
  });

  it('getStats returns correct counts', () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([{
      id: 'n1', type: 'function', fileId: 'f1',
      startLine: 1, endLine: 5, name: 'f', parentId: null, body: '...',
    }]);
    store.writeEdges([{
      source: 'n1', target: 'n1', type: 'calls', weight: 1.0,
    }]);
    const stats = store.getStats();
    expect(stats.fileCount).toBe(1);
    expect(stats.nodeCount).toBe(1);
    expect(stats.edgeCount).toBe(1);
    expect(stats.lastBuilt).toBe(1000);
  });

  it('overwrites existing data on re-insert', () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 2000, astHash: 'def' }]);
    const files = store.getFiles();
    expect(files).toHaveLength(1);
    expect(files[0].mtime).toBe(2000);
  });

  // -----------------------------------------------------------------------
  // Fix 1: hasStaleFiles tests
  // -----------------------------------------------------------------------

  it('detects stale files when mtime has changed', () => {
    const tmp = join(tmpdir(), 'cc-opts-stale-' + Math.random().toString(36).slice(2));
    mkdirSync(tmp, { recursive: true });
    const srcDir = join(tmp, 'src');
    mkdirSync(srcDir, { recursive: true });
    const filePath = join(srcDir, 'auth.ts');
    writeFileSync(filePath, 'function login() {}');
    const realMtime = statSync(filePath).mtimeMs;

    // Write file record with an older mtime to simulate staleness
    store.writeFiles([{
      id: 'f1', path: 'src/auth.ts', mtime: realMtime - 10000, astHash: 'abc',
    }]);

    // The file on disk has mtime > stored mtime → stale
    const result = store.hasStaleFiles(tmp);
    expect(result).toBe(true);

    rmSync(tmp, { recursive: true, force: true });
  });

  it('reports no staleness when all files are up to date', () => {
    const tmp = join(tmpdir(), 'cc-opts-stale-' + Math.random().toString(36).slice(2));
    mkdirSync(tmp, { recursive: true });
    const srcDir = join(tmp, 'src');
    mkdirSync(srcDir, { recursive: true });
    const filePath = join(srcDir, 'auth.ts');
    writeFileSync(filePath, 'function login() {}');
    const realMtime = statSync(filePath).mtimeMs;

    // Write file record with a newer mtime → not stale
    store.writeFiles([{
      id: 'f1', path: 'src/auth.ts', mtime: realMtime + 10000, astHash: 'abc',
    }]);

    const result = store.hasStaleFiles(tmp);
    expect(result).toBe(false);

    rmSync(tmp, { recursive: true, force: true });
  });

  // -----------------------------------------------------------------------
  // Fix 2: searchVectors edge case tests
  // -----------------------------------------------------------------------

  it('searchVectors returns empty when no vectors stored', () => {
    const results = store.searchVectors([1, 0, 0], 5);
    expect(results).toEqual([]);
  });

  it('searchVectors handles zero query vector', () => {
    store.writeVectors(new Map([
      ['n1', [1, 0, 0]],
    ]));
    const results = store.searchVectors([0, 0, 0], 5);
    expect(results).toHaveLength(1);
    expect(results[0].score).toBe(0);
  });

  it('searchVectors returns fewer results when k > available', () => {
    store.writeVectors(new Map([
      ['n1', [1, 0, 0]],
      ['n2', [0, 1, 0]],
    ]));
    const results = store.searchVectors([1, 0, 0], 10);
    expect(results).toHaveLength(2);
  });
});
