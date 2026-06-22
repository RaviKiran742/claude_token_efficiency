import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { resolve } from 'node:path';

// ---------------------------------------------------------------------------
// Mock factories — these are hoisted by vitest, but their closures capture
// the variables below (which are initialized before the factory is called).
// ---------------------------------------------------------------------------

const mockBuildIndexAtomic = vi.fn();
const mockStoreOpen = vi.fn();
const mockStoreClose = vi.fn();
const mockStoreGetStats = vi.fn();
const mockStoreHasStaleFiles = vi.fn();
const mockRetrieverQuery = vi.fn();
const mockMcpStart = vi.fn();

vi.mock('@cc-optimize/core', () => ({
  loadConfig: vi.fn(),
  buildIndexAtomic: mockBuildIndexAtomic,
  Store: {
    open: mockStoreOpen,
  },
  Retriever: vi.fn().mockImplementation(() => ({
    query: mockRetrieverQuery,
  })),
  OpenAIEmbeddingAdapter: vi.fn().mockReturnValue({ embed: vi.fn(), dimension: vi.fn() }),
  OllamaEmbeddingAdapter: vi.fn().mockReturnValue({ embed: vi.fn(), dimension: vi.fn() }),
}));

vi.mock('node:fs', async () => {
  const actual = await vi.importActual<typeof import('node:fs')>('node:fs');
  return {
    ...actual,
    existsSync: vi.fn(actual.existsSync),
    rmSync: vi.fn(actual.rmSync),
  };
});

// Mock the MCP package so the dynamic import in cmdMcp works
vi.mock('@cc-optimize/mcp', () => ({
  start: mockMcpStart,
}));

// ---------------------------------------------------------------------------
// Test state
// ---------------------------------------------------------------------------

let logOutput: string[] = [];
let warnOutput: string[] = [];
let errorOutput: string[] = [];
let exitCode: number | undefined;

beforeEach(() => {
  logOutput = [];
  warnOutput = [];
  errorOutput = [];
  exitCode = undefined;

  // Use stubGlobal for console to ensure reliable interception in ESM context
  vi.stubGlobal('console', {
    log: (...args: unknown[]) => { logOutput.push(args.map(String).join(' ')); },
    warn: (...args: unknown[]) => { warnOutput.push(args.map(String).join(' ')); },
    error: (...args: unknown[]) => { errorOutput.push(args.map(String).join(' ')); },
  });

  // Stub process.exit and process.cwd
  vi.stubGlobal('process', {
    ...process,
    argv: process.argv,
    env: process.env,
    stdout: process.stdout,
    stderr: process.stderr,
    stdin: process.stdin,
    exit: vi.fn((code?: number) => { exitCode = code; }) as unknown as typeof process.exit,
    cwd: vi.fn(() => '/fake/root') as typeof process.cwd,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  mockBuildIndexAtomic.mockReset();
  mockStoreOpen.mockReset();
  mockStoreClose.mockReset();
  mockStoreGetStats.mockReset();
  mockStoreHasStaleFiles.mockReset();
  mockRetrieverQuery.mockReset();
  mockMcpStart.mockReset();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function getRun() {
  const mod = await import('../index.js');
  return mod.run;
}

function makeStoreStub() {
  return {
    close: mockStoreClose,
    getStats: mockStoreGetStats,
    hasStaleFiles: mockStoreHasStaleFiles,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('run dispatcher', () => {
  it('exits with error for unknown command', async () => {
    const run = await getRun();
    await run(['foobar']);
    expect(errorOutput.join(' ')).toContain('Unknown command: foobar');
    expect(exitCode).toBe(1);
  });

  it('exits with error when no command given', async () => {
    const run = await getRun();
    await run([]);
    expect(errorOutput.join(' ')).toContain('Unknown command: undefined');
    expect(exitCode).toBe(1);
  });

  it('catches thrown errors and exits', async () => {
    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockImplementation(() => {
      throw new Error('boom');
    });
    const run = await getRun();
    await run(['build']);
    expect(errorOutput.join(' ')).toContain('boom');
    expect(exitCode).toBe(1);
  });
});

describe('cmdBuild', () => {
  it('builds index and logs stats', async () => {
    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);
    mockBuildIndexAtomic.mockResolvedValue({ nodeCount: 42, edgeCount: 7 });

    const run = await getRun();
    await run(['build']);

    expect(core.loadConfig).toHaveBeenCalled();
    expect(mockBuildIndexAtomic).toHaveBeenCalled();
    expect(logOutput[0]).toContain('Index built: 42 nodes, 7 edges');
  });

  it('accepts --root flag', async () => {
    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'ollama', model: 'nomic-embed-text', apiKey: '' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);
    mockBuildIndexAtomic.mockResolvedValue({ nodeCount: 10, edgeCount: 3 });

    const run = await getRun();
    await run(['build', '--root', '/custom/root']);

    const configArg = mockBuildIndexAtomic.mock.calls[0][0];
    expect(configArg).toBe(resolve('/custom/root'));
  });
});

describe('cmdRebuild', () => {
  it('removes old index and rebuilds', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);
    mockBuildIndexAtomic.mockResolvedValue({ nodeCount: 99, edgeCount: 15 });

    const run = await getRun();
    await run(['rebuild']);

    expect(fs.existsSync).toHaveBeenCalled();
    expect(fs.rmSync).toHaveBeenCalled();
    expect(mockBuildIndexAtomic).toHaveBeenCalled();
    expect(logOutput[0]).toContain('Index rebuilt: 99 nodes, 15 edges');
  });

  it('works even if no previous index exists', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockImplementation(() => false);
    vi.mocked(fs.rmSync).mockClear();

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);
    mockBuildIndexAtomic.mockResolvedValue({ nodeCount: 1, edgeCount: 0 });

    const run = await getRun();
    await run(['rebuild']);

    expect(vi.mocked(fs.rmSync)).not.toHaveBeenCalled();
    expect(logOutput[0]).toContain('Index rebuilt');
  });
});

describe('cmdQuery', () => {
  it('exits if no task provided', async () => {
    const run = await getRun();
    await run(['query']);
    expect(exitCode).toBe(1);
    expect(errorOutput.join(' ')).toContain('Usage');
  });

  it('exits if no index found', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(false);

    const run = await getRun();
    await run(['query', 'find the auth logic']);
    expect(exitCode).toBe(1);
    expect(errorOutput.join(' ')).toContain('No index found');
  });

  it('queries with defaults and prints context output', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreHasStaleFiles.mockReturnValue(false);
    mockRetrieverQuery.mockResolvedValue([
      { file: 'src/auth.ts', startLine: 10, endLine: 25, name: 'login', score: 0.95, body: 'function login() {}' },
    ]);

    const run = await getRun();
    await run(['query', 'find the auth logic']);

    expect(mockRetrieverQuery).toHaveBeenCalledWith('find the auth logic', {
      file: undefined,
      line: undefined,
      tokens: undefined,
    });
    expect(mockStoreClose).toHaveBeenCalled();
    expect(logOutput[0]).toContain('src/auth.ts');
    expect(logOutput[0]).toContain('0.95');
  });

  it('warns when index is stale', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreHasStaleFiles.mockReturnValue(true);
    mockRetrieverQuery.mockResolvedValue([]);

    const run = await getRun();
    await run(['query', 'something']);

    expect(warnOutput.join(' ')).toContain('Some files have been modified');
  });

  it('supports --file, --line, --tokens flags', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreHasStaleFiles.mockReturnValue(false);
    mockRetrieverQuery.mockResolvedValue([]);

    const run = await getRun();
    await run(['query', 'my task', '--file', 'src/index.ts', '--line', '42', '--tokens', '500']);

    expect(mockRetrieverQuery).toHaveBeenCalledWith('my task', {
      file: 'src/index.ts',
      line: 42,
      tokens: 500,
    });
  });

  it('supports --format json', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreHasStaleFiles.mockReturnValue(false);
    mockRetrieverQuery.mockResolvedValue([
      { file: 'src/auth.ts', startLine: 1, endLine: 5, name: null, score: 0.8, body: 'code' },
    ]);

    const run = await getRun();
    await run(['query', 'task', '--format', 'json']);

    const jsonLog = logOutput[logOutput.length - 1];
    const parsed = JSON.parse(jsonLog);
    expect(parsed[0].file).toBe('src/auth.ts');
  });

  it('supports --format manifest', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: 'sk-test' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreHasStaleFiles.mockReturnValue(false);
    mockRetrieverQuery.mockResolvedValue([
      { file: 'src/auth.ts', startLine: 10, endLine: 25, name: null, score: 0.9, body: 'code' },
    ]);

    const run = await getRun();
    await run(['query', 'task', '--format', 'manifest']);

    expect(logOutput[0]).toBe('src/auth.ts:10-25');
  });
});

describe('cmdStatus', () => {
  it('reports no index when directory missing', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(false);

    const run = await getRun();
    await run(['status']);

    expect(logOutput[0]).toContain('No index found');
  });

  it('prints stats and current status', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreGetStats.mockReturnValue({
      fileCount: 10,
      nodeCount: 100,
      edgeCount: 45,
      lastBuilt: 1700000000000,
    });
    mockStoreHasStaleFiles.mockReturnValue(false);

    const run = await getRun();
    await run(['status']);

    expect(logOutput[0]).toBe('Files:  10');
    expect(logOutput[1]).toBe('Nodes:  100');
    expect(logOutput[2]).toBe('Edges:  45');
    expect(logOutput[3]).toContain('2023');
    expect(logOutput[4]).toBe('Status: current');
    expect(mockStoreClose).toHaveBeenCalled();
  });

  it('shows stale status', async () => {
    const fs = await import('node:fs');
    vi.mocked(fs.existsSync).mockReturnValue(true);

    const storeStub = makeStoreStub();
    mockStoreOpen.mockReturnValue(storeStub);
    mockStoreGetStats.mockReturnValue({
      fileCount: 1,
      nodeCount: 1,
      edgeCount: 0,
      lastBuilt: null,
    });
    mockStoreHasStaleFiles.mockReturnValue(true);

    const run = await getRun();
    await run(['status']);

    expect(logOutput[3]).toBe('Built:  unknown');
    expect(logOutput[4]).toBe('Status: STALE — some files modified since build');
  });
});

describe('cmdMcp', () => {
  it('dynamically imports and starts MCP server', async () => {
    const run = await getRun();
    await run(['mcp']);

    expect(mockMcpStart).toHaveBeenCalled();
  });
});

describe('createEmbedder (via build command)', () => {
  it('throws for unknown provider', async () => {
    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'unknown' as any, model: 'foo', apiKey: '' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const run = await getRun();
    await run(['build']);

    expect(errorOutput.join(' ')).toContain('Unknown embedding provider');
    expect(exitCode).toBe(1);
  });

  it('throws when openai provider has no apiKey', async () => {
    const core = await import('@cc-optimize/core');
    vi.mocked(core.loadConfig).mockReturnValue({
      embeddings: { provider: 'openai', model: 'text-embedding-3-small', apiKey: '' },
      index: { blockSplitThreshold: 60, diffusionAlpha: 0.3, defaultTokenBudget: 8000, ignore: [] },
    } as any);

    const run = await getRun();
    await run(['build']);

    expect(errorOutput.join(' ')).toContain('OpenAI API key not set');
    expect(exitCode).toBe(1);
  });
});
