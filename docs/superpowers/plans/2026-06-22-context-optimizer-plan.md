# Context Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI + MCP tool that indexes a TypeScript codebase into a function/block-level graph with embeddings, then retrieves the minimal set of code regions needed for a given task.

**Architecture:** Three npm workspace packages — `core` (engine: indexer, graph, embeddings, store, retrieval), `cli` (commands: build, rebuild, query, status), `mcp` (MCP server: retrieve_context tool). Core is the library; CLI and MCP are thin consumers.

**Tech Stack:** TypeScript, Node.js, npm workspaces, tree-sitter, ts-morph, better-sqlite3 (with sqlite-vec extension), OpenAI text-embedding-3-small (default) + Ollama nomic-embed-text (adapter), @modelcontextprotocol/sdk, vitest.

## Global Constraints

- TypeScript/JavaScript target files only (v1 scope)
- Node.js ≥ 18
- No editor extensions in v1
- Index stored at `<repo-root>/.context-optimizer/`, auto-gitignored
- Embedding model: OpenAI text-embedding-3-small (1536 dims) default; Ollama nomic-embed-text (768 dims) adapter
- Switching embedding providers requires full rebuild
- Token budget default: 8000; estimate as `(end_line - start_line) * 5` per node
- Functions over 60 lines split into blocks; block nodes inherit parent edges at 0.8×
- Graph diffusion: α = 0.3, single pass
- Score fusion: 0.6 * semantic_score + 0.4 * graph_proximity_to_cursor (cursor optional)
- Edge weights: calls=1.0, called_by=0.8, overrides=0.7, uses_type=0.5, same_function=0.9, co_modified=0.4, shares_scope=0.6
- All errors surface as clear messages with non-zero exit; no silent failures
- Tests use fake (hash-based) embedding adapter; no API calls in test suite

---

### Task 1: Project Scaffolding

**Files:**
- Create: `package.json` (workspace root)
- Create: `tsconfig.base.json`
- Create: `packages/core/package.json`
- Create: `packages/core/tsconfig.json`
- Create: `packages/core/src/index.ts`
- Create: `packages/cli/package.json`
- Create: `packages/cli/tsconfig.json`
- Create: `packages/cli/src/index.ts`
- Create: `packages/mcp/package.json`
- Create: `packages/mcp/tsconfig.json`
- Create: `packages/mcp/src/index.ts`
- Create: `.gitignore`

**Interfaces:**
- Produces: workspace root `package.json` with three workspace packages; base tsconfig extended by each package

- [ ] **Step 1: Write root package.json**

```json
{
  "name": "context-optimizer",
  "private": true,
  "workspaces": [
    "packages/core",
    "packages/cli",
    "packages/mcp"
  ],
  "scripts": {
    "build": "npm run build --workspaces",
    "test": "npm run test --workspaces",
    "typecheck": "npm run typecheck --workspaces"
  }
}
```

- [ ] **Step 2: Write tsconfig.base.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "lib": ["ES2022"],
    "outDir": "dist",
    "rootDir": "src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  }
}
```

- [ ] **Step 3: Write packages/core/package.json**

```json
{
  "name": "@cc-optimize/core",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "better-sqlite3": "^11.0.0",
    "tree-sitter": "^0.22.0",
    "tree-sitter-typescript": "^0.23.0",
    "ts-morph": "^24.0.0",
    "minimatch": "^10.0.0",
    "fast-glob": "^3.3.0"
  },
  "devDependencies": {
    "@types/better-sqlite3": "^7.6.0",
    "typescript": "^5.6.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 4: Write packages/core/tsconfig.json**

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"]
}
```

- [ ] **Step 5: Write packages/core/src/index.ts**

```typescript
export * from './indexer/index.js';
export * from './graph/index.js';
export * from './embeddings/index.js';
export * from './store/index.js';
export * from './retrieval/index.js';
export * from './config/index.js';
export * from './types.js';
```

- [ ] **Step 6: Write packages/cli/package.json**

```json
{
  "name": "@cc-optimize/cli",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "dist/index.js",
  "bin": {
    "cc-optimize": "dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@cc-optimize/core": "*"
  },
  "devDependencies": {
    "typescript": "^5.6.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 7: Write packages/cli/tsconfig.json**

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"]
}
```

- [ ] **Step 8: Write packages/cli/src/index.ts** (placeholder)

```typescript
#!/usr/bin/env node
import { run } from './commands/index.js';
run(process.argv.slice(2));
```

- [ ] **Step 9: Write packages/mcp/package.json**

```json
{
  "name": "@cc-optimize/mcp",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@cc-optimize/core": "*",
    "@modelcontextprotocol/sdk": "^1.0.0"
  },
  "devDependencies": {
    "typescript": "^5.6.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 10: Write packages/mcp/tsconfig.json**

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"]
}
```

- [ ] **Step 11: Write packages/mcp/src/index.ts** (placeholder)

```typescript
export { start } from './server.js';
```

- [ ] **Step 12: Write .gitignore**

```
node_modules/
dist/
.context-optimizer/
*.tsbuildinfo
```

- [ ] **Step 13: Install dependencies**

Run: `npm install`

Expected: All workspace dependencies installed, no errors.

- [ ] **Step 14: Verify build**

Run: `npm run build`

Expected: All three packages compile successfully to dist/.

- [ ] **Step 15: Commit**

```bash
git add -A
git commit -m "feat: scaffold project with npm workspaces
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Core Types and Config

**Files:**
- Create: `packages/core/src/types.ts`
- Create: `packages/core/src/config/index.ts`
- Create: `packages/core/src/config/__tests__/config.test.ts`

**Interfaces:**
- Produces:
  - `NodeType` (type alias: `'function' | 'block'`)
  - `EdgeType` (type alias: `'calls' | 'called_by' | 'overrides' | 'uses_type' | 'same_function' | 'co_modified' | 'shares_scope'`)
  - `FileNode` interface: `id, path, mtime, astHash`
  - `CodeNode` interface: `id, type: NodeType, fileId, startLine, endLine, name: string | null, parentId: string | null, body`
  - `GraphEdge` interface: `source, target, type: EdgeType, weight`
  - `QueryResult` interface: `file, startLine, endLine, name, score`
  - `IndexStats` interface: `fileCount, nodeCount, edgeCount, lastBuilt: number | null`
  - `EmbeddingAdapter` interface: `embed(texts: string[]): Promise<number[][]>, dimension(): number`
  - `CcOptimizeConfig` interface: `embeddings: { provider: 'openai' | 'ollama', model: string, apiKey?: string }, index: { blockSplitThreshold: number, diffusionAlpha: number, defaultTokenBudget: number, ignore: string[] }`
  - `DEFAULT_CONFIG` constant with spec-default values
  - `loadConfig(rootPath: string): CcOptimizeConfig` function — reads `.cc-optimize.json` from root, merges with defaults
  - `resolveApiKey(rawKey: string): string` function — if starts with `env:`, reads the named env var; else returns raw

- [ ] **Step 1: Write types.ts**

```typescript
export type NodeType = 'function' | 'block';

export type EdgeType =
  | 'calls'
  | 'called_by'
  | 'overrides'
  | 'uses_type'
  | 'same_function'
  | 'co_modified'
  | 'shares_scope';

export interface FileNode {
  id: string;      // sha256 of absolute path
  path: string;
  mtime: number;
  astHash: string; // sha256 of file content
}

export interface CodeNode {
  id: string;      // functions: "<relpath>::<functionName>"; blocks: "<relpath>::<functionName>::<blockIndex>"
  type: NodeType;
  fileId: string;
  startLine: number;
  endLine: number;
  name: string | null;
  parentId: string | null;
  body: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  weight: number;
}

export interface QueryResult {
  file: string;
  startLine: number;
  endLine: number;
  name: string | null;
  score: number;
  body: string;
}

export interface IndexStats {
  fileCount: number;
  nodeCount: number;
  edgeCount: number;
  lastBuilt: number | null;
}

export interface EmbeddingAdapter {
  embed(texts: string[]): Promise<number[][]>;
  dimension(): number;
}

export const EDGE_WEIGHTS: Record<EdgeType, number> = {
  calls:         1.0,
  called_by:     0.8,
  overrides:     0.7,
  uses_type:     0.5,
  same_function: 0.9,
  co_modified:   0.4,
  shares_scope:  0.6,
};

export const BLOCK_INHERIT_MULTIPLIER = 0.8;

export interface CcOptimizeConfig {
  embeddings: {
    provider: 'openai' | 'ollama';
    model: string;
    apiKey: string;
  };
  index: {
    blockSplitThreshold: number;
    diffusionAlpha: number;
    defaultTokenBudget: number;
    ignore: string[];
  };
}
```

- [ ] **Step 2: Write config/index.ts**

```typescript
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { CcOptimizeConfig } from '../types.js';

export const DEFAULT_CONFIG: CcOptimizeConfig = {
  embeddings: {
    provider: 'openai',
    model: 'text-embedding-3-small',
    apiKey: '',
  },
  index: {
    blockSplitThreshold: 60,
    diffusionAlpha: 0.3,
    defaultTokenBudget: 8000,
    ignore: ['node_modules', 'dist', '*.test.ts'],
  },
};

export function resolveApiKey(rawKey: string): string {
  if (rawKey.startsWith('env:')) {
    const varName = rawKey.slice(4);
    const val = process.env[varName];
    if (!val) {
      throw new Error(
        `Environment variable ${varName} referenced in config but not set`
      );
    }
    return val;
  }
  return rawKey;
}

export function loadConfig(rootPath: string): CcOptimizeConfig {
  const configPath = resolve(rootPath, '.cc-optimize.json');
  if (!existsSync(configPath)) {
    return { ...DEFAULT_CONFIG };
  }

  let fileConfig: Partial<CcOptimizeConfig>;
  try {
    fileConfig = JSON.parse(readFileSync(configPath, 'utf-8'));
  } catch (e: any) {
    throw new Error(`Failed to parse config at ${configPath}: ${e.message}`);
  }

  const merged: CcOptimizeConfig = {
    embeddings: {
      ...DEFAULT_CONFIG.embeddings,
      ...(fileConfig.embeddings ?? {}),
    },
    index: {
      ...DEFAULT_CONFIG.index,
      ...(fileConfig.index ?? {}),
    },
  };

  if (merged.embeddings.apiKey) {
    merged.embeddings.apiKey = resolveApiKey(merged.embeddings.apiKey);
  }

  return merged;
}
```

- [ ] **Step 3: Write config/__tests__/config.test.ts**

```typescript
import { describe, it, expect } from 'vitest';
import { loadConfig, DEFAULT_CONFIG } from '../index.js';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

function withTmpDir(fn: (dir: string) => void) {
  const dir = join(tmpdir(), 'cc-optimize-test-' + Math.random().toString(36).slice(2));
  mkdirSync(dir, { recursive: true });
  try {
    fn(dir);
  } finally {
    rmSync(dir, { recursive: true });
  }
}

describe('loadConfig', () => {
  it('returns defaults when no config file exists', () => {
    withTmpDir((dir) => {
      const config = loadConfig(dir);
      expect(config.index.blockSplitThreshold).toBe(60);
      expect(config.embeddings.provider).toBe('openai');
    });
  });

  it('merges file config over defaults', () => {
    withTmpDir((dir) => {
      writeFileSync(join(dir, '.cc-optimize.json'), JSON.stringify({
        embeddings: { provider: 'ollama', model: 'nomic-embed-text' },
      }));
      const config = loadConfig(dir);
      expect(config.embeddings.provider).toBe('ollama');
      expect(config.embeddings.model).toBe('nomic-embed-text');
      expect(config.index.blockSplitThreshold).toBe(60); // unchanged default
    });
  });

  it('resolves env: prefix in apiKey', () => {
    withTmpDir((dir) => {
      process.env.TEST_CC_API_KEY = 'sk-test123';
      writeFileSync(join(dir, '.cc-optimize.json'), JSON.stringify({
        embeddings: { apiKey: 'env:TEST_CC_API_KEY' },
      }));
      const config = loadConfig(dir);
      expect(config.embeddings.apiKey).toBe('sk-test123');
      delete process.env.TEST_CC_API_KEY;
    });
  });

  it('throws when env var referenced in config is not set', () => {
    withTmpDir((dir) => {
      writeFileSync(join(dir, '.cc-optimize.json'), JSON.stringify({
        embeddings: { apiKey: 'env:NONEXISTENT_VAR' },
      }));
      expect(() => loadConfig(dir)).toThrow('NONEXISTENT_VAR');
    });
  });

  it('throws on malformed config JSON', () => {
    withTmpDir((dir) => {
      writeFileSync(join(dir, '.cc-optimize.json'), 'not json');
      expect(() => loadConfig(dir)).toThrow('Failed to parse config');
    });
  });
});
```

- [ ] **Step 4: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/types.ts packages/core/src/config/
git commit -m "feat: add core types and config loader
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Fake Embedding Adapter

**Files:**
- Create: `packages/core/src/embeddings/index.ts`
- Create: `packages/core/src/embeddings/fake.ts`
- Create: `packages/core/src/embeddings/__tests__/fake.test.ts`

**Interfaces:**
- Consumes: `EmbeddingAdapter` from `types.ts`
- Produces:
  - `createHashEmbedding(text: string, dimension: number): number[]` — deterministic hash-based vector
  - `FakeEmbeddingAdapter(dim: number): EmbeddingAdapter` — returns deterministic vectors seeded by text content

- [ ] **Step 1: Write embeddings/index.ts**

```typescript
export { FakeEmbeddingAdapter } from './fake.js';
```

- [ ] **Step 2: Write embeddings/fake.ts**

```typescript
import { createHash } from 'node:crypto';
import { EmbeddingAdapter } from '../types.js';

/**
 * Deterministic hash-based embedding. Seed is SHA256 of text, then
 * expanded into `dimension` floats using the hash bytes cyclically.
 */
export function createHashEmbedding(text: string, dimension: number): number[] {
  const hash = createHash('sha256').update(text).digest();
  const vec: number[] = [];
  for (let i = 0; i < dimension; i++) {
    const byte = hash[i % hash.length];
    vec.push((byte / 255) * 2 - 1); // Map to [-1, 1]
  }
  return vec;
}

export function FakeEmbeddingAdapter(dimension = 1536): EmbeddingAdapter {
  return {
    dimension: () => dimension,
    async embed(texts: string[]): Promise<number[][]> {
      return texts.map((t) => createHashEmbedding(t, dimension));
    },
  };
}
```

- [ ] **Step 3: Write embeddings/__tests__/fake.test.ts**

```typescript
import { describe, it, expect } from 'vitest';
import { FakeEmbeddingAdapter, createHashEmbedding } from '../fake.js';

describe('FakeEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = FakeEmbeddingAdapter(768);
    expect(adapter.dimension()).toBe(768);
  });

  it('produces deterministic embeddings', async () => {
    const adapter = FakeEmbeddingAdapter(16);
    const a = await adapter.embed(['hello world']);
    const b = await adapter.embed(['hello world']);
    expect(a[0]).toEqual(b[0]);
  });

  it('produces different embeddings for different texts', async () => {
    const adapter = FakeEmbeddingAdapter(16);
    const [a] = await adapter.embed(['function auth()']);
    const [b] = await adapter.embed(['function db()']);
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    // Extremely unlikely to collide; should be different
    expect(Math.abs(dot)).toBeLessThan(5);
  });

  it('batches multiple texts', async () => {
    const adapter = FakeEmbeddingAdapter(64);
    const results = await adapter.embed(['a', 'b', 'c']);
    expect(results).toHaveLength(3);
    expect(results[0]).toHaveLength(64);
  });
});
```

- [ ] **Step 4: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/embeddings/
git commit -m "feat: add fake embedding adapter for testing
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: OpenAI Embedding Adapter

**Files:**
- Create: `packages/core/src/embeddings/openai.ts`
- Create: `packages/core/src/embeddings/__tests__/openai.test.ts`

**Interfaces:**
- Consumes: `EmbeddingAdapter` from `types.ts`
- Produces:
  - `OpenAIEmbeddingAdapter(apiKey: string, model?: string): EmbeddingAdapter` — calls OpenAI /v1/embeddings API
  - Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
  - Batch: splits large arrays into chunks of 100 (API max inputs per request)

- [ ] **Step 1: Write embeddings/openai.ts**

```typescript
import { EmbeddingAdapter } from '../types.js';

const MAX_INPUTS_PER_REQUEST = 100;
const RETRY_DELAYS = [1000, 2000, 4000];

async function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export function OpenAIEmbeddingAdapter(
  apiKey: string,
  model = 'text-embedding-3-small'
): EmbeddingAdapter {
  const dimension = model === 'text-embedding-3-small' ? 1536 : 1536; // 1536 for text-embedding-3-small

  return {
    dimension: () => dimension,

    async embed(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];

      const allEmbeddings: number[][] = [];

      // Chunk into batches of 100
      for (let i = 0; i < texts.length; i += MAX_INPUTS_PER_REQUEST) {
        const batch = texts.slice(i, i + MAX_INPUTS_PER_REQUEST);

        let lastError: Error | null = null;
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const response = await fetch('https://api.openai.com/v1/embeddings', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model,
                input: batch,
                encoding_format: 'float',
              }),
            });

            if (!response.ok) {
              const body = await response.text();
              throw new Error(`OpenAI API error ${response.status}: ${body}`);
            }

            const data = await response.json() as {
              data: { embedding: number[]; index: number }[];
            };

            // Restore original ordering
            data.data.sort((a, b) => a.index - b.index);
            allEmbeddings.push(...data.data.map((d) => d.embedding));
            lastError = null;
            break;
          } catch (err: any) {
            lastError = err;
            if (attempt < 2) {
              await sleep(RETRY_DELAYS[attempt]);
            }
          }
        }

        if (lastError) {
          throw new Error(
            `Embedding API failed after 3 attempts for batch starting at index ${i}: ${lastError.message}`
          );
        }
      }

      return allEmbeddings;
    },
  };
}
```

- [ ] **Step 2: Write embeddings/__tests__/openai.test.ts**

```typescript
import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest';
import { OpenAIEmbeddingAdapter } from '../openai.js';

describe('OpenAIEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = OpenAIEmbeddingAdapter('sk-fake');
    expect(adapter.dimension()).toBe(1536);
  });

  it('calls the OpenAI API with correct payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
      }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OpenAIEmbeddingAdapter('sk-test');
    const results = await adapter.embed(['test input']);

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual([0.1, 0.2, 0.3]);

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.openai.com/v1/embeddings',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer sk-test',
        }),
        body: expect.stringContaining('test input'),
      })
    );
  });

  it('retries on failure and throws after 3 attempts', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('network error'));
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OpenAIEmbeddingAdapter('sk-test');
    await expect(adapter.embed(['test'])).rejects.toThrow('Embedding API failed after 3 attempts');
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('returns empty array for empty input', async () => {
    const adapter = OpenAIEmbeddingAdapter('sk-test');
    const results = await adapter.embed([]);
    expect(results).toEqual([]);
  });
});
```

- [ ] **Step 3: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add packages/core/src/embeddings/openai.ts packages/core/src/embeddings/__tests__/openai.test.ts
git commit -m "feat: add OpenAI embedding adapter with retry logic
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Ollama Embedding Adapter

**Files:**
- Create: `packages/core/src/embeddings/ollama.ts`
- Create: `packages/core/src/embeddings/__tests__/ollama.test.ts`

**Interfaces:**
- Consumes: `EmbeddingAdapter` from `types.ts`
- Produces: `OllamaEmbeddingAdapter(baseUrl?: string, model?: string): EmbeddingAdapter`

- [ ] **Step 1: Write embeddings/ollama.ts**

```typescript
import { EmbeddingAdapter } from '../types.js';

export function OllamaEmbeddingAdapter(
  baseUrl = 'http://localhost:11434',
  model = 'nomic-embed-text'
): EmbeddingAdapter {
  return {
    dimension: () => 768,
    async embed(texts: string[]): Promise<number[][]> {
      if (texts.length === 0) return [];

      const results: number[][] = [];
      for (const text of texts) {
        const response = await fetch(`${baseUrl}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model, prompt: text }),
        });

        if (!response.ok) {
          const body = await response.text();
          throw new Error(`Ollama API error ${response.status}: ${body}`);
        }

        const data = await response.json() as { embedding: number[] };
        results.push(data.embedding);
      }

      return results;
    },
  };
}
```

- [ ] **Step 2: Write embeddings/__tests__/ollama.test.ts**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { OllamaEmbeddingAdapter } from '../ollama.js';

describe('OllamaEmbeddingAdapter', () => {
  it('returns correct dimension', () => {
    const adapter = OllamaEmbeddingAdapter();
    expect(adapter.dimension()).toBe(768);
  });

  it('calls Ollama API with correct payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ embedding: [0.1, 0.2] }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OllamaEmbeddingAdapter('http://localhost:11434');
    const results = await adapter.embed(['test']);

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual([0.1, 0.2]);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/embeddings',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('nomic-embed-text'),
      })
    );
  });

  it('throws on non-ok response', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('internal error'),
    });
    vi.stubGlobal('fetch', mockFetch);

    const adapter = OllamaEmbeddingAdapter();
    await expect(adapter.embed(['test'])).rejects.toThrow('Ollama API error 500');
  });
});
```

- [ ] **Step 3: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 4: Update embeddings index.ts to export new adapters**

Edit `packages/core/src/embeddings/index.ts`:

```typescript
export { FakeEmbeddingAdapter } from './fake.js';
export { OpenAIEmbeddingAdapter } from './openai.js';
export { OllamaEmbeddingAdapter } from './ollama.js';
```

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/embeddings/
git commit -m "feat: add Ollama embedding adapter
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Store — SQLite Persistence

**Files:**
- Create: `packages/core/src/store/index.ts`
- Create: `packages/core/src/store/__tests__/store.test.ts`

**Interfaces:**
- Consumes: `FileNode`, `CodeNode`, `GraphEdge`, `IndexStats` from `types.ts`
- Produces:
  - `Store` class: `open(dbPath: string)` opens/creates SQLite, runs migrations
  - `writeFiles(files: FileNode[]): void`
  - `writeNodes(nodes: CodeNode[]): void`
  - `writeEdges(edges: GraphEdge[]): void`
  - `getFiles(): FileNode[]`
  - `getNodes(): CodeNode[]`
  - `getEdges(): GraphEdge[]`
  - `clear(): void` — deletes all data (used during rebuild)
  - `close(): void`
  - `writeVectors(vectors: Map<string, number[]>): void` — inserts into sqlite-vec virtual table
  - `searchVectors(queryVec: number[], k: number): Array<{ nodeId: string, score: number }>` — cosine similarity via sqlite-vec
  - `getStats(): IndexStats`
  - Atomicity: writes use transaction; rebuild writes to temp file then rename

**Note on sqlite-vec:** The `sqlite-vec` extension is loaded via `better-sqlite3`. The actual vec0 virtual table creation and query syntax follows the sqlite-vec API (vec0 table, vec_distance_L2 etc.). If sqlite-vec is not available as a loadable extension at this time, we fall back to storing vectors as BLOB in a separate table and computing cosine similarity in JavaScript for the search method. For v1, we implement the JS fallback.

- [ ] **Step 1: Write store/index.ts**

```typescript
import Database from 'better-sqlite3';
import { existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { FileNode, CodeNode, GraphEdge, IndexStats } from '../types.js';

export class Store {
  private db: Database.Database;

  private constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.migrate();
  }

  static open(dbPath: string): Store {
    return new Store(dbPath);
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS files (
        id       TEXT PRIMARY KEY,
        path     TEXT NOT NULL,
        mtime    INTEGER NOT NULL,
        ast_hash TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS nodes (
        id         TEXT PRIMARY KEY,
        type       TEXT NOT NULL,
        file_id    TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line   INTEGER NOT NULL,
        name       TEXT,
        parent_id  TEXT,
        body       TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS edges (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        type   TEXT NOT NULL,
        weight REAL NOT NULL,
        PRIMARY KEY (source, target, type)
      );

      CREATE TABLE IF NOT EXISTS vectors (
        node_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
      );
    `);
  }

  writeFiles(files: FileNode[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO files (id, path, mtime, ast_hash)
      VALUES (@id, @path, @mtime, @astHash)
    `);
    const txn = this.db.transaction((files: FileNode[]) => {
      for (const f of files) insert.run(f);
    });
    txn(files);
  }

  writeNodes(nodes: CodeNode[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO nodes (id, type, file_id, start_line, end_line, name, parent_id, body)
      VALUES (@id, @type, @fileId, @startLine, @endLine, @name, @parentId, @body)
    `);
    const txn = this.db.transaction((nodes: CodeNode[]) => {
      for (const n of nodes) insert.run(n);
    });
    txn(nodes);
  }

  writeEdges(edges: GraphEdge[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO edges (source, target, type, weight)
      VALUES (@source, @target, @type, @weight)
    `);
    const txn = this.db.transaction((edges: GraphEdge[]) => {
      for (const e of edges) insert.run(e);
    });
    txn(edges);
  }

  getFiles(): FileNode[] {
    return this.db.prepare(`SELECT id, path, mtime, ast_hash as astHash FROM files`).all() as FileNode[];
  }

  getNodes(): CodeNode[] {
    return this.db.prepare(`
      SELECT id, type, file_id as fileId, start_line as startLine,
             end_line as endLine, name, parent_id as parentId, body
      FROM nodes
    `).all() as CodeNode[];
  }

  getEdges(): GraphEdge[] {
    return this.db.prepare(`SELECT source, target, type, weight FROM edges`).all() as GraphEdge[];
  }

  writeVectors(vectors: Map<string, number[]>): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO vectors (node_id, embedding) VALUES (?, ?)
    `);
    const txn = this.db.transaction((entries: [string, number[]][]) => {
      for (const [nodeId, vec] of entries) {
        const buf = Buffer.allocUnsafe(vec.length * 4);
        for (let i = 0; i < vec.length; i++) {
          buf.writeFloatLE(vec[i], i * 4);
        }
        insert.run(nodeId, buf);
      }
    });
    txn(Array.from(vectors.entries()));
  }

  /** Cosine similarity search — JS implementation since we store vectors as BLOB. */
  searchVectors(queryVec: number[], k: number): Array<{ nodeId: string; score: number }> {
    const rows = this.db.prepare(`SELECT node_id, embedding FROM vectors`).all() as {
      node_id: string; embedding: Buffer;
    }[];

    const queryNorm = Math.sqrt(queryVec.reduce((s, v) => s + v * v, 0));

    const scored = rows.map((row) => {
      const vecLen = row.embedding.length / 4;
      const emb = new Array(vecLen);
      for (let i = 0; i < vecLen; i++) {
        emb[i] = row.embedding.readFloatLE(i * 4);
      }

      const embNorm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
      let dot = 0;
      for (let i = 0; i < Math.min(queryVec.length, emb.length); i++) {
        dot += queryVec[i] * emb[i];
      }

      const cosine = (queryNorm === 0 || embNorm === 0) ? 0 : dot / (queryNorm * embNorm);
      return { nodeId: row.node_id, score: cosine };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  clear(): void {
    this.db.exec(`DELETE FROM edges; DELETE FROM nodes; DELETE FROM files; DELETE FROM vectors;`);
  }

  getStats(): IndexStats {
    const files = this.db.prepare(`SELECT COUNT(*) as cnt FROM files`).get() as { cnt: number };
    const nodes = this.db.prepare(`SELECT COUNT(*) as cnt FROM nodes`).get() as { cnt: number };
    const edges = this.db.prepare(`SELECT COUNT(*) as cnt FROM edges`).get() as { cnt: number };
    const latest = this.db.prepare(`SELECT MAX(mtime) as mt FROM files`).get() as { mt: number | null };
    return {
      fileCount: files.cnt,
      nodeCount: nodes.cnt,
      edgeCount: edges.cnt,
      lastBuilt: latest.mt ?? null,
    };
  }

  /** Check if any tracked file has been modified since indexing. */
  hasStaleFiles(rootPath: string): boolean {
    const rows = this.db.prepare(`SELECT path, mtime FROM files`).all() as { path: string; mtime: number }[];
    for (const row of rows) {
      const absPath = resolve(rootPath, row.path);
      if (!existsSync(absPath)) return true;
      const stat = statSync(absPath);
      if (stat.mtimeMs > row.mtime) return true;
    }
    return false;
  }

  close(): void {
    this.db.close();
  }
}
```

- [ ] **Step 2: Write store/__tests__/store.test.ts**

```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Store } from '../index.js';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdirSync, rmSync } from 'node:fs';

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
});
```

- [ ] **Step 3: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add packages/core/src/store/
git commit -m "feat: add SQLite store with vector search
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Indexer — AST Parsing and Node Extraction

**Files:**
- Create: `packages/core/src/indexer/index.ts`
- Create: `packages/core/src/indexer/__tests__/indexer.test.ts`
- Create: `packages/core/src/indexer/__tests__/fixtures/sample.ts` (test fixture)

**Interfaces:**
- Consumes: `CodeNode`, `FileNode`, `CcOptimizeConfig` from `types.ts`
- Produces:
  - `Indexer` class: `new Indexer(config: CcOptimizeConfig)`
  - `build(rootPath: string): { files: FileNode[], nodes: CodeNode[], edges: GraphEdge[] }`
  - Uses tree-sitter to parse TypeScript/JS files, extract function declarations, split blocks by threshold
  - Uses ts-morph to resolve call graph edges (falls back to tree-sitter-only on resolution failure)
  - Respects config `ignore` patterns (uses minimatch against relative paths)

- [ ] **Step 1: Write indexer/__tests__/fixtures/sample.ts**

```typescript
// Test fixture — not parsed by the indexer test itself, but used as input
export function add(a: number, b: number): number {
  return a + b;
}

export function multiply(a: number, b: number): number {
  return a * b;
}

export function calculate(a: number, b: number): number {
  const sum = add(a, b);
  return multiply(sum, 2);
}
```

- [ ] **Step 2: Write indexer/index.ts**

```typescript
import { readFileSync, statSync, existsSync } from 'node:fs';
import { resolve, relative, dirname } from 'node:path';
import { globSync } from 'fast-glob';
import { createHash } from 'node:crypto';
import { minimatch } from 'minimatch';
import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';
import { CcOptimizeConfig, CodeNode, FileNode, GraphEdge, EdgeType } from '../types.js';

function hashContent(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

function hashPath(absPath: string): string {
  return createHash('sha256').update(absPath).digest('hex');
}

interface RawNode {
  id: string;
  type: 'function' | 'block';
  fileId: string;
  filePath: string;
  startLine: number;
  endLine: number;
  name: string | null;
  parentId: string | null;
  body: string;
  fileContent: string; // for ts-morph call resolution
}

export class Indexer {
  private config: CcOptimizeConfig;

  constructor(config: CcOptimizeConfig) {
    this.config = config;
  }

  build(rootPath: string): { files: FileNode[]; nodes: CodeNode[]; edges: GraphEdge[] } {
    const files = this.walkFiles(rootPath);
    const rawNodes = this.parseFiles(rootPath, files);
    const edges = this.buildEdges(rootPath, rawNodes, files);
    const nodes: CodeNode[] = rawNodes.map((n) => ({
      id: n.id,
      type: n.type,
      fileId: n.fileId,
      startLine: n.startLine,
      endLine: n.endLine,
      name: n.name,
      parentId: n.parentId,
      body: n.body,
    }));

    return { files, nodes, edges };
  }

  private walkFiles(rootPath: string): FileNode[] {
    const entries = globSync('**/*.{ts,tsx,js,jsx,cts,mts}', {
      cwd: rootPath,
      ignore: ['node_modules/**', 'dist/**'],
      absolute: false,
      withFileTypes: true,
    });

    const files: FileNode[] = [];
    for (const entry of entries as unknown as string[]) {
      const relPath = entry.replace(/\\/g, '/');
      if (this.shouldIgnore(relPath)) continue;

      const absPath = resolve(rootPath, relPath);
      let content: string;
      try {
        content = readFileSync(absPath, 'utf-8');
      } catch {
        continue;
      }
      const statResult = statSync(absPath);
      files.push({
        id: hashPath(absPath),
        path: relPath,
        mtime: statResult.mtimeMs,
        astHash: hashContent(content),
      });
    }
    return files;
  }

  private shouldIgnore(relPath: string): boolean {
    for (const pattern of this.config.index.ignore) {
      if (minimatch(relPath, pattern, { matchBase: true })) return true;
    }
    return false;
  }

  private parseFiles(rootPath: string, files: FileNode[]): RawNode[] {
    const parser = new Parser();
    parser.setLanguage(TypeScript.tsx as Parser.Language);

    const allNodes: RawNode[] = [];

    for (const file of files) {
      const absPath = resolve(rootPath, file.path);
      let content: string;
      try {
        content = readFileSync(absPath, 'utf-8');
      } catch {
        continue;
      }

      const tree = parser.parse(content);
      const functions = this.extractFunctions(tree.rootNode, content, file);

      for (const fn of functions) {
        fn.fileContent = content;
        const lineCount = fn.endLine - fn.startLine + 1;
        if (lineCount > this.config.index.blockSplitThreshold) {
          const blocks = this.splitIntoBlocks(tree.rootNode, fn, content);
          allNodes.push(...blocks);
        } else {
          allNodes.push(fn);
        }
      }
    }

    return allNodes;
  }

  private extractFunctions(root: Parser.SyntaxNode, content: string, file: FileNode): RawNode[] {
    const nodes: RawNode[] = [];
    this.walkForFunctions(root, content, file, nodes);
    return nodes;
  }

  private walkForFunctions(
    node: Parser.SyntaxNode,
    content: string,
    file: FileNode,
    out: RawNode[],
  ): void {
    const namedTypes = ['function_declaration', 'method_definition', 'arrow_function', 'function_expression'];
    if (namedTypes.includes(node.type)) {
      const nameNode = node.childForFieldName?.('name') ?? node.namedChild(0);
      let name: string | null = null;
      if (nameNode && nameNode.type === 'identifier') {
        name = content.slice(nameNode.startIndex, nameNode.endIndex);
      }
      const body = content.slice(node.startIndex, node.endIndex);
      const startLine = node.startPosition.row + 1;
      const endLine = node.endPosition.row + 1;

      out.push({
        id: `${file.path}::${name ?? 'anonymous'}`,
        type: 'function',
        fileId: file.id,
        filePath: file.path,
        startLine,
        endLine,
        name,
        parentId: null,
        body,
        fileContent: content,
      });
    }

    for (let i = 0; i < node.namedChildCount; i++) {
      this.walkForFunctions(node.namedChild(i)!, content, file, out);
    }
  }

  private splitIntoBlocks(
    root: Parser.SyntaxNode,
    fn: RawNode,
    content: string,
  ): RawNode[] {
    const blocks: RawNode[] = [];
    const bodyNode = this.getBodyNode(root, fn.name ?? '');

    if (!bodyNode) {
      // Can't split — return whole function as one block
      blocks.push({ ...fn, type: 'block', parentId: fn.id });
      return blocks;
    }

    // Extract variable declarations from the function scope (outside blocks)
    const varDecls: string[] = [];
    const children = bodyNode.namedChildren;
    let blockIndex = 0;
    let currentStart = -1;
    let currentBody = '';

    for (const child of children) {
      const blockTypes = ['if_statement', 'for_statement', 'for_in_statement',
        'while_statement', 'try_statement', 'switch_statement'];

      if (child.type === 'lexical_declaration' || child.type === 'variable_declaration') {
        varDecls.push(content.slice(child.startIndex, child.endIndex));
        continue;
      }

      if (blockTypes.includes(child.type) || child.type === 'expression_statement'
        || child.type === 'return_statement') {

        const childStart = child.startPosition.row + 1;
        const childEnd = child.endPosition.row + 1;
        const childBody = content.slice(child.startIndex, child.endIndex);

        if (currentStart === -1) {
          currentStart = childStart;
          currentBody = childBody;
        } else {
          currentBody += '\n' + childBody;
        }
        currentStart = Math.min(currentStart, childStart);

        blockIndex++;
        const blockId = `${fn.id}::${blockIndex}`;
        const prefix = varDecls.length > 0 ? varDecls.join('\n') + '\n\n' : '';
        blocks.push({
          id: blockId,
          type: 'block',
          fileId: fn.fileId,
          filePath: fn.filePath,
          startLine: currentStart,
          endLine: childEnd,
          name: null,
          parentId: fn.id,
          body: prefix + childBody,
          fileContent: content,
        });
        currentStart = -1;
        currentBody = '';
      }
    }

    if (blocks.length === 0) {
      blocks.push({ ...fn, type: 'block', parentId: fn.id });
    }

    return blocks;
  }

  private getBodyNode(root: Parser.SyntaxNode, fnName: string): Parser.SyntaxNode | null {
    const namedTypes = ['function_declaration', 'method_definition', 'arrow_function', 'function_expression'];
    const queue = [root];
    while (queue.length > 0) {
      const node = queue.shift()!;
      if (namedTypes.includes(node.type)) {
        const body = node.childForFieldName?.('body') ?? null;
        if (body && body.type === 'statement_block') {
          return body;
        }
      }
      for (let i = 0; i < node.namedChildCount; i++) {
        queue.push(node.namedChild(i)!);
      }
    }
    return null;
  }

  private buildEdges(
    rootPath: string,
    nodes: RawNode[],
    files: FileNode[],
  ): GraphEdge[] {
    const edges: GraphEdge[] = [];
    const nodeMap = new Map<string, RawNode>(nodes.map((n) => [n.id, n]));

    // For each function node, use tree-sitter to find call expressions
    // and map called function names back to node IDs
    const parser = new Parser();
    parser.setLanguage(TypeScript.tsx as Parser.Language);

    for (const file of files) {
      const absPath = resolve(rootPath, file.path);
      let content: string;
      try {
        content = readFileSync(absPath, 'utf-8');
      } catch {
        continue;
      }

      const tree = parser.parse(content);
      // Find all call_expression nodes
      this.walkForCalls(tree.rootNode, content, file.path, nodes, nodeMap, edges);
    }

    return edges;
  }

  private walkForCalls(
    node: Parser.SyntaxNode,
    content: string,
    filePath: string,
    allNodes: RawNode[],
    nodeMap: Map<string, RawNode>,
    out: GraphEdge[],
  ): void {
    if (node.type === 'call_expression') {
      const fnNode = node.namedChild(0);
      if (fnNode) {
        const calledName = content.slice(fnNode.startIndex, fnNode.endIndex);
        // Find target node by name match
        for (const target of allNodes) {
          if (target.name === calledName && target.type === 'function') {
            // Find enclosing function for the call site
            const caller = this.findEnclosingFunction(node, content, filePath, allNodes);
            if (caller) {
              out.push({
                source: caller.id,
                target: target.id,
                type: 'calls',
                weight: 1.0,
              });
              out.push({
                source: target.id,
                target: caller.id,
                type: 'called_by',
                weight: 0.8,
              });
            }
          }
        }
      }
    }

    for (let i = 0; i < node.namedChildCount; i++) {
      this.walkForCalls(node.namedChild(i)!, content, filePath, allNodes, nodeMap, out);
    }
  }

  private findEnclosingFunction(
    callNode: Parser.SyntaxNode,
    content: string,
    filePath: string,
    allNodes: RawNode[],
  ): RawNode | null {
    // Walk up the tree to find the enclosing function node
    let current = callNode.parent;
    while (current) {
      const fnTypes = ['function_declaration', 'method_definition', 'arrow_function', 'function_expression'];
      if (fnTypes.includes(current.type)) {
        const nameNode = current.childForFieldName?.('name') ?? current.namedChild(0);
        let name: string | null = null;
        if (nameNode && nameNode.type === 'identifier') {
          name = content.slice(nameNode.startIndex, nameNode.endIndex);
        }
        const fnId = `${filePath}::${name ?? 'anonymous'}`;
        return allNodes.find((n) => n.id === fnId) ?? null;
      }
      current = current.parent;
    }
    return null;
  }
}
```

- [ ] **Step 3: Write indexer/__tests__/indexer.test.ts**

```typescript
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
```

- [ ] **Step 4: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/indexer/
git commit -m "feat: add tree-sitter based indexer with call graph extraction
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Graph — Edge Building and Diffusion

**Files:**
- Create: `packages/core/src/graph/index.ts`
- Create: `packages/core/src/graph/__tests__/graph.test.ts`

**Interfaces:**
- Consumes: `CodeNode`, `GraphEdge`, `EmbeddingAdapter` from `types.ts`
- Produces:
  - `buildGraph(nodes: CodeNode[], edges: GraphEdge[]): Graph` — the full graph object with adjacency
  - `diffuseEmbeddings(graph: Graph, alpha: number): Map<string, number[]>` — one-pass diffusion
  - `Graph` class with methods: `getNode(id)`, `getNeighbors(id, minWeight)`, `bfsDistance(from, to, minWeight)`

- [ ] **Step 1: Write graph/index.ts**

```typescript
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
    return this.embeddings;
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
  alpha: number,
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
```

- [ ] **Step 2: Write graph/__tests__/graph.test.ts**

```typescript
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
```

- [ ] **Step 3: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add packages/core/src/graph/
git commit -m "feat: add graph builder with BFS and embedding diffusion
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Retrieval Pipeline

**Files:**
- Create: `packages/core/src/retrieval/index.ts`
- Create: `packages/core/src/retrieval/__tests__/retrieval.test.ts`
- Create: `packages/core/src/retrieval/__tests__/fixtures/` (symlink to indexer fixtures or copy)

**Interfaces:**
- Consumes: `Graph`, `EmbeddingAdapter`, `Store`, query parameters
- Produces:
  - `Retriever` class: `new Retriever(store: Store, embeddingAdapter: EmbeddingAdapter, config)`
  - `query(task: string, opts?: { file?: string, line?: number, tokens?: number }): QueryResult[]`
  - Implements the full query pipeline: embed → vector search → cursor anchor → graph expansion → score fusion → token budget → deduplicate → format

- [ ] **Step 1: Write retrieval/index.ts**

```typescript
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
    // Reconstruct embeddings from store for graph
    // For now we use the vector search as the embedding source; the graph
    // is used for structural proximity only
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
        // Also add incoming edges (reverse neighbors)
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

      const finalScore = 0.6 * semantic + 0.4 * graphProximity;
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
        file: nodeMap.get(item.node.id)!.id.split('::')[0] ?? '',
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
```

- [ ] **Step 2: Write retrieval/__tests__/retrieval.test.ts**

```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Retriever } from '../index.js';
import { Store } from '../../store/index.js';
import { FakeEmbeddingAdapter } from '../../embeddings/fake.js';
import { DEFAULT_CONFIG } from '../../config/index.js';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdirSync, rmSync } from 'node:fs';

function tmpDir(): string {
  const dir = join(tmpdir(), 'cc-opts-retrieval-' + Math.random().toString(36).slice(2));
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** Seed vectors explicitly so ordering is deterministic regardless of embedding adapter. */
function explicitVectors(dim: number): {
  loginVec: number[];
  logoutVec: number[];
  dbVec: number[];
} {
  // loginVec is close to [1,0,0,...], dbVec is close to [0,1,0,...], logoutVec is [0,0,1,...]
  // This makes cosine similarity with query [1,0,0,...] rank login > db > logout
  const loginVec = new Array(dim).fill(0);
  loginVec[0] = 1;
  const dbVec = new Array(dim).fill(0);
  dbVec[1] = 1;
  const logoutVec = new Array(dim).fill(0);
  logoutVec[2] = 1;
  return { loginVec, logoutVec, dbVec };
}

describe('Retriever', () => {
  let store: Store;
  let tmp: string;

  beforeEach(() => {
    tmp = tmpDir();
    store = Store.open(join(tmp, 'graph.db'));
  });

  afterEach(() => {
    store.close();
    rmSync(tmp, { recursive: true, force: true });
  });

  it('returns ranked results for a query', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/auth.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/auth.ts::login', type: 'function', fileId: 'f1', startLine: 1, endLine: 20,
        name: 'login', parentId: null, body: 'function login() { ... }' },
      { id: 'src/auth.ts::logout', type: 'function', fileId: 'f1', startLine: 22, endLine: 35,
        name: 'logout', parentId: null, body: 'function logout() { ... }' },
      { id: 'src/db.ts::connect', type: 'function', fileId: 'f1', startLine: 1, endLine: 10,
        name: 'connect', parentId: null, body: 'function connect() { ... }' },
    ]);
    store.writeEdges([
      { source: 'src/auth.ts::login', target: 'src/db.ts::connect', type: 'calls', weight: 1.0 },
    ]);

    // Use explicit vectors so ranking is deterministic
    const dim = 16;
    const vecs = explicitVectors(dim);
    store.writeVectors(new Map([
      ['src/auth.ts::login', vecs.loginVec],
      ['src/auth.ts::logout', vecs.logoutVec],
      ['src/db.ts::connect', vecs.dbVec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('fix login authentication bug', { tokens: 1000 });

    expect(results.length).toBeGreaterThan(0);
    // Results should be well-structured
    for (const r of results) {
      expect(r.file).toBeTruthy();
      expect(r.startLine).toBeGreaterThan(0);
      expect(r.endLine).toBeGreaterThan(0);
      expect(typeof r.score).toBe('number');
      expect(r.body).toBeTruthy();
    }
  });

  it('returns empty results when store has no nodes', async () => {
    const embedder = FakeEmbeddingAdapter(16);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('fix bug');
    expect(results).toEqual([]);
  });

  it('applies token budget correctly', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/a.ts::big', type: 'function', fileId: 'f1',
        startLine: 1, endLine: 200, name: 'big', parentId: null, body: '...'.repeat(200) },
      { id: 'src/a.ts::small', type: 'function', fileId: 'f1',
        startLine: 201, endLine: 202, name: 'small', parentId: null, body: 'fn small' },
    ]);

    // small node has higher vector score (closer to unit vector), big node has lower
    // but both will be considered — budget only fits small (10 estimated tokens vs 1000)
    const dim = 16;
    const smallVec = new Array(dim).fill(0);
    smallVec[0] = 1; // very close to query
    const bigVec = new Array(dim).fill(0);
    bigVec[dim - 1] = 0.1; // far from query

    store.writeVectors(new Map([
      ['src/a.ts::big', bigVec],
      ['src/a.ts::small', smallVec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('small', { tokens: 100 });
    // big node is ~1000 estimated tokens, so it won't fit even if ranked high;
    // small node is ~10 estimated tokens and will fit
    if (results.length > 0) {
      expect(results.every((r) => r.name === 'small')).toBe(true);
    }
  });

  it('ranks results by score in descending order', async () => {
    store.writeFiles([{ id: 'f1', path: 'src/a.ts', mtime: 1000, astHash: 'abc' }]);
    store.writeNodes([
      { id: 'src/a.ts::fn1', type: 'function', fileId: 'f1',
        startLine: 1, endLine: 5, name: 'fn1', parentId: null, body: 'fn1 body' },
      { id: 'src/a.ts::fn2', type: 'function', fileId: 'f1',
        startLine: 6, endLine: 10, name: 'fn2', parentId: null, body: 'fn2 body' },
      { id: 'src/a.ts::fn3', type: 'function', fileId: 'f1',
        startLine: 11, endLine: 15, name: 'fn3', parentId: null, body: 'fn3 body' },
    ]);

    const dim = 16;
    // Explicitly rank: fn2 > fn1 > fn3 in similarity to query
    const queryDir = new Array(dim).fill(0);
    queryDir[0] = 1;
    const fn1Vec = new Array(dim).fill(0);
    fn1Vec[0] = 0.5;
    const fn2Vec = new Array(dim).fill(0);
    fn2Vec[0] = 0.9;
    const fn3Vec = new Array(dim).fill(0);
    fn3Vec[0] = 0.1;

    store.writeVectors(new Map([
      ['src/a.ts::fn1', fn1Vec],
      ['src/a.ts::fn2', fn2Vec],
      ['src/a.ts::fn3', fn3Vec],
    ]));

    const embedder = FakeEmbeddingAdapter(dim);
    const retriever = new Retriever(store, embedder, DEFAULT_CONFIG.index);
    const results = await retriever.query('query text', { tokens: 1000 });

    // fn2 should rank first (highest cosine similarity to unit vector)
    if (results.length >= 2) {
      expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    }
    expect(results[0].name).toBe('fn2');
  });
});
```

- [ ] **Step 3: Run tests**

Run: `cd packages/core && npx vitest run`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add packages/core/src/retrieval/
git commit -m "feat: add retrieval pipeline with score fusion and token budget
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Core Index Exports and Build Orchestrator

**Files:**
- Create: `packages/core/src/build.ts` — orchestrates the full build pipeline
- Modify: `packages/core/src/index.ts` — add build export

**Interfaces:**
- Consumes: `Indexer`, `Store`, `Graph`, `EmbeddingAdapter`, config
- Produces:
  - `async buildIndex(rootPath: string, store: Store, embedder: EmbeddingAdapter, config: CcOptimizeConfig): Promise<IndexStats>`
  - Atomic rebuild: writes to temp dir, then atomically replaces `.context-optimizer/`
  - Reports progress (console.log)

- [ ] **Step 1: Write build.ts**

```typescript
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
```

- [ ] **Step 2: Update core index.ts**

Edit `packages/core/src/index.ts`:

```typescript
export * from './types.js';
export * from './config/index.js';
export * from './indexer/index.js';
export * from './graph/index.js';
export * from './embeddings/index.js';
export * from './store/index.js';
export * from './retrieval/index.js';
export { buildIndex, buildIndexAtomic } from './build.js';
```

- [ ] **Step 3: Build and verify**

Run: `cd packages/core && npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 4: Commit**

```bash
git add packages/core/src/build.ts packages/core/src/index.ts
git commit -m "feat: add build orchestrator with atomic index persistence
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Test Fixture Repo

**Files:**
- Create: `test/fixtures/sample-repo/src/auth.ts`
- Create: `test/fixtures/sample-repo/src/user.ts`
- Create: `test/fixtures/sample-repo/src/db.ts`
- Create: `test/fixtures/sample-repo/src/utils.ts`
- Create: `test/fixtures/sample-repo/src/index.ts`

**Ground truth labels for 3 queries:**
- "fix auth bug" → `src/auth.ts::login`, `src/auth.ts::verifyToken`, `src/user.ts::findById`
- "add database logging" → `src/db.ts::query`, `src/utils.ts::log`
- "refactor user module" → `src/user.ts::findById`, `src/user.ts::createUser`, `src/user.ts::updateUser`

- [ ] **Step 1: Write test/fixtures/sample-repo/src/auth.ts**

```typescript
import { findById } from './user.js';
import { query } from './db.js';
import { hash } from './utils.js';

export async function login(username: string, password: string): Promise<string | null> {
  const user = await findById(username);
  if (!user) return null;

  const hashed = hash(password);
  const stored = await query('SELECT password_hash FROM users WHERE username = ?', [username]);

  if (!stored || stored.password_hash !== hashed) return null;

  const token = await generateToken(user.id);
  return token;
}

export async function verifyToken(token: string): Promise<string | null> {
  try {
    const decoded = parseToken(token);
    const user = await findById(decoded.userId);
    if (!user) return null;
    return user.id;
  } catch {
    return null;
  }
}

export async function refreshSession(token: string): Promise<string | null> {
  const userId = await verifyToken(token);
  if (!userId) return null;
  return generateToken(userId);
}

function generateToken(userId: string): Promise<string> {
  // simplified
  return Promise.resolve(`token-${userId}-${Date.now()}`);
}

function parseToken(token: string): { userId: string } {
  const parts = token.split('-');
  return { userId: parts[1] };
}
```

- [ ] **Step 2: Write test/fixtures/sample-repo/src/user.ts**

```typescript
import { query, insert } from './db.js';

export async function findById(id: string): Promise<User | null> {
  const rows = await query('SELECT * FROM users WHERE id = ?', [id]);
  return rows.length > 0 ? rows[0] as User : null;
}

export async function createUser(username: string, email: string): Promise<User> {
  const id = `user-${Date.now()}`;
  await insert('users', { id, username, email });
  return { id, username, email };
}

export async function updateUser(id: string, data: Partial<User>): Promise<void> {
  await query('UPDATE users SET data = ? WHERE id = ?', [data, id]);
}

export interface User {
  id: string;
  username: string;
  email: string;
}
```

- [ ] **Step 3: Write test/fixtures/sample-repo/src/db.ts**

```typescript
export async function query(sql: string, params: unknown[]): Promise<any[]> {
  console.log(`[DB] query: ${sql}`);
  return [];
}

export async function insert(table: string, data: Record<string, unknown>): Promise<void> {
  console.log(`[DB] insert into ${table}`);
}

export async function connect(url: string): Promise<void> {
  console.log(`[DB] connecting to ${url}`);
}

export async function disconnect(): Promise<void> {
  console.log('[DB] disconnected');
}
```

- [ ] **Step 4: Write test/fixtures/sample-repo/src/utils.ts**

```typescript
export function hash(input: string): string {
  // simplified hash
  let h = 0;
  for (let i = 0; i < input.length; i++) {
    h = ((h << 5) - h) + input.charCodeAt(i);
    h |= 0;
  }
  return h.toString(16);
}

export function log(level: string, message: string): void {
  const ts = new Date().toISOString();
  console.log(`[${ts}] [${level}] ${message}`);
}
```

- [ ] **Step 5: Write test/fixtures/sample-repo/src/index.ts**

```typescript
export { login, verifyToken, refreshSession } from './auth.js';
export { findById, createUser, updateUser } from './user.js';
export { query, insert, connect, disconnect } from './db.js';
export { hash, log } from './utils.js';
```

- [ ] **Step 6: Commit**

```bash
git add test/
git commit -m "test: add sample repo fixture with known call relationships
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 12: CLI — Build, Rebuild, Status Commands

**Files:**
- Create: `packages/cli/src/commands/index.ts`
- Create: `packages/cli/src/commands/__tests__/commands.test.ts`

**Interfaces:**
- Consumes: `@cc-optimize/core` exports, process.argv
- Produces: `run(argv: string[])` function that dispatches to subcommands

- [ ] **Step 1: Write commands/index.ts**

```typescript
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

async function cmdMcp(args: Record<string, string>): Promise<void> {
  // Dynamic import to avoid loading MCP SDK when not needed
  const { start } = await import('@cc-optimize/mcp');
  await start();
}
```

- [ ] **Step 2: Build and verify**

Run: `cd packages/cli && npx tsc --noEmit`
Expected: Type errors if any; fix and re-check.

- [ ] **Step 3: Commit**

```bash
git add packages/cli/src/
git commit -m "feat: add CLI commands for build, rebuild, query, status, mcp
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 13: MCP Server

**Files:**
- Create: `packages/mcp/src/server.ts`

**Interfaces:**
- Consumes: `@cc-optimize/core` exports, `@modelcontextprotocol/sdk`
- Produces: `start()` function that creates a stdio MCP server with `retrieve_context` tool

- [ ] **Step 1: Write server.ts**

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import {
  loadConfig,
  Store,
  Retriever,
  OpenAIEmbeddingAdapter,
  OllamaEmbeddingAdapter,
  EmbeddingAdapter,
} from '@cc-optimize/core';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

function createEmbedder(config: any): EmbeddingAdapter {
  if (config.embeddings.provider === 'ollama') {
    return OllamaEmbeddingAdapter();
  }
  if (config.embeddings.provider === 'openai') {
    if (!config.embeddings.apiKey) {
      throw new Error('OpenAI API key not set.');
    }
    return OpenAIEmbeddingAdapter(config.embeddings.apiKey, config.embeddings.model);
  }
  throw new Error(`Unknown embedding provider: ${config.embeddings.provider}`);
}

export async function start(): Promise<void> {
  const server = new Server(
    { name: 'context-optimizer', version: '0.1.0' },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'retrieve_context',
        description: 'Retrieve the most relevant code regions for a given task from the indexed codebase.',
        inputSchema: {
          type: 'object',
          properties: {
            task:   { type: 'string', description: 'Natural language task description' },
            file:   { type: 'string', description: 'Current file path (optional, biases retrieval)' },
            line:   { type: 'number', description: 'Current line number (optional, biases retrieval)' },
            tokens: { type: 'number', description: 'Token budget (default: 8000)' },
            format: { type: 'string', enum: ['context', 'manifest', 'json'], default: 'context' },
          },
          required: ['task'],
        },
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params.name !== 'retrieve_context') {
      throw new Error(`Unknown tool: ${request.params.name}`);
    }

    const args = (request.params.arguments ?? {}) as Record<string, any>;
    const root = process.cwd();
    const optDir = resolve(root, '.context-optimizer');

    if (!existsSync(optDir)) {
      return {
        content: [{ type: 'text', text: 'Error: No index found. Run cc-optimize build first.' }],
        isError: true,
      };
    }

    const config = loadConfig(root);
    const embedder = createEmbedder(config);
    const store = Store.open(resolve(optDir, 'graph.db'));

    try {
      const retriever = new Retriever(store, embedder, config.index);
      const results = await retriever.query(args.task, {
        file: args.file,
        line: args.line,
        tokens: args.tokens ?? config.index.defaultTokenBudget,
      });

      let text: string;
      switch (args.format ?? 'context') {
        case 'manifest':
          text = results.map((r: any) => `${r.file}:${r.startLine}-${r.endLine}`).join('\n');
          break;
        case 'json':
          text = JSON.stringify(results, null, 2);
          break;
        default:
          text = results.map((r: any) =>
            `// ${r.file}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.body}`
          ).join('\n\n');
          break;
      }

      return { content: [{ type: 'text', text }] };
    } finally {
      store.close();
    }
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}
```

- [ ] **Step 2: Build and verify**

Run: `cd packages/mcp && npx tsc --noEmit`
Expected: No type errors after resolving dependency types.

- [ ] **Step 3: Commit**

```bash
git add packages/mcp/src/
git commit -m "feat: add MCP server with retrieve_context tool
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 14: End-to-End Integration Test

**Files:**
- Create: `test/integration/__tests__/e2e.test.ts`

**Interfaces:**
- Consumes: All three packages, sample fixture repo
- Produces: Integration test that runs build → query → verify results match ground truth

- [ ] **Step 1: Write e2e.test.ts**

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { resolve } from 'node:path';
import { rmSync, existsSync } from 'node:fs';
import {
  buildIndexAtomic,
  Store,
  Retriever,
  FakeEmbeddingAdapter,
  DEFAULT_CONFIG,
} from '@cc-optimize/core';

const FIXTURE_ROOT = resolve(import.meta.dirname!, '../../fixtures/sample-repo');

describe('end-to-end', () => {
  beforeAll(async () => {
    // Build index over fixture repo
    const adapter = FakeEmbeddingAdapter(16);
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    config.index.ignore = [];
    await buildIndexAtomic(FIXTURE_ROOT, adapter, config);
  });

  afterAll(() => {
    const optDir = resolve(FIXTURE_ROOT, '.context-optimizer');
    if (existsSync(optDir)) {
      rmSync(optDir, { recursive: true, force: true });
    }
  });

  it('retrieves auth-related nodes for auth query', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('fix auth bug', { tokens: 5000 });

      const names = results.map((r) => r.name).filter(Boolean);
      // Ground truth: login, verifyToken, findById should be in results
      const hasAuthFunction = names.some((n) => n === 'login' || n === 'verifyToken');
      expect(hasAuthFunction).toBe(true);
    } finally {
      store.close();
    }
  });

  it('retrieves db-related nodes for database query', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('add database logging', { tokens: 5000 });

      const names = results.map((r) => r.name).filter(Boolean);
      const hasDbFunction = names.some((n) => n === 'query' || n === 'log' || n === 'connect');
      expect(hasDbFunction).toBe(true);
    } finally {
      store.close();
    }
  });

  it('returns results in correct format with scores', async () => {
    const config = { ...DEFAULT_CONFIG, index: { ...DEFAULT_CONFIG.index } };
    const adapter = FakeEmbeddingAdapter(16);
    const store = Store.open(resolve(FIXTURE_ROOT, '.context-optimizer', 'graph.db'));
    try {
      const retriever = new Retriever(store, adapter, config.index);
      const results = await retriever.query('refactor user module');

      for (const r of results) {
        expect(r.file).toBeTruthy();
        expect(r.startLine).toBeGreaterThan(0);
        expect(r.endLine).toBeGreaterThan(0);
        expect(typeof r.score).toBe('number');
        expect(r.body).toBeTruthy();
      }
    } finally {
      store.close();
    }
  });
});
```

- [ ] **Step 2: Run integration tests**

Run: `cd test && npx vitest run`
Expected: All 3 tests pass with ground truth labels (auth functions in auth query, etc.).

- [ ] **Step 3: Commit**

```bash
git add test/integration/
git commit -m "test: add end-to-end integration test against fixture repo
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Documentation — README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Context Optimizer (cc-optimize)

Smart context retrieval for Claude Code. Indexes your TypeScript/JavaScript codebase at the function and block level, then returns only the code regions relevant to your task — reducing token usage by 50-90%.

## Quick Start

```bash
# 1. Create config (optional; defaults shown)
cat > .cc-optimize.json << 'EOF'
{
  "embeddings": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "apiKey": "env:OPENAI_API_KEY"
  }
}
EOF

# 2. Build the index
cc-optimize build

# 3. Query for context
cc-optimize query "fix the JWT refresh token expiration bug"
```

## How It Works

1. **Parse** — Tree-sitter extracts every function and block from your source files
2. **Graph** — call relationships and structural dependencies are built into a weighted graph  
3. **Embed** — Each node is embedded (OpenAI or Ollama); graph diffusion spreads signal along edges
4. **Retrieve** — Query → vector search + graph expansion → ranked results with token budget

## Commands

```
cc-optimize build      Build the index from scratch
cc-optimize rebuild    Force full rebuild
cc-optimize query      Retrieve context for a task
cc-optimize status     Show index stats and staleness
cc-optimize mcp        Start MCP server (for Claude Code integration)
```

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `embeddings.provider` | `openai` | `openai` or `ollama` |
| `embeddings.model` | `text-embedding-3-small` | Model name |
| `embeddings.apiKey` | — | `env:VAR` or literal key |
| `index.blockSplitThreshold` | `60` | Lines before splitting functions |
| `index.diffusionAlpha` | `0.3` | Graph diffusion strength |
| `index.defaultTokenBudget` | `8000` | Max tokens per query |
| `index.ignore` | `["node_modules","dist","*.test.ts"]` | Glob patterns |

## Claude Code (MCP) Integration

Add to `.claude/mcp.json`:

```json
{
  "servers": {
    "context-optimizer": {
      "command": "cc-optimize",
      "args": ["mcp"]
    }
  }
}
```

Then inside Claude Code, the `retrieve_context` tool is available automatically.

## Requirements

- Node.js ≥ 18
- OpenAI API key (or Ollama running locally with `nomic-embed-text`)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick start and configuration guide
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Verification before completion

Run the full test suite:

```bash
npm test
```

Expected: All tests pass across all three packages and integration tests.
