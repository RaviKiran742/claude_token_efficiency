# Context Optimizer — Design Spec

**Date:** 2026-06-22  
**Status:** Approved

---

## Overview

A CLI tool that builds a smart retrieval index over a TypeScript/JavaScript codebase and returns the minimal set of code regions needed to complete a given task. The goal is to reduce tokens sent to Claude Code by sending only relevant functions and blocks rather than whole files.

**v1 scope:** TypeScript/JavaScript only. CLI only. No editor extensions. No log compression, CLAUDE.md optimization, or conversation compaction.

---

## Architecture

Three packages in a single npm workspace:

```
context-optimizer/
├── packages/
│   ├── core/               # engine — indexing, graph, retrieval
│   │   └── src/
│   │       ├── indexer/    # walk files, parse AST, extract nodes
│   │       ├── graph/      # build + diffuse weighted graph
│   │       ├── embeddings/ # OpenAI + Ollama adapters
│   │       ├── store/      # SQLite + sqlite-vec persistence
│   │       └── retrieval/  # query → ranked node list
│   ├── cli/                # thin CLI layer, calls core
│   │   └── src/
│   │       └── commands/   # build, rebuild, query, status, mcp
│   └── mcp/                # MCP server, calls core
│       └── src/
│           └── server.ts   # exposes retrieve_context tool
├── package.json            # npm workspaces root
└── docs/
    └── superpowers/specs/
```

**Tech stack:** TypeScript, Node.js, tree-sitter, ts-morph, SQLite, sqlite-vec, OpenAI embeddings (default), Ollama (adapter), `@modelcontextprotocol/sdk`.

---

## Data Flow

### Build

```
cc-optimize build
  → indexer walks repo (respects ignore patterns)
  → tree-sitter parses each file → function/block nodes
  → ts-morph resolves call graph edges
  → embedding adapter batches + embeds all nodes
  → graph diffusion runs (α-weighted adjacency)
  → store writes nodes + vectors to SQLite
```

### Query

```
cc-optimize query "fix auth bug" --file auth.ts --line 156
  → embed query string
  → vector search → top-20 candidate nodes
  → cursor anchor: resolve --file/--line → nearest node + depth-1 neighbors
  → graph expansion: walk edges with weight ≥ 0.5 from all candidates
  → score fusion
  → apply token budget (greedy by score)
  → extract file ranges
  → output
```

---

## Data Model

### SQLite (`graph.db`)

```sql
CREATE TABLE files (
  id       TEXT PRIMARY KEY,  -- sha256 of absolute path
  path     TEXT NOT NULL,
  mtime    INTEGER NOT NULL,
  ast_hash TEXT NOT NULL       -- sha256 of file content
);

CREATE TABLE nodes (
  id         TEXT PRIMARY KEY,  -- functions: "<file_path>::<function_name>"
                                --    blocks: "<file_path>::<function_name>::<block_index>"
  type       TEXT NOT NULL,     -- "function" | "block"
  file_id    TEXT NOT NULL REFERENCES files(id),
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL,
  name       TEXT,              -- function name, null for blocks
  parent_id  TEXT,              -- null for functions, function node id for blocks
  body       TEXT NOT NULL
);

CREATE TABLE edges (
  source  TEXT NOT NULL REFERENCES nodes(id),
  target  TEXT NOT NULL REFERENCES nodes(id),
  type    TEXT NOT NULL,
  weight  REAL NOT NULL,
  PRIMARY KEY (source, target, type)
);
```

### Vector index (`vectors.db`)

sqlite-vec table: `node_id TEXT`, `embedding FLOAT[N]` where N is adapter-dependent: 1536 for OpenAI `text-embedding-3-small`, 768 for Ollama `nomic-embed-text`. The dimension is recorded in the config snapshot and must match the adapter in use; switching adapters requires a full rebuild.

### Edge types and weights

| Type           | Weight |
|----------------|--------|
| calls          | 1.0    |
| called_by      | 0.8    |
| overrides      | 0.7    |
| uses_type      | 0.5    |
| same_function  | 0.9    |
| co_modified    | 0.4    |
| shares_scope   | 0.6    |

Block nodes inherit their parent function's call graph edges at 0.8× weight multiplier.

---

## Graph Diffusion

Runs once at build time after all nodes are embedded. Propagates embedding signal along weighted edges:

```
E_final = (1 - α) * E_semantic + α * (A_normalized * E_semantic)
```

- `α = 0.3` default (configurable)
- `A_normalized` = adjacency matrix normalized by out-degree
- Single diffusion pass for v1
- Diffused embeddings replace raw embeddings in `vectors.db`

---

## Retrieval Scoring

```
score_fusion = 0.6 * semantic_score + 0.4 * graph_proximity_to_cursor
```

`graph_proximity_to_cursor` = `1 / (1 + bfs_distance)` where `bfs_distance` is the number of hops in the weighted graph from the candidate node to the cursor-anchored node (using edges with weight ≥ 0.5 only). Nodes unreachable from the cursor anchor get a proximity score of 0.

If no cursor provided: `score_fusion = semantic_score`

**Token budget:** Estimated as `(end_line - start_line) * 5` per node. Nodes added greedily by score until budget exhausted. Default: 8000 tokens.

---

## Block Splitting

Functions over 60 lines are split at AST block boundaries (`if`, `for`, `try`, sequential statement groups). Each block node:
- Has `parent_id` pointing to its function node
- Retains a reference to the function signature for context reconstruction
- Inherits parent function's variable declarations in scope (prepended on extraction)

---

## CLI Interface

```bash
cc-optimize build    [--root <path>] [--config <path>]
cc-optimize rebuild  [--root <path>]
cc-optimize query <task> [--file <path>] [--line <n>] [--tokens <n>] [--format context|manifest|json]
cc-optimize status
```

**Output formats:**

- `context` (default): raw source blocks with `file:line` headers, ready to paste into Claude Code
- `manifest`: one `file:start-end` per line
- `json`: array of `{ file, start, end, name, score }`

**Staleness warning:** On query, if any tracked file's mtime has changed since last build, emit a non-blocking warning and proceed.

---

## MCP Server

The MCP server exposes a single tool to Claude Code:

**Tool: `retrieve_context`**

```json
{
  "name": "retrieve_context",
  "description": "Retrieve the most relevant code regions for a given task from the indexed codebase.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "task":   { "type": "string", "description": "Natural language task description" },
      "file":   { "type": "string", "description": "Current file path (optional, biases retrieval)" },
      "line":   { "type": "number", "description": "Current line number (optional, biases retrieval)" },
      "tokens": { "type": "number", "description": "Token budget (default: 8000)" },
      "format": { "type": "string", "enum": ["context", "manifest", "json"], "default": "context" }
    },
    "required": ["task"]
  }
}
```

**Starting the server:**

```bash
cc-optimize mcp        # stdio transport (default, used by Claude Code)
```

**Claude Code configuration (`.claude/mcp.json` at repo root):**

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

**Behaviour:** The MCP server is stateless between calls — each `retrieve_context` invocation opens the SQLite index, runs the query pipeline, and returns results. No persistent process state. The index must be built first (`cc-optimize build`); if no index exists, the tool returns a clear error message rather than failing silently.

---

## Configuration

`.cc-optimize.json` at repo root:

```json
{
  "embeddings": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "apiKey": "env:OPENAI_API_KEY"
  },
  "index": {
    "blockSplitThreshold": 60,
    "diffusionAlpha": 0.3,
    "defaultTokenBudget": 8000,
    "ignore": ["node_modules", "dist", "*.test.ts"]
  }
}
```

Ollama adapter selected by setting `"provider": "ollama"` and `"model": "nomic-embed-text"`.

---

## Error Handling

**Embedding API failure:** Retry with exponential backoff (3 attempts). On persistent failure, abort build and report which nodes failed. Partial indexes are not written (atomic write via temp file + rename).

**ts-morph call resolution failure:** Log unresolved symbol, continue with tree-sitter-only edges. Degraded accuracy, not a hard failure.

**All other errors:** Clear message with offending file path, non-zero exit.

---

## Testing

**Unit tests** (`packages/core`):
- `indexer/` — parse fixture file, assert nodes extracted correctly
- `graph/` — build small graph, assert edge weights and diffusion output
- `retrieval/` — build index over fixture repo, assert query results match ground truth

**CLI smoke tests** (`packages/cli`):
- Run each command against fixture repo, assert output shape and exit code

**MCP server tests** (`packages/mcp`):
- Start server in-process, call `retrieve_context` with fixture repo index, assert response shape and that labeled nodes appear in results

**Fixture repo** (`test/fixtures/sample-repo/`): ~10 synthetic TypeScript files, ~20 functions, known call relationships. Ground truth labels for 3 sample queries — labeled nodes must appear in top-5 results.

**Fake embedding adapter:** Returns hash-based deterministic vectors. Used in all tests — no API calls, fast, stable.

---

## Index Storage Location

```
<repo-root>/.context-optimizer/
  graph.db          # nodes, edges, files
  vectors.db        # embeddings
  config.json       # resolved config snapshot
```

Added to `.gitignore` automatically on first build.
