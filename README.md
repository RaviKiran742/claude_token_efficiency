# Context Optimizer (cc-optimize)

Smart context retrieval for Claude Code. Indexes your TypeScript/JavaScript codebase at the function and block level, then returns only the code regions relevant to your task — reducing token usage by 50-90%.

## How It Works

1. **Parse** — Tree-sitter extracts every function and block from your source files
2. **Graph** — Call relationships and structural dependencies are built into a weighted graph
3. **Embed** — Each node is embedded (OpenAI or Ollama); graph diffusion spreads signal along edges
4. **Retrieve** — Query → vector search + graph expansion → ranked results with token budget

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

# 4. Query with cursor hint
cc-optimize query "fix the JWT bug" --file src/auth.ts --line 156

# 5. Check status
cc-optimize status
```

## Claude Code (MCP) Integration

Add to `.claude/mcp.json` in your project root:

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

Then inside Claude Code, the `retrieve_context` tool is available automatically. Claude can call it to get context without you running commands manually.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd context-optimizer
npm install
npm run build

# Optionally link globally
npm link
```

## Commands

| Command | Description |
|---------|-------------|
| `cc-optimize build` | Build the index from scratch |
| `cc-optimize rebuild` | Force full rebuild (clears existing index) |
| `cc-optimize query <task>` | Retrieve context for a task description |
| `cc-optimize status` | Show index stats and staleness |
| `cc-optimize mcp` | Start MCP server (for Claude Code integration) |

### Query Options

```
--file <path>     Current file (biases retrieval toward cursor position)
--line <n>        Current line number (biases retrieval toward cursor position)
--tokens <n>      Token budget (default: 8000)
--format <f>      Output format: context | manifest | json (default: context)
--root <path>     Project root directory (default: current directory)
```

### Output Formats

**context** (default) — annotated code blocks ready to paste into Claude Code:
```
// src/auth.ts:142-189 (score: 0.94)
function refreshToken(userId: string) { ... }
```

**manifest** — file:line references:
```
src/auth.ts:142-189
src/user.ts:44-67
```

**json** — structured data:
```json
[{ "file": "src/auth.ts", "startLine": 142, "endLine": 189, "name": "refreshToken", "score": 0.94 }]
```

## Configuration

`.cc-optimize.json` at your project root:

| Key | Default | Description |
|-----|---------|-------------|
| `embeddings.provider` | `openai` | `openai` or `ollama` |
| `embeddings.model` | `text-embedding-3-small` | Model name |
| `embeddings.apiKey` | — | `env:OPENAI_API_KEY` or literal key |
| `index.blockSplitThreshold` | `60` | Lines before splitting functions into blocks |
| `index.diffusionAlpha` | `0.3` | Graph diffusion strength (0-1) |
| `index.defaultTokenBudget` | `8000` | Max estimated tokens per query |
| `index.ignore` | `["node_modules","dist","*.test.ts"]` | Glob patterns to exclude |

### Ollama (local)

```json
{
  "embeddings": {
    "provider": "ollama",
    "model": "nomic-embed-text"
  }
}
```

Requires Ollama running locally with `nomic-embed-text` pulled.

## Requirements

- Node.js ≥ 22 (uses built-in `node:sqlite`)
- For OpenAI: API key set via `OPENAI_API_KEY` env var or config
- For Ollama: running instance with `nomic-embed-text` model

## Technical Details

- **Index storage:** `.context-optimizer/graph.db` (SQLite, WAL mode)
- **Parser:** tree-sitter with TypeScript grammar
- **Embeddings:** OpenAI text-embedding-3-small (1536 dims) or Ollama nomic-embed-text (768 dims)
- **Graph:** Weighted directed graph with call relationships, diffusion with α=0.3
- **Retrieval:** Vector search (top-20) → graph expansion → score fusion → greedy token budget
