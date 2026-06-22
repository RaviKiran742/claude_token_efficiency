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
