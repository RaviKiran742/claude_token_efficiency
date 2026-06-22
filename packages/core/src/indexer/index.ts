import { readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import fg from 'fast-glob';
import { createHash } from 'node:crypto';
import { minimatch } from 'minimatch';
import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';
import { CcOptimizeConfig, CodeNode, FileNode, GraphEdge } from '../types.js';

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
  fileContent: string;
  astNode?: Parser.SyntaxNode;
}

export class Indexer {
  private config: CcOptimizeConfig;

  constructor(config: CcOptimizeConfig) {
    this.config = config;
  }

  // ── public API ──────────────────────────────────────────────

  /** Walk the repo and return FileNode entries (lightweight — paths + hashes only). */
  walkFiles(rootPath: string): FileNode[] {
    const entries = fg.globSync('**/*.{ts,tsx,js,jsx,cts,mts}', {
      cwd: rootPath,
      ignore: ['node_modules/**', 'dist/**'],
      absolute: false,
    } as fg.Options);

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

  /**
   * Parse a single file, returning its CodeNode entries.
   * Does NOT build cross-file edges — call buildEdgesForFile separately.
   */
  parseFile(rootPath: string, file: FileNode): CodeNode[] {
    const parser = new Parser();
    parser.setLanguage(TypeScript.tsx as Parser.Language);

    const absPath = resolve(rootPath, file.path);
    let content: string;
    try {
      content = readFileSync(absPath, 'utf-8');
    } catch {
      return [];
    }

    const tree = parser.parse(content);
    const rawNodes = this.extractFunctions(tree.rootNode, content, file);
    const nodes: CodeNode[] = [];

    for (const fn of rawNodes) {
      fn.fileContent = content;
      const lineCount = fn.endLine - fn.startLine + 1;
      if (lineCount > this.config.index.blockSplitThreshold) {
        nodes.push(this.toCodeNode(fn));
        const blocks = this.splitIntoBlocks(fn, content);
        nodes.push(...blocks.map((b) => this.toCodeNode(b)));
      } else {
        nodes.push(this.toCodeNode(fn));
      }
    }

    return nodes;
  }

  /**
   * Parse a file and build cross-file call-graph edges using a global
   * name→node-id lookup map keyed by function name.
   */
  buildEdgesForFile(
    rootPath: string,
    file: FileNode,
    nodeNameMap: Map<string, string[]>,
  ): GraphEdge[] {
    const parser = new Parser();
    parser.setLanguage(TypeScript.tsx as Parser.Language);

    const absPath = resolve(rootPath, file.path);
    let content: string;
    try {
      content = readFileSync(absPath, 'utf-8');
    } catch {
      return [];
    }

    const tree = parser.parse(content);
    const edges: GraphEdge[] = [];
    this.walkForCalls(tree.rootNode, content, file.path, nodeNameMap, edges);
    return edges;
  }

  // ── private helpers ─────────────────────────────────────────

  private toCodeNode(raw: RawNode): CodeNode {
    return {
      id: raw.id,
      type: raw.type,
      fileId: raw.fileId,
      startLine: raw.startLine,
      endLine: raw.endLine,
      name: raw.name,
      parentId: raw.parentId,
      body: raw.body,
    };
  }

  private shouldIgnore(relPath: string): boolean {
    for (const pattern of this.config.index.ignore) {
      if (minimatch(relPath, pattern, { matchBase: true })) return true;
    }
    return false;
  }

  // ── AST extraction ──────────────────────────────────────────

  private extractFunctions(
    root: Parser.SyntaxNode,
    content: string,
    file: FileNode,
  ): RawNode[] {
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
    const namedTypes = [
      'function_declaration', 'method_definition',
      'arrow_function', 'function_expression',
    ];
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
        astNode: node,
      });
    }

    for (let i = 0; i < node.namedChildCount; i++) {
      this.walkForFunctions(node.namedChild(i)!, content, file, out);
    }
  }

  // ── block splitting ─────────────────────────────────────────

  private splitIntoBlocks(fn: RawNode, content: string): RawNode[] {
    const blocks: RawNode[] = [];
    const bodyNode = fn.astNode?.childForFieldName('body') ?? null;

    if (!bodyNode) {
      blocks.push({ ...fn, type: 'block', parentId: fn.id });
      return blocks;
    }

    const varDecls: string[] = [];
    const children = bodyNode.namedChildren;
    let blockIndex = 0;

    for (const child of children) {
      const blockTypes = [
        'if_statement', 'for_statement', 'for_in_statement',
        'while_statement', 'try_statement', 'switch_statement',
      ];

      if (child.type === 'lexical_declaration' || child.type === 'variable_declaration') {
        varDecls.push(content.slice(child.startIndex, child.endIndex));
        continue;
      }

      if (blockTypes.includes(child.type) || child.type === 'expression_statement'
        || child.type === 'return_statement') {

        const childStart = child.startPosition.row + 1;
        const childEnd = child.endPosition.row + 1;
        const childBody = content.slice(child.startIndex, child.endIndex);

        blockIndex++;
        const blockId = `${fn.id}::${blockIndex}`;
        const prefix = varDecls.length > 0 ? varDecls.join('\n') + '\n\n' : '';
        blocks.push({
          id: blockId,
          type: 'block',
          fileId: fn.fileId,
          filePath: fn.filePath,
          startLine: childStart,
          endLine: childEnd,
          name: null,
          parentId: fn.id,
          body: prefix + childBody,
          fileContent: content,
        });
      }
    }

    if (blocks.length === 0) {
      blocks.push({ ...fn, type: 'block', parentId: fn.id });
    }

    return blocks;
  }

  // ── edge building ───────────────────────────────────────────

  private walkForCalls(
    node: Parser.SyntaxNode,
    content: string,
    filePath: string,
    nodeNameMap: Map<string, string[]>,
    out: GraphEdge[],
  ): void {
    if (node.type === 'call_expression') {
      const fnNode = node.namedChild(0);
      if (fnNode) {
        const calledName = content.slice(fnNode.startIndex, fnNode.endIndex);
        // O(1) name lookup instead of O(n) scan
        const targetIds = nodeNameMap.get(calledName);
        if (targetIds) {
          const callerId = this.findEnclosingFunctionId(node, content, filePath);
          if (callerId) {
            for (const targetId of targetIds) {
              out.push({ source: callerId, target: targetId, type: 'calls', weight: 1.0 });
              out.push({ source: targetId, target: callerId, type: 'called_by', weight: 0.8 });
            }
          }
        }
      }
    }

    for (let i = 0; i < node.namedChildCount; i++) {
      this.walkForCalls(node.namedChild(i)!, content, filePath, nodeNameMap, out);
    }
  }

  private findEnclosingFunctionId(
    callNode: Parser.SyntaxNode,
    content: string,
    filePath: string,
  ): string | null {
    let current = callNode.parent;
    while (current) {
      const fnTypes = [
        'function_declaration', 'method_definition',
        'arrow_function', 'function_expression',
      ];
      if (fnTypes.includes(current.type)) {
        const nameNode = current.childForFieldName?.('name') ?? current.namedChild(0);
        let name: string | null = null;
        if (nameNode && nameNode.type === 'identifier') {
          name = content.slice(nameNode.startIndex, nameNode.endIndex);
        }
        return `${filePath}::${name ?? 'anonymous'}`;
      }
      current = current.parent;
    }
    return null;
  }
}
