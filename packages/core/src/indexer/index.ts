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
  fileContent: string; // for ts-morph call resolution
  astNode?: Parser.SyntaxNode;
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
          allNodes.push(fn);
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
        astNode: node,
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
    const bodyNode = fn.astNode?.childForFieldName('body') ?? null;

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

  private buildEdges(
    rootPath: string,
    nodes: RawNode[],
    files: FileNode[],
  ): GraphEdge[] {
    const edges: GraphEdge[] = [];

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
      this.walkForCalls(tree.rootNode, content, file.path, nodes, edges);
    }

    return edges;
  }

  private walkForCalls(
    node: Parser.SyntaxNode,
    content: string,
    filePath: string,
    allNodes: RawNode[],
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
      this.walkForCalls(node.namedChild(i)!, content, filePath, allNodes, out);
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
