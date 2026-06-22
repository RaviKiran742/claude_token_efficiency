import { createRequire } from 'node:module';
import { existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { FileNode, CodeNode, GraphEdge, IndexStats } from '../types.js';

// Load node:sqlite via createRequire to bypass Vite's module resolution.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const _sqlite = createRequire(import.meta.url)('node:sqlite') as {
  DatabaseSync: new (path: string) => DatabaseInstance;
};
const DatabaseSync = _sqlite.DatabaseSync;

interface DatabaseInstance {
  exec(sql: string): void;
  prepare(sql: string): PreparedStatement;
  close(): void;
}

interface PreparedStatement {
  run(...args: unknown[]): { changes: number; lastInsertRowid: number };
  get(...args: unknown[]): unknown;
  all(...args: unknown[]): unknown[];
}

export class Store {
  private db: DatabaseInstance;

  private constructor(dbPath: string) {
    this.db = new DatabaseSync(dbPath);
    this.db.exec('PRAGMA journal_mode = WAL');
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
        node_id   TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
      );
    `);
  }

  writeFiles(files: FileNode[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO files (id, path, mtime, ast_hash)
      VALUES (@id, @path, @mtime, @astHash)
    `);
    this.db.exec('BEGIN');
    try {
      for (const f of files) insert.run(f);
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  writeNodes(nodes: CodeNode[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO nodes (id, type, file_id, start_line, end_line, name, parent_id, body)
      VALUES (@id, @type, @fileId, @startLine, @endLine, @name, @parentId, @body)
    `);
    this.db.exec('BEGIN');
    try {
      for (const n of nodes) insert.run(n);
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  writeEdges(edges: GraphEdge[]): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO edges (source, target, type, weight)
      VALUES (@source, @target, @type, @weight)
    `);
    this.db.exec('BEGIN');
    try {
      for (const e of edges) insert.run(e);
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  getFiles(): FileNode[] {
    return this.db.prepare(
      `SELECT id, path, mtime, ast_hash as astHash FROM files`
    ).all() as unknown as FileNode[];
  }

  getNodes(): CodeNode[] {
    return this.db.prepare(`
      SELECT id, type, file_id as fileId, start_line as startLine,
             end_line as endLine, name, parent_id as parentId, body
      FROM nodes
    `).all() as unknown as CodeNode[];
  }

  getEdges(): GraphEdge[] {
    return this.db.prepare(
      `SELECT source, target, type, weight FROM edges`
    ).all() as unknown as GraphEdge[];
  }

  writeVectors(vectors: Map<string, number[]>): void {
    const insert = this.db.prepare(`
      INSERT OR REPLACE INTO vectors (node_id, embedding) VALUES (?, ?)
    `);
    this.db.exec('BEGIN');
    try {
      for (const [nodeId, vec] of vectors) {
        const buf = Buffer.allocUnsafe(vec.length * 4);
        for (let i = 0; i < vec.length; i++) {
          buf.writeFloatLE(vec[i], i * 4);
        }
        insert.run(nodeId, buf);
      }
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  /** Load all vectors for graph diffusion. */
  getAllVectors(): Map<string, number[]> {
    const rows = this.db.prepare(
      `SELECT node_id, embedding FROM vectors`
    ).all() as unknown as { node_id: string; embedding: Buffer }[];
    const result = new Map<string, number[]>();
    for (const row of rows) {
      const buf = Buffer.isBuffer(row.embedding) ? row.embedding : Buffer.from(row.embedding);
      const vecLen = buf.length / 4;
      const emb = new Array(vecLen);
      for (let i = 0; i < vecLen; i++) {
        emb[i] = buf.readFloatLE(i * 4);
      }
      result.set(row.node_id, emb);
    }
    return result;
  }

  /** Cosine similarity search — JS implementation since we store vectors as BLOB. */
  searchVectors(queryVec: number[], k: number): Array<{ nodeId: string; score: number }> {
    const rows = this.db.prepare(
      `SELECT node_id, embedding FROM vectors`
    ).all() as unknown as { node_id: string; embedding: Buffer }[];

    const queryNorm = Math.sqrt(queryVec.reduce((s, v) => s + v * v, 0));

    const scored = rows.map((row) => {
      const buf = Buffer.isBuffer(row.embedding) ? row.embedding : Buffer.from(row.embedding);
      const vecLen = buf.length / 4;
      const emb = new Array(vecLen);
      for (let i = 0; i < vecLen; i++) {
        emb[i] = buf.readFloatLE(i * 4);
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
    this.db.exec(
      `DELETE FROM edges; DELETE FROM nodes; DELETE FROM files; DELETE FROM vectors;`
    );
  }

  getStats(): IndexStats {
    const files = this.db.prepare(
      `SELECT COUNT(*) as cnt FROM files`
    ).get() as unknown as { cnt: number };
    const nodes = this.db.prepare(
      `SELECT COUNT(*) as cnt FROM nodes`
    ).get() as unknown as { cnt: number };
    const edges = this.db.prepare(
      `SELECT COUNT(*) as cnt FROM edges`
    ).get() as unknown as { cnt: number };
    const latest = this.db.prepare(
      `SELECT MAX(mtime) as mt FROM files`
    ).get() as unknown as { mt: number | null };
    return {
      fileCount: files.cnt,
      nodeCount: nodes.cnt,
      edgeCount: edges.cnt,
      lastBuilt: latest.mt ?? null,
    };
  }

  /** Check if any tracked file has been modified since indexing. */
  hasStaleFiles(rootPath: string): boolean {
    const rows = this.db.prepare(
      `SELECT path, mtime FROM files`
    ).all() as unknown as { path: string; mtime: number }[];
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
