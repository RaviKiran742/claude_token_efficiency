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
