import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
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
