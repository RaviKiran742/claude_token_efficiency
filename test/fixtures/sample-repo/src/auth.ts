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
