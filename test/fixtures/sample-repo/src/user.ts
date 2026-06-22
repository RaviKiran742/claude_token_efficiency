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
