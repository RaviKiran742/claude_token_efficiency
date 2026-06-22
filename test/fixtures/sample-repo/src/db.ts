export async function query(sql: string, params: unknown[]): Promise<any[]> {
  console.log(`[DB] query: ${sql}`);
  return [];
}

export async function insert(table: string, data: Record<string, unknown>): Promise<void> {
  console.log(`[DB] insert into ${table}`);
}

export async function connect(url: string): Promise<void> {
  console.log(`[DB] connecting to ${url}`);
}

export async function disconnect(): Promise<void> {
  console.log('[DB] disconnected');
}
