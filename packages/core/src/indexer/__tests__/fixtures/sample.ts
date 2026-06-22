// Test fixture — not parsed by the indexer test itself, but used as input
export function add(a: number, b: number): number {
  return a + b;
}

export function multiply(a: number, b: number): number {
  return a * b;
}

export function calculate(a: number, b: number): number {
  const sum = add(a, b);
  return multiply(sum, 2);
}
