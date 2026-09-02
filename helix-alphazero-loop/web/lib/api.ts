export type TreeRow = { action?: number; col: number; label?: string; n: number; q: number; p: number; share: number };

export type GameInfo = {
  id: string;
  name: string;
  rows: number;
  cols: number;
  action_size: number;
  layout: "drop" | "cells" | "hex";
  win: string;
  blurb: string;
};

export type Think = {
  game?: string;
  layout?: string;
  rows?: number;
  cols?: number;
  grid: number[][];
  to_play: number;
  terminal: boolean;
  winner: number;
  policy: number[];
  value: number;
  tree: TreeRow[];
  pv: number[];
  pv_labels?: string[];
  choice: number | null;
  legal?: number[];
  mode: string;
  sims?: number;
};

export type Board = {
  game: string;
  name: string;
  layout: "drop" | "cells" | "hex";
  rows: number;
  cols: number;
  grid: number[][];
  to_play: number;
  terminal: boolean;
  winner: number;
  legal: number[];
  blurb: string;
  win: string;
};

export type Meta = {
  model: string;
  game: string;
  n_params: number;
  checkpoint_games: number;
  has_checkpoint: boolean;
  board: string;
  layout: string;
  win: string;
  blurb: string;
  games: GameInfo[];
  note: string;
};

export function getApi(): string {
  const env = process.env.NEXT_PUBLIC_API_URL;
  if (env) return env.replace(/\/$/, "");
  return "http://localhost:8000";
}

export async function api<T>(path: string, init: RequestInit = {}, timeoutMs = 90_000): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${getApi()}${path}`, { ...init, headers, signal: ctrl.signal });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data as { detail?: string }).detail || res.statusText);
    return data as T;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error(`API at ${getApi()} did not answer. Start uvicorn on 8000.`);
    }
    throw e;
  } finally {
    clearTimeout(t);
  }
}
