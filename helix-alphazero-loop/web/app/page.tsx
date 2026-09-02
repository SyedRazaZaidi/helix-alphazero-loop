"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Board, type GameInfo, type Meta, type Think, type TreeRow } from "@/lib/api";

const STAGES = ["SELF-PLAY", "REPLAY BUFFER", "POLICY/VALUE SGD", "MCTS", "ARENA"];
type Mode = "mcts" | "net" | "random";
type MetricRow = {
  game?: number;
  loss?: number;
  vs_random?: { wins: number; games: number; draws?: number; losses?: number };
};

const OTHELLO_PASS = 36;

export default function LabPage() {
  const [gameId, setGameId] = useState("connect4");
  const [catalog, setCatalog] = useState<GameInfo[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [think, setThink] = useState<Think | null>(null);
  const [mode, setMode] = useState<Mode>("mcts");
  const [sims, setSims] = useState(48);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [evalMsg, setEvalMsg] = useState("");
  const [metrics, setMetrics] = useState<MetricRow[]>([]);

  const hydrate = useCallback((b: Board) => {
    setBoard(b);
  }, []);

  const loadGame = useCallback(
    async (id: string) => {
      setError("");
      setThink(null);
      setEvalMsg("");
      setGameId(id);
      setSims(id === "connect4" ? 48 : 32);
      try {
        const [m, b, rows] = await Promise.all([
          api<Meta>(`/meta?game=${id}`),
          api<Board>(`/board?game=${id}`),
          api<{ rows: MetricRow[] }>(`/metrics?game=${id}`),
        ]);
        setMeta(m);
        setCatalog(m.games || []);
        hydrate(b);
        setMetrics(rows.rows || []);
      } catch (e) {
        setError(e instanceof Error ? e.message : "load failed");
      }
    },
    [hydrate],
  );

  useEffect(() => {
    loadGame("connect4");
  }, [loadGame]);

  async function reset() {
    setThink(null);
    const b = await api<Board>(`/board?game=${gameId}`);
    hydrate(b);
  }

  async function playAction(action: number) {
    if (!board || busy || board.terminal || board.to_play !== 1 || !board.legal.includes(action)) return;
    setBusy(true);
    setError("");
    try {
      const after = await api<Board>("/move", {
        method: "POST",
        body: JSON.stringify({ game: gameId, grid: board.grid, to_play: board.to_play, action }),
      });
      hydrate(after);
      if (after.terminal) {
        setThink(null);
        return;
      }
      const t = await api<Think>("/think", {
        method: "POST",
        body: JSON.stringify({ game: gameId, grid: after.grid, to_play: after.to_play, sims, mode }),
      });
      setThink(t);
      if (t.choice == null) return;
      const bot = await api<Board>("/move", {
        method: "POST",
        body: JSON.stringify({ game: gameId, grid: after.grid, to_play: after.to_play, action: t.choice }),
      });
      hydrate(bot);
    } catch (e) {
      setError(e instanceof Error ? e.message : "move failed");
    } finally {
      setBusy(false);
    }
  }

  async function runEval() {
    setEvalMsg("running arena vs random…");
    try {
      const r = await api<{ wins: number; draws: number; losses: number; games: number }>(`/eval?game=${gameId}`);
      setEvalMsg(`vs random  ${r.wins}W ${r.draws}D ${r.losses}L  / ${r.games}`);
    } catch (e) {
      setEvalMsg(e instanceof Error ? e.message : "eval failed");
    }
  }

  const last = metrics.filter((m) => m.vs_random).at(-1);
  const grid = board?.grid ?? [];
  const legal = board?.legal ?? [];
  const layout = board?.layout ?? "drop";

  return (
    <div className="flex min-h-full flex-col bg-void">
      <header className="flex flex-wrap items-start justify-between gap-4 px-6 py-4">
        <div>
          <p className="font-mono text-[11px] tracking-[0.28em] text-signal">CLOSED-LOOP LEARNING · FOUR RULESETS</p>
          <h1 className="text-3xl tracking-tight">Helix</h1>
        </div>
        {meta ? (
          <div className="flex flex-wrap gap-2 font-mono text-[11px]">
            <span className="border border-signal/40 px-2 py-1 text-signal">{meta.model}</span>
            <span className="border border-line px-2 py-1 text-ash">{Math.round(meta.n_params / 1000)}k params</span>
            <span className="border border-line px-2 py-1 text-ash">{meta.checkpoint_games} train games</span>
            <span className={`border px-2 py-1 ${meta.has_checkpoint ? "border-signal/40 text-signal" : "border-copper/50 text-copper"}`}>
              {meta.has_checkpoint ? "checkpoint" : "untrained net"}
            </span>
          </div>
        ) : null}
      </header>

      <div className="flex flex-wrap gap-2 px-6 pb-4">
        {(catalog.length ? catalog : [{ id: "connect4", name: "Connect Four" } as GameInfo]).map((g) => (
          <button
            key={g.id}
            type="button"
            onClick={() => loadGame(g.id)}
            className={`border px-3 py-2 text-left font-mono text-[12px] ${
              gameId === g.id ? "border-signal bg-signal text-void" : "border-line text-ash hover:text-signal"
            }`}
          >
            <span className="block tracking-wide">{g.name}</span>
            <span className={`block text-[10px] ${gameId === g.id ? "text-void/70" : "text-ash/80"}`}>{g.win}</span>
          </button>
        ))}
      </div>

      <div className="mx-6 mb-4 grid grid-cols-5 gap-1">
        {STAGES.map((s, i) => (
          <div key={s} className="border border-line bg-panel px-2 py-2">
            <p className="font-mono text-[10px] text-ash">0{i + 1}</p>
            <p className="font-mono text-[11px] text-signal">{s}</p>
          </div>
        ))}
      </div>

      <TrainStrip metrics={metrics} />

      <div className="flex flex-wrap gap-2 px-6 font-mono text-[11px]">
        {(["mcts", "net", "random"] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`border px-3 py-1 ${mode === m ? "border-signal bg-signal text-void" : "border-line text-ash"}`}
          >
            {m === "mcts" ? "net + MCTS" : m === "net" ? "net only (no search)" : "random (ablation)"}
          </button>
        ))}
        <label className="ml-2 flex items-center gap-2 text-ash">
          sims
          <input
            type="number"
            min={8}
            max={256}
            value={sims}
            onChange={(e) => setSims(Number(e.target.value))}
            className="w-16 border border-line bg-panel px-1 py-0.5 text-signal"
          />
        </label>
        <button type="button" onClick={reset} className="border border-line px-3 py-1 text-ash hover:text-signal">
          New game
        </button>
        <button type="button" onClick={runEval} className="border border-line px-3 py-1 text-ash hover:text-signal">
          Arena vs random
        </button>
        {gameId === "othello" && legal.includes(OTHELLO_PASS) ? (
          <button
            type="button"
            disabled={busy}
            onClick={() => playAction(OTHELLO_PASS)}
            className="border border-signal px-3 py-1 text-signal"
          >
            Pass
          </button>
        ) : null}
      </div>

      {error ? <p className="px-6 pt-3 text-sm text-copper">{error}</p> : null}
      {board ? <p className="px-6 pt-3 font-mono text-[11px] text-ash">{board.blurb}</p> : null}

      <div className="mx-auto mt-6 grid w-full max-w-6xl gap-8 px-6 pb-10 lg:grid-cols-[1.1fr_0.9fr]">
        <section>
          <p className="font-mono text-[11px] tracking-widest text-ash">
            YOU ARE LIME · HELIX IS BLUE · {busy ? "SEARCHING" : board?.terminal ? statusLine(board.winner) : "YOUR MOVE"}
          </p>
          <Playfield
            layout={layout}
            grid={grid}
            legal={legal}
            disabled={busy || !!board?.terminal || board?.to_play !== 1}
            onPlay={playAction}
            policy={think?.policy}
          />
          {evalMsg ? <p className="mt-3 font-mono text-xs text-signal">{evalMsg}</p> : null}
          {last?.vs_random ? (
            <p className="mt-2 font-mono text-[11px] text-ash">
              last train eval vs random: {last.vs_random.wins}/{last.vs_random.games} wins
              {last.loss != null ? ` · loss ${last.loss.toFixed(3)}` : ""}
            </p>
          ) : (
            <p className="mt-2 font-mono text-[11px] text-ash">
              python train.py --game {gameId} writes this environment’s checkpoint.
            </p>
          )}
        </section>

        <section>
          <p className="font-mono text-[11px] tracking-widest text-ash">SEARCH / POLICY</p>
          <p className="mt-1 font-mono text-3xl text-signal">{think ? `v ${think.value.toFixed(2)}` : "—"}</p>
          <p className="font-mono text-[11px] text-ash">
            value head from Helix’s seat before the reply
            {think?.pv_labels?.length ? ` · PV ${think.pv_labels.join("→")}` : ""}
          </p>
          <TreeBars rows={think?.tree || []} />
          <p className="mt-4 font-mono text-[11px] leading-relaxed text-ash">
            Same loop on every tab. Ablate search or the net. If MCTS+net is stronger, the architecture did work — not the
            skin.
          </p>
        </section>
      </div>

      <footer className="px-6 py-4 font-mono text-[11px] text-ash/80">{meta?.note}</footer>
    </div>
  );
}

function statusLine(winner: number) {
  if (winner === 0) return "DRAW";
  if (winner === 1) return "YOU WIN";
  return "HELIX WINS";
}

function Playfield({
  layout,
  grid,
  legal,
  disabled,
  onPlay,
  policy,
}: {
  layout: string;
  grid: number[][];
  legal: number[];
  disabled: boolean;
  onPlay: (a: number) => void;
  policy?: number[];
}) {
  if (!grid.length) return null;
  const cols = grid[0].length;
  const maxP = Math.max(...(policy || [0]), 0.04);

  if (layout === "drop") {
    return (
      <div className="mt-4">
        <div className="mb-1 grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
          {Array.from({ length: cols }).map((_, c) => (
            <button
              key={c}
              type="button"
              disabled={disabled || !legal.includes(c)}
              onClick={() => onPlay(c)}
              className="h-8 border border-line bg-panel font-mono text-[10px] text-ash disabled:opacity-30"
            >
              {c}
              {policy ? ` ${((policy[c] || 0) * 100).toFixed(0)}%` : ""}
            </button>
          ))}
        </div>
        <StoneGrid grid={grid} round />
      </div>
    );
  }

  return (
    <div className="mt-4">
      {grid.map((row, r) => (
        <div
          key={r}
          className="mb-1 flex gap-1"
          style={{ marginLeft: layout === "hex" ? (r % 2 ? 14 : 0) : 0 }}
        >
          {row.map((cell, c) => {
            const action = r * cols + c;
            const can = legal.includes(action);
            const prior = policy?.[action] || 0;
            return (
              <button
                key={`${r}-${c}`}
                type="button"
                disabled={disabled || !can}
                onClick={() => onPlay(action)}
                className={`aspect-square w-9 border border-line disabled:cursor-default ${layout === "hex" ? "rounded-full" : ""}`}
                style={{
                  background:
                    cell === 1
                      ? "#c8ff4a"
                      : cell === 2
                        ? "#5b8cff"
                        : can
                          ? `rgba(200,255,74,${0.08 + 0.35 * (prior / maxP)})`
                          : "#161c26",
                }}
                title={can ? `play ${r},${c}` : undefined}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

function StoneGrid({ grid, round }: { grid: number[][]; round?: boolean }) {
  return (
    <div className="grid gap-1 rounded-sm border border-line bg-[#0b1018] p-2">
      {grid.map((row, r) => (
        <div key={r} className="grid gap-1" style={{ gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))` }}>
          {row.map((cell, c) => (
            <div
              key={`${r}-${c}`}
              className={`aspect-square border border-line ${round ? "rounded-full" : ""}`}
              style={{ background: cell === 1 ? "#c8ff4a" : cell === 2 ? "#5b8cff" : "#161c26" }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

function TrainStrip({
  metrics,
}: {
  metrics: { game?: number; loss?: number; vs_random?: { wins: number; games: number } }[];
}) {
  const evals = metrics.filter((m) => m.vs_random && m.game);
  const losses = metrics.filter((m) => m.loss != null && m.game);
  if (!evals.length && !losses.length) return null;
  const maxL = Math.max(...losses.map((m) => m.loss || 0), 0.01);
  const minL = Math.min(...losses.map((m) => m.loss || 0), maxL);
  const pts = losses.map((m, i) => {
    const x = (i / Math.max(losses.length - 1, 1)) * 220;
    const y = 28 - ((m.loss! - minL) / (maxL - minL || 1)) * 24;
    return `${x},${y}`;
  });
  return (
    <div className="mx-6 mb-4 flex flex-wrap items-end gap-6 border border-line bg-panel px-3 py-3">
      <div>
        <p className="font-mono text-[10px] tracking-widest text-ash">ARENA VS RANDOM</p>
        <div className="mt-2 flex gap-4">
          {evals.map((m) => (
            <div key={m.game}>
              <p className="font-mono text-[10px] text-ash">g{m.game}</p>
              <p className="font-mono text-lg text-signal">
                {m.vs_random!.wins}/{m.vs_random!.games}
              </p>
            </div>
          ))}
        </div>
      </div>
      {pts.length > 1 ? (
        <div>
          <p className="font-mono text-[10px] tracking-widest text-ash">SGD LOSS</p>
          <svg width="228" height="32" className="mt-2 block" aria-hidden>
            <polyline fill="none" stroke="#c8ff4a" strokeWidth="1.5" points={pts.join(" ")} />
          </svg>
        </div>
      ) : null}
    </div>
  );
}

function TreeBars({ rows }: { rows: TreeRow[] }) {
  if (!rows.length) {
    return <p className="mt-4 font-mono text-xs text-ash">MCTS visits appear here after Helix moves.</p>;
  }
  const maxN = Math.max(...rows.map((r) => r.n), 1);
  return (
    <ul className="mt-4 space-y-2">
      {rows.map((r) => (
        <li key={r.label || r.col} className="flex items-center gap-3 font-mono text-xs">
          <span className="w-14 text-ash">{r.label ?? `col ${r.col}`}</span>
          <div className="h-4 flex-1 bg-line">
            <div className="h-full bg-signal" style={{ width: `${(r.n / maxN) * 100}%` }} />
          </div>
          <span className="w-28 text-right text-ash">
            N {r.n} · Q {r.q.toFixed(2)}
          </span>
        </li>
      ))}
    </ul>
  );
}
