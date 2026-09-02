# Helix

The **AlphaZero loop**, not a chatbot and not a dashboard with a model glued on.

One engine, four rulesets. Self-play generates games. A replay buffer stores `(board, search policy, outcome)`. SGD fits a **policy-value network per game**. At test time **MCTS** uses that net as prior and leaf evaluator. An arena scores the checkpoint against random. That is the architecture DeepMind used for Go, sized to a laptop CPU.

| Environment | Board | What is different |
| --- | --- | --- |
| Connect Four | 6×7 | gravity, 4-in-a-row, 7 actions |
| Gomoku | 8×8 | free placement, 5-in-a-row, 64 actions |
| Hex | 6×6 | 6-neighbor connection (P1 left–right, P2 top–bottom) |
| Othello | 6×6 | captures / disc count, pass if no flip |

Same loop on every tab. If `net + MCTS` beats `net only` beats `random`, search is doing the work.

This is still not AlphaZero on 19×19 Go. Claiming that would be a lie.

## Walkthrough

1. Open [http://localhost:3000](http://localhost:3000)
2. Switch **Connect Four / Gomoku / Hex / Othello**
3. You are lime. Helix is blue.
4. Read the search tree (N, Q) and the principal variation
5. Ablate **net only** and **random**
6. **Arena vs random** for the active environment

## Train

```bash
cd api
python train.py --game connect4 --games 80 --sims 32
python train.py --game gomoku --games 40 --sims 24
python train.py --game hex --games 40 --sims 24
python train.py --game othello --games 40 --sims 24
```

Checkpoints: `api/artifacts/helix-{game}.pt`

## Run locally

```bash
cd api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
cd web
npm install
copy .env.example .env.local
npm run dev
```

## Tests

```bash
cd api
pytest -q
```

## License

Personal portfolio. The algorithm follows the AlphaZero papers (Silver et al.). The four rulesets are public-domain.
