from app.game import COLS, ConnectFour, P1
from app.games import catalog, make_game
from app.mcts import MCTS
from app.net import PolicyValueNet
from app.selfplay import play_game, train_step
from torch.optim import Adam

from app.main import app
from fastapi.testclient import TestClient


def test_win_horizontal() -> None:
    s = ConnectFour()
    for c in range(4):
        s = s.drop(c)
        if c < 3:
            s = s.drop(6)
    assert s.outcome().terminal and s.outcome().winner == P1


def test_gomoku_five() -> None:
    s = make_game("gomoku")
    # P1 on row 4 cols 0-4, P2 elsewhere
    for i in range(4):
        s = s.play(4 * 8 + i)
        s = s.play(0 * 8 + i)
    s = s.play(4 * 8 + 4)
    assert s.outcome().terminal and s.outcome().winner == P1


def test_hex_left_right() -> None:
    s = make_game("hex")
    # P1 fills a left-right path on row 2; P2 plays row 0
    for c in range(6):
        s = s.play(2 * 6 + c)
        if c < 5:
            s = s.play(c)
    assert s.outcome().terminal and s.outcome().winner == P1


def test_othello_opening_legal() -> None:
    s = make_game("othello")
    legal = s.legal_moves()
    assert legal
    s = s.play(legal[0])
    assert s.board.sum() > 4  # a flip happened


def test_mcts_legal() -> None:
    net = PolicyValueNet()
    s = ConnectFour()
    pi, root = MCTS(net, sims=16).search(s)
    assert abs(pi.sum() - 1) < 1e-5
    assert set(root.edges) == set(range(COLS))


def test_selfplay_and_sgd() -> None:
    net = PolicyValueNet()
    opt = Adam(net.parameters(), lr=1e-3)
    samples = play_game(net, sims=8, temperature=1.0, game_id="connect4")
    assert samples
    stats = train_step(net, samples[: min(8, len(samples))], opt)
    assert "loss" in stats


def test_health_and_catalog() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] == "helix"
        games = client.get("/games").json()["games"]
        assert {g["id"] for g in games} == {"connect4", "gomoku", "hex", "othello"}
        meta = client.get("/meta", params={"game": "gomoku"})
        assert meta.status_code == 200
        assert meta.json()["game"] == "gomoku"
        board = client.get("/board", params={"game": "hex"}).json()
        assert board["rows"] == 6
        th = client.post(
            "/think",
            json={"game": "connect4", "grid": client.get("/board").json()["grid"], "to_play": 1, "sims": 12, "mode": "net"},
        )
        assert th.status_code == 200
        assert th.json()["choice"] in range(7)
        catalog_ok = catalog()
        assert len(catalog_ok) == 4
