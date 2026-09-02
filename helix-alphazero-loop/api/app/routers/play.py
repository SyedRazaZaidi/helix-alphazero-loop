from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..engine import apply_move, checkpoint_info, empty_board, metrics_tail, quick_eval, think
from ..games import FACTORIES, catalog

router = APIRouter(tags=["helix"])


def _gid(game: str) -> str:
    if game not in FACTORIES:
        raise HTTPException(400, f"unknown game {game}")
    return game


class ThinkIn(BaseModel):
    game: str = "connect4"
    grid: list[list[int]]
    to_play: int = 1
    sims: int = Field(default=48, ge=8, le=256)
    mode: str = "mcts"


class MoveIn(BaseModel):
    game: str = "connect4"
    grid: list[list[int]]
    to_play: int = 1
    action: int | None = None
    col: int | None = None


@router.get("/games")
def games() -> dict:
    return {"games": catalog()}


@router.get("/meta")
def meta(game: str = Query(default="connect4")) -> dict:
    return checkpoint_info(_gid(game))


@router.get("/board")
def board(game: str = Query(default="connect4")) -> dict:
    return empty_board(_gid(game))


@router.get("/metrics")
def metrics(game: str = Query(default="connect4")) -> dict:
    gid = _gid(game)
    rows = metrics_tail(gid)
    return {"game": gid, "rows": rows, "n": len(rows)}


@router.post("/think")
def post_think(body: ThinkIn) -> dict:
    if body.mode not in {"mcts", "net", "random"}:
        raise HTTPException(400, "mode must be mcts | net | random")
    try:
        return think(body.grid, body.to_play, sims=body.sims, mode=body.mode, game_id=_gid(body.game))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/move")
def post_move(body: MoveIn) -> dict:
    action = body.action if body.action is not None else body.col
    if action is None:
        raise HTTPException(400, "action required")
    try:
        return apply_move(body.grid, body.to_play, int(action), game_id=_gid(body.game))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/eval")
def eval_random(game: str = Query(default="connect4")) -> dict:
    return quick_eval(_gid(game))
