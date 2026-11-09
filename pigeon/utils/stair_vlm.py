from typing import Any


STAIR_DIRECTION_PROMPT = (
        """You are an indoor navigation assistant reviewing the robot's current view.\n"""
        """Decide whether the visible staircase leads up or down from the robot's current floor.\n"""
        """Respond with exactly one label:\n"""
        """- up: a staircase clearly goes up from current floor to somewhere higher.\n"""
        """- down: a staircase or drop-off clearly goes down from current floor to somewhere lower.\n"""
        """- unconfirmed: no staircase is visible, or the direction is unclear.\n"""
        """Please follow the format like this, without bracket --\n"""
        """<thought> [put your step-by-step thinking process here] </thought>\n"""
        """<answer> [put up, down, or unconfirmed here] </answer>\n"""
)

STAIR_DIRECTION_FAILED_RESPONSE = "<answer> unconfirmed </answer>"


def parse_stair_direction_response(response: Any) -> str:
    try:
        answer = str(response).split("<answer>")[1].split("</answer>")[0].strip().lower()
    except Exception:
        answer = str(response or "").strip().lower()
    if answer in ("upstairs", "upstair", "up"):
        return "upstair"
    if answer in ("downstairs", "downstair", "down"):
        return "downstair"
    return "unconfirmed"
