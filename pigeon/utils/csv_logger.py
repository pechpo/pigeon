import csv
from pathlib import Path
from typing import Dict, Optional, Set


class CSVLogger:
    """Append-only CSV logger with resume support.

    - Creates directories as needed.
    - Keeps a small in-memory set of episode IDs that already finished.
    - Opens the file in append mode per write to reduce risk of data loss.
    """

    BASE_HEADER = [
        "episode_id",
        "worker_id",
        "steps",
        "success",
        "spl",
        "elapsed_sec",
        "failure_cause",
        "traveled_distance",
        "distance_to_goal",
        "downstair_detected",
        "upstair_detected",
        "vlm_calls",
        "vlm_input_tokens",
        "vlm_output_tokens",
        "banned_points",
        "scene",
        "target",
        "stop_x",
        "stop_y",
        "nav_goal_x",
        "nav_goal_y",
    ]
    HEADER = BASE_HEADER

    def __init__(self, csv_path: str) -> None:
        self.path = Path(csv_path)
        self._desired_header = self.HEADER
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._desired_header)
        else:
            # If existing file is missing columns, upgrade in-place
            self._ensure_column("failure_cause")
            self._ensure_column("traveled_distance")
            self._ensure_column("distance_to_goal")
            self._ensure_column("downstair_detected")
            self._ensure_column("upstair_detected")
            self._ensure_column("vlm_calls")
            self._ensure_column("vlm_input_tokens")
            self._ensure_column("vlm_output_tokens")
            self._ensure_column("banned_points")
            self._ensure_column("scene")
            self._ensure_column("target")
            self._ensure_column("stop_x")
            self._ensure_column("stop_y")
            self._ensure_column("nav_goal_x")
            self._ensure_column("nav_goal_y")
        self._header = self._read_header()

    def _read_header(self) -> list[str]:
        try:
            with self.path.open("r", newline="") as f:
                return next(csv.reader(f), list(self._desired_header))
        except Exception:
            return list(self._desired_header)

    def _ensure_column(self, column: str) -> None:
        try:
            # Read entire file and detect header
            with self.path.open("r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                # Empty file, write header fresh
                with self.path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self._desired_header)
                return
            header = rows[0]
            if column in header:
                return  # Already present
            # Append missing column to header and pad existing rows with empty string
            new_header = header + [column]
            new_rows = [new_header]
            for r in rows[1:]:
                new_rows.append(r + [""])
            with self.path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)
        except Exception:
            # Best-effort: don't block logging if upgrade fails
            pass

    def existing_episode_ids(self) -> Set[str]:
        if not self.path.exists():
            return set()
        ids: Set[str] = set()
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_id = row.get("episode_id")
                if ep_id:
                    ids.add(ep_id)
        return ids

    def existing_episode_counts(self) -> Dict[str, int]:
        """Return a count of how many times each episode_id appears in the CSV.

        This is used for multi-epoch resume: if an evaluation runs for N epochs and
        logs the same episode_id once per epoch, then the count indicates how many
        epochs of that episode have already finished.
        """
        if not self.path.exists():
            return {}
        counts: Dict[str, int] = {}
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_id = row.get("episode_id")
                if not ep_id:
                    continue
                counts[ep_id] = counts.get(ep_id, 0) + 1
        return counts

    def append(
        self,
        episode_id: str,
        worker_id: int,
        steps: int,
        success: int,
        spl: float,
        elapsed_sec: float,
        failure_cause: str,
        traveled_distance: float,
        distance_to_goal: Optional[float],
        downstair_detected: bool,
        upstair_detected: bool,
        vlm_calls: int,
        vlm_input_tokens: int,
        vlm_output_tokens: int,
        banned_points: int,
        scene: str = "",
        target: str = "",
        stop_x: Optional[float] = None,
        stop_y: Optional[float] = None,
        nav_goal_x: Optional[float] = None,
        nav_goal_y: Optional[float] = None,
    ) -> None:
        # Open in append mode per write to avoid buffering a lot of rows in memory
        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            values = {
                "episode_id": episode_id,
                "worker_id": worker_id,
                "steps": steps,
                "success": success,
                "spl": f"{spl:.3f}",
                "elapsed_sec": f"{elapsed_sec:.3f}",
                "failure_cause": failure_cause,
                "traveled_distance": f"{traveled_distance:.3f}",
                "distance_to_goal": (
                    "" if distance_to_goal is None else f"{float(distance_to_goal):.3f}"
                ),
                "downstair_detected": int(bool(downstair_detected)),
                "upstair_detected": int(bool(upstair_detected)),
                "vlm_calls": int(vlm_calls),
                "vlm_input_tokens": int(vlm_input_tokens),
                "vlm_output_tokens": int(vlm_output_tokens),
                "banned_points": int(banned_points),
                "scene": scene,
                "target": target,
                "stop_x": "" if stop_x is None else f"{float(stop_x):.4f}",
                "stop_y": "" if stop_y is None else f"{float(stop_y):.4f}",
                "nav_goal_x": "" if nav_goal_x is None else f"{float(nav_goal_x):.4f}",
                "nav_goal_y": "" if nav_goal_y is None else f"{float(nav_goal_y):.4f}",
            }
            writer.writerow([values.get(column, "") for column in self._header])
