import sys
import re
from pathlib import Path
import pandas as pd

CSV_PATH = Path("outputs/pigeon/eval_results.csv")
REQUIRES_STAIRS_CSV_PATH = Path("outputs/required_stairs_mp3d.csv")
TIMING_COLUMNS = (
    "map_fusion_sec",
    "target_detection_sec",
    "target_segmentation_sec",
    "vlm_decision_sec",
    "visualization_sec",
    "temporary_ban_maintenance_sec",
    "goal_observation_pool_sec",
    "other_sec",
)
LEGACY_TIMING_COLUMNS = (
    "observation_preprocess_sec",
    "map_fusion_sec",
    "target_detection_sec",
    "target_segmentation_sec",
    "vlm_decision_sec",
    "navigation_action_sec",
)


def normalize_scene_name(scene):
    if pd.isna(scene):
        return ""
    scene = Path(str(scene)).stem
    scene = re.sub(r"\.(basis|semantic)$", "", scene)
    scene = re.sub(r"^\d+-", "", scene)
    return scene


path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH

if not path.exists():
    print(f"File unexists: {path}", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(path)
max_ep = int(sys.argv[2]) if len(sys.argv) > 2 else None
indices = df["episode_id"] < max_ep if max_ep is not None else pd.Series(True, index=df.index)
num_eval_rows = int(indices.sum())

requires_stairs_df = None
requires_stairs_mask = None
if REQUIRES_STAIRS_CSV_PATH.exists():
    requires_stairs_df = pd.read_csv(REQUIRES_STAIRS_CSV_PATH)
    right_episode_col = "dataset_index" if "dataset_index" in requires_stairs_df.columns else "episode_id"
    left_on = ["episode_id"]
    right_on = [right_episode_col]
    if "scene" in df.columns and "scene" in requires_stairs_df.columns:
        df["_scene_key"] = df["scene"].map(normalize_scene_name)
        requires_stairs_df["_scene_key"] = requires_stairs_df["scene"].map(normalize_scene_name)
        left_on.append("_scene_key")
        right_on.append("_scene_key")
    requires_stairs_df = requires_stairs_df[right_on + ["requires_stairs"]].drop_duplicates(subset=right_on)
    df = df.merge(requires_stairs_df, left_on=left_on, right_on=right_on, how="left")
    requires_stairs_mask = indices & (df["requires_stairs"] == 1)

results = {}

results["success_mean"] = float(df["success"][indices].mean())
if requires_stairs_mask is not None and requires_stairs_mask.any():
    results["success_mean_requires_stairs"] = float(df["success"][requires_stairs_mask].mean())
    results["spl_mean_requires_stairs"] = float(df["spl"][requires_stairs_mask].mean())
    results["requires_stairs_episode_count"] = int(requires_stairs_mask.sum())

results["spl_mean"] = float(df["spl"][indices].mean())

results["total_time"] = float(df["elapsed_sec"][indices].sum())

results["worker_num"] = int(df["worker_id"][indices].nunique())

# Profile timing columns are only present when the evaluation was run with
# --profile_timing. A resumed CSV can contain both timing schemas, so use only
# rows that have a complete set of values from one schema.
timing_columns = ()
timing_indices = pd.Series(False, index=df.index)
for columns in (TIMING_COLUMNS, LEGACY_TIMING_COLUMNS):
    if not all(column in df.columns for column in columns):
        continue
    numeric_values = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    candidate_indices = indices & numeric_values.notna().all(axis=1)
    if candidate_indices.any():
        timing_columns = columns
        timing_indices = candidate_indices
        break

if timing_columns:
    for column in timing_columns:
        values = pd.to_numeric(df.loc[timing_indices, column], errors="coerce")
        results[f"{column}_mean"] = float(values.mean())

if "vlm_input_tokens" in df.columns:
    results["vlm_input_tokens_total"] = int(df["vlm_input_tokens"][indices].fillna(0).sum())
if "vlm_output_tokens" in df.columns:
    results["vlm_output_tokens_total"] = int(df["vlm_output_tokens"][indices].fillna(0).sum())

if "success_mean" in results:
    print(f"Success Rate Average: {results['success_mean']:.4f}")
if "success_mean_requires_stairs" in results:
    print(
        "Success Rate Average (requires_stairs=1): "
        f"{results['success_mean_requires_stairs']:.4f} "
        f"({results['requires_stairs_episode_count']} episodes)"
    )
if "spl_mean" in results:
    print(f"SPL Average: {results['spl_mean']:.4f}")
if "distance_to_goal" in df.columns:
    distance_to_goal = pd.to_numeric(df.loc[indices, "distance_to_goal"], errors="coerce").dropna()
    if not distance_to_goal.empty:
        print(
            "Distance to Goal (geodesic, final): "
            f"mean={distance_to_goal.mean():.4f}, "
            f"median={distance_to_goal.median():.4f}, "
            f"min={distance_to_goal.min():.4f}, "
            f"max={distance_to_goal.max():.4f}"
        )
if "spl_mean_requires_stairs" in results:
    print(
        "SPL Average (requires_stairs=1): "
        f"{results['spl_mean_requires_stairs']:.4f} "
        f"({results['requires_stairs_episode_count']} episodes)"
    )
if "worker_num" in results:
    print(f"Worker Number: {results['worker_num']}")
if timing_columns:
    print("Average Timing per Episode (sec):")
    for column in timing_columns:
        mean = results[f"{column}_mean"]
        formatted = "N/A" if pd.isna(mean) else f"{mean:.6f}"
        print(f"  {column}: {formatted}")
if "total_time" in results:
    total_secs = int(round(results["total_time"]))
    print(f"Total Time: {results['total_time']:.2f} sec")
    avg_secs = total_secs // results["worker_num"]
    hours = avg_secs // 3600
    minutes = (avg_secs % 3600) // 60
    seconds = avg_secs % 60
    print(f"Average Time per Worker: {hours:02d}:{minutes:02d}:{seconds:02d}")
if "vlm_input_tokens_total" in results:
    print(f"Total VLM Input Tokens: {results['vlm_input_tokens_total']}")
if "vlm_output_tokens_total" in results:
    print(f"Total VLM Output Tokens: {results['vlm_output_tokens_total']}")
if requires_stairs_df is not None:
    matched = int(df.loc[indices, "requires_stairs"].notna().sum())
    print(f"Matched requires_stairs labels: {matched}/{num_eval_rows}")

# Show probability distribution of failure causes across all episodes
if "failure_cause" in df.columns:
    counts = df["failure_cause"][indices].fillna("Unknown").value_counts()
    total = num_eval_rows
    print("Failure Cause Distribution (all episodes):")
    for cause, cnt in counts.items():
        prob = cnt / total if total > 0 else 0.0
        print(f"  - {cause}: {prob:.3f} ({cnt}/{total})")
else:
    print("Column 'failure_cause' not found in CSV; skipping distribution.")

# Per-scene success rate + SPL mean
if "scene" in df.columns:
    per_scene = (
        df.loc[indices]
        .groupby("scene")
        .agg(success_rate=("success", "mean"), spl_mean=("spl", "mean"), episodes=("episode_id", "size"))
        .sort_index()
    )
    print("Per-scene success rate + SPL mean (all episodes):")
    print(per_scene.to_string(float_format=lambda x: f"{x:.4f}"))

if requires_stairs_mask is not None and "scene" in df.columns:
    per_scene_requires_stairs = (
        df.loc[requires_stairs_mask]
        .groupby("scene")
        .agg(success_rate=("success", "mean"), spl_mean=("spl", "mean"), episodes=("episode_id", "size"))
        .sort_index()
    )
    print("Per-scene success rate + SPL mean (requires_stairs=1):")
    print(per_scene_requires_stairs.to_string(float_format=lambda x: f"{x:.4f}"))
