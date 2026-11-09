"""Opt-in random seeds for paired evaluations, independent of worker ordering."""

import os
import random

import numpy as np


def seed_episode(episode_index):
    raw_seed = os.environ.get("PIGEON_EVAL_SEED", "").strip()
    if not raw_seed:
        return None

    import torch

    seed = (int(raw_seed) + int(episode_index)) % (2**63 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    return seed


def policy_action(policy, depth, goal, sampling_seed=None):
    if sampling_seed is None:
        return policy.act(depth=depth, pointgoal_rho_theta=goal)

    import torch

    # Keep stochastic action sampling, but prevent other workers/model loads
    # from consuming this request's random numbers (or vice versa).
    device = torch.device(policy.device)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(sampling_seed))
        return policy.act(depth=depth, pointgoal_rho_theta=goal)
