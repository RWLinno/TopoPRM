from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


class RewardConfig:
    # Length reward
    LENGTH_LOW = env_int("TOPO_LENGTH_LOW", 2000)
    LENGTH_HIGH = env_int("TOPO_LENGTH_HIGH", 4000)

    # Continuity reward
    CONTINUITY_BROKEN_CHAIN_PENALTY = env_float("TOPO_CONTINUITY_BROKEN_CHAIN_PENALTY", 0.8)

    # Topology reward weights / gating
    TOPO_W_VALID = env_float("TOPO_W_VALID", 0.20)
    TOPO_W_ACYCLIC = env_float("TOPO_W_ACYCLIC", 0.15)
    TOPO_W_NO_ORPHAN = env_float("TOPO_W_NO_ORPHAN", 0.15)
    TOPO_W_DIRECTION = env_float("TOPO_W_DIRECTION", 0.15)
    TOPO_W_STEP_ALIGN = env_float("TOPO_W_STEP_ALIGN", 0.10)
    TOPO_W_REF_EDGE_F1 = env_float("TOPO_W_REF_EDGE_F1", 0.25)
    # Formula-aligned lambda terms for:
    # r_topo = λ_b*I[|V|>0] + λ_a*I[acyclic] + λ_o*I[rho_orphan=0] + λ_d*delta + λ_k*kappa
    TOPO_LAMBDA_BASE = env_float("TOPO_LAMBDA_BASE", TOPO_W_VALID)
    TOPO_LAMBDA_ACYCLIC = env_float("TOPO_LAMBDA_ACYCLIC", TOPO_W_ACYCLIC)
    TOPO_LAMBDA_ORPHAN = env_float("TOPO_LAMBDA_ORPHAN", TOPO_W_NO_ORPHAN)
    TOPO_LAMBDA_DELTA = env_float("TOPO_LAMBDA_DELTA", TOPO_W_DIRECTION)
    TOPO_LAMBDA_KAPPA = env_float("TOPO_LAMBDA_KAPPA", TOPO_W_REF_EDGE_F1)
    TOPO_REQUIRE_VALID_DAG = env_bool("TOPO_REQUIRE_VALID_DAG", True)
    TOPO_VERIFY_LOG_EVERY = env_int("TOPO_VERIFY_LOG_EVERY", 0)

    # Composite dynamic weighting (ABLATION ONLY, default OFF post 2026-04-23
    # moderate cleanup: the dynamic-weighting scheme is not described in the
    # paper's method section so we keep it disabled by default; set
    # TOPO_DYNAMIC_REWARD=1 to re-enable for ablation experiments).
    TOPO_REWARD_LOG_EVERY = env_int("TOPO_REWARD_LOG_EVERY", 10)
    TOPO_DYNAMIC_REWARD = env_bool("TOPO_DYNAMIC_REWARD", False)
    TOPO_DYNAMIC_ETA = env_float("TOPO_DYNAMIC_ETA", 0.50)
    TOPO_DYNAMIC_MIN_WEIGHT = env_float("TOPO_DYNAMIC_MIN_WEIGHT", 0.05)
    TOPO_DYNAMIC_OUTCOME_FLOOR = env_float("TOPO_DYNAMIC_OUTCOME_FLOOR", 0.35)

    # Hierarchical reward controls.
    # ALPHA (topo vs continuity mix) is the single principal hyperparameter
    # appearing in the paper's Eq. R_hier.
    TOPO_HIER_ALPHA = env_float("TOPO_HIER_ALPHA", 0.60)
    # Anti-collapse hacks below default to OFF post 2026-04-23 moderate cleanup:
    # ms-swift GRPO already performs group-wise advantage normalization
    # (scale_rewards='group' is its default), so explicit std-floor noise
    # injection is redundant.  Kept for ablation, not for main runs.
    TOPO_HIER_NOISE_EPS = env_float("TOPO_HIER_NOISE_EPS", 0.0)
    TOPO_HIER_MIN_STD = env_float("TOPO_HIER_MIN_STD", 0.005)
    # Reward temperature rescaling (=1 means identity).  No formal motivation
    # in the paper; default neutralised.
    TOPO_HIER_REWARD_TEMP = env_float("TOPO_HIER_REWARD_TEMP", 1.0)
    # Floor on the multiplicative base term so that topology gain is never zero-ed
    # out when r_base = 0 (outcome=format=length=0).  This is a BUG FIX and is
    # documented in the paper appendix as a zero-variance-group remedy.  Set to 0
    # only for ablation purposes.
    TOPO_HIER_BASE_FLOOR = env_float("TOPO_HIER_BASE_FLOOR", 0.05)

    # Shared anti-collapse controls for Gated / Composite (off by default;
    # see Hier rationale above).
    TOPO_GATED_NOISE_EPS = env_float("TOPO_GATED_NOISE_EPS", 0.0)
    TOPO_GATED_MIN_STD = env_float("TOPO_GATED_MIN_STD", 0.001)
    TOPO_COMPOSITE_NOISE_EPS = env_float("TOPO_COMPOSITE_NOISE_EPS", 0.0)
    TOPO_COMPOSITE_MIN_STD = env_float("TOPO_COMPOSITE_MIN_STD", 0.001)
