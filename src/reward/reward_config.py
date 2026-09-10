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

    # P5: unit used for the length reward. 'chars' preserves the released
    # behaviour (len(text)); 'tokens' switches to whitespace-delimited token
    # counts so thresholds align with generation budgets on long-CoT traces.
    LENGTH_UNIT = os.environ.get("TOPO_LENGTH_UNIT", "chars")

    # Continuity reward
    CONTINUITY_BROKEN_CHAIN_PENALTY = env_float("TOPO_CONTINUITY_BROKEN_CHAIN_PENALTY", 0.8)

    # P3: when a step contains no extractable expression AND no claim key,
    # the default implementation counts it as 'continuous' (benefit of the
    # doubt).  On natural-language CoT traces this silently drives q_cont
    # towards 1.0 and destroys the gradient.  Setting this flag requires
    # *evidence* (expression-overlap or claim-overlap or a 'given' marker)
    # to count a step as continuous, so ambient-text-only steps count as
    # broken and q_cont drops.  Default OFF to preserve v1 behaviour.
    CONTINUITY_REQUIRE_EVIDENCE = env_bool("TOPO_CONT_REQUIRE_EVIDENCE", False)

    # Topology reward weights / gating
    TOPO_W_VALID = env_float("TOPO_W_VALID", 0.20)
    TOPO_W_ACYCLIC = env_float("TOPO_W_ACYCLIC", 0.15)
    TOPO_W_NO_ORPHAN = env_float("TOPO_W_NO_ORPHAN", 0.15)
    TOPO_W_DIRECTION = env_float("TOPO_W_DIRECTION", 0.15)
    TOPO_W_STEP_ALIGN = env_float("TOPO_W_STEP_ALIGN", 0.10)
    TOPO_W_REF_EDGE_F1 = env_float("TOPO_W_REF_EDGE_F1", 0.25)
    # Formula-aligned terms: graph existence, coverage-adjusted acyclicity,
    # no-orphan support, coverage-adjusted direction, and optional reference F1.
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

    # ------------------------------------------------------------------
    # Diagnostic patches (see the method notes)
    # All default OFF.  Currently released checkpoints reproduce byte-for-byte
    # when these are unset.  v2 retraining flips them on.
    # ------------------------------------------------------------------
    # P0: do not min-max stretch low-variance topology/continuity scores into
    # full [0, 1].  When intra-group spread is below TOPO_RESCALE_MIN_SPAN,
    # return a constant 0.5 vector instead.  Prevents topology noise from
    # dominating advantages on outcome-saturated batches (easy GSM8K/MATH500).
    TOPO_RESCALE_PATCH = env_bool("TOPO_RESCALE_PATCH", False)
    TOPO_RESCALE_MIN_SPAN = env_float("TOPO_RESCALE_MIN_SPAN", 0.05)
    # P2: switch from additive `r_base = w_o*o + w_f*f + w_l*l` to truly
    # multiplicative aggregation `r_base = o * (w_f*f + w_l*l + slack)` so
    # outcome=0 implies r_base=0 (correctness primacy is hard, not soft).
    TOPO_HIER_AGG = os.environ.get("TOPO_HIER_AGG", "additive")  # 'additive' | 'multiplicative'

    # P4: when extract_steps_from_answer finds zero explicit step markers
    # (Step N:, (1), bullet, etc.), fall back to sentence-level segmentation
    # so that natural-language CoT traces still produce a multi-node DAG
    # with meaningful q_topo.  Default OFF to preserve v1 behaviour.
    DAG_SENTENCE_FALLBACK = env_bool("TOPO_DAG_SENTENCE_FALLBACK", False)
    DAG_SENTENCE_MIN_LEN = env_int("TOPO_DAG_SENTENCE_MIN_LEN", 20)
    DAG_EXTRA_STEP_MARKERS = env_bool("TOPO_DAG_EXTRA_STEP_MARKERS", False)
    DAG_LATEX_EXPR = env_bool("TOPO_DAG_LATEX_EXPR", False)
    DAG_BARRIER_STRICT = env_bool("TOPO_DAG_BARRIER_STRICT", False)
    DAG_BARRIER_MIN_OVERLAP = env_float("TOPO_DAG_BARRIER_MIN_OVERLAP", 0.20)
    DAG_RAW_DIRECTED = env_bool("TOPO_DAG_RAW_DIRECTED", False)

    # Hybrid DAG extraction: rule-based bootstrap plus local pretrained-LLM
    # refinement for implicit semantic dependencies.  This is intended for
    # offline data preprocessing / DAG caching.  Keep it OFF in online GRPO
    # reward calls and evaluation unless cached DAGs are being materialized;
    # otherwise every completion would trigger an LLM forward pass.
    DAG_LLM_REFINE = env_bool("TOPO_DAG_LLM_REFINE", False)
    DAG_LLM_MODEL = os.environ.get(
        "TOPO_DAG_LLM_MODEL",
        os.path.expandvars("${MODEL_ROOT}/Qwen/Qwen2.5-Math-1.5B-Instruct"),
    )
    DAG_LLM_DEVICE = os.environ.get("TOPO_DAG_LLM_DEVICE", "auto")
    DAG_LLM_MAX_STEPS = env_int("TOPO_DAG_LLM_MAX_STEPS", 16)
    DAG_LLM_MAX_NEW_TOKENS = env_int("TOPO_DAG_LLM_MAX_NEW_TOKENS", 512)

    # Optional low-variance stabilizer for topology rewards. When enabled, a
    # per-batch post-pass reweights lambda terms by observed component spread.
    TOPO_QTOPO_SELF_NORM = env_bool("TOPO_QTOPO_SELF_NORM", False)
    TOPO_QTOPO_TARGET_VAR = env_float("TOPO_QTOPO_TARGET_VAR", 0.05)
    TOPO_QTOPO_MIN_SPREAD = env_float("TOPO_QTOPO_MIN_SPREAD", 1e-3)

    # P1: TopoSCAEReward — preserve outcome magnitude across strata so
    # that B+ rewards are always > B- rewards in absolute value.  Default
    # OFF (released checkpoints use TopoHierarchicalReward, not SCAE).
    SCAE_PRESERVE_OUTCOME = env_bool("TOPO_SCAE_PRESERVE_OUTCOME", False)
    SCAE_FLOOR_POS = env_float("TOPO_SCAE_FLOOR_POS", 0.3)
    SCAE_FLOOR_NEG = env_float("TOPO_SCAE_FLOOR_NEG", 0.3)

    # ------------------------------------------------------------------
    # Shared anti-collapse controls for Gated / Composite (off by default;
    # see Hier rationale above).
    # ------------------------------------------------------------------
    TOPO_GATED_NOISE_EPS = env_float("TOPO_GATED_NOISE_EPS", 0.0)
    TOPO_GATED_MIN_STD = env_float("TOPO_GATED_MIN_STD", 0.001)
    TOPO_COMPOSITE_NOISE_EPS = env_float("TOPO_COMPOSITE_NOISE_EPS", 0.0)
    TOPO_COMPOSITE_MIN_STD = env_float("TOPO_COMPOSITE_MIN_STD", 0.001)
