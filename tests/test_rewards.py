import json

import pytest

from src.capacity_profiles import json_capacity_profiles
from src.reward.outcome_reward import OutcomeReward
from src.reward.format_reward import FormatReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from src.eval.unified_benchmark import (
    bootstrap_item_metrics,
    evaluate_predictions,
    paired_bootstrap_difference,
)
from src.reward.composite_reward import (
    LengthReward,
    MatchedLaserDReward,
    OutcomeFormatLengthReward,
    TopoChoquetReward,
    TopoCompositeReward,
    TopoEqualAdditiveReward,
    TopoMatchedAdditiveReward,
    TopoMatchedMultiplicativeReward,
    TopoIndependentChoquetReward,
    TopoIndependentMatchedAdditiveReward,
    TopoIndependentHeroReward,
)


def _wrap(text: str):
    """Wrap text into the ``completions`` format expected by reward callables."""
    return [[{"content": text}]]


# ---------------------------------------------------------------------------
# OutcomeReward
# ---------------------------------------------------------------------------

class TestOutcomeReward:
    def setup_method(self):
        self.reward = OutcomeReward()

    def test_exact_match(self):
        text = r"<think>Compute the result.</think>Final answer: \boxed{5}"
        solution = "5"
        scores = self.reward(_wrap(text), solution=solution)
        assert scores == [1.0]

    def test_close_match(self):
        text = r"<think>Simplify the fraction.</think>Final answer: \boxed{\frac{1}{2}}"
        solution = "0.5"
        scores = self.reward(_wrap(text), solution=solution)
        assert scores == [1.0]

    def test_no_match(self):
        text = r"<think>Compute the result.</think>Final answer: \boxed{4}"
        solution = "5"
        scores = self.reward(_wrap(text), solution=solution)
        assert scores == [0.0]

    def test_missing_solution(self):
        text = r"<think>Compute the result.</think>Final answer: \boxed{5}"
        scores = self.reward(_wrap(text), solution=None)
        assert scores == [0.0]

    def test_multiple_completions(self):
        completions = [
            [{"content": r"Final answer: \boxed{5}"}],
            [{"content": r"Final answer: \boxed{3}"}],
        ]
        solution = "5"
        scores = self.reward(completions, solution=solution)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_nested_boxed_extraction(self):
        text = r"First \boxed{0}; final \boxed{\frac{1}{\sqrt{2}}}."
        assert self.reward._extract_last_boxed(text) == r"\frac{1}{\sqrt{2}}"

    def test_empty_final_box_is_ignored(self):
        text = r"First \boxed{7}; malformed final \boxed{}."
        assert self.reward._extract_last_boxed(text) == "7"

    def test_tuple_equivalence_and_gold_order(self):
        assert self.reward._verify_equivalence(
            r"(3, \frac{\pi}{2})",
            r"\left(3, \frac{\pi}{2}\right)",
        )
        assert not self.reward._verify_equivalence("1", "1,-2")

    def test_math_response_uses_last_nonempty_box(self):
        response = r"Candidate \boxed{7}; final \boxed{\frac{1}{2}}."
        assert self.reward.verify_math_response(response, "0.5")
        assert not self.reward.verify_math_response(response, "7")

    def test_math_response_accepts_explicit_unboxed_final_answer(self):
        response = r"Work through the derivation. Final answer: \frac{1}{2}."
        assert self.reward.verify_math_response(response, "0.5")

    def test_math_response_rejects_intermediate_match_in_truncated_trace(self):
        response = r"An intermediate value is \frac{1}{2}. Continue by considering"
        assert not self.reward.verify_math_response(response, "0.5")

    def test_math_response_uses_last_explicit_final_answer(self):
        response = r"The answer is 7. Correction: final answer = \frac{1}{2}."
        assert self.reward.verify_math_response(response, "0.5")
        assert not self.reward.verify_math_response(response, "7")

    def test_math_response_rejects_nonterminal_final_answer_planning(self):
        response = (
            "I plan to state final answer: 5. "
            + "I must continue checking this derivation carefully. " * 20
        )
        assert not self.reward.verify_math_response(response, "5")

    def test_eval_uses_the_same_symbolic_matcher_for_all_k(self):
        metrics = evaluate_predictions(
            predictions_per_item=[[r"\frac{1}{2}", "0.5", "7"]],
            gold_answers=["0.5"],
            answer_extractor=lambda value: value,
            answer_matcher=lambda pred, gold: bool(
                pred is not None
                and OutcomeReward._verify_equivalence(str(pred), str(gold))
            ),
            k_values=[1, 3],
            token_counts=[[4, 5, 6]],
        )
        assert metrics["pass@1"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["pass@3"] == 1.0
        assert metrics["maj@3"] == 1.0
        assert metrics["avg_tokens"] == 5.0

    def test_bootstrap_uses_shared_item_ids(self):
        baseline = [
            {"item_id": "a", "correct_pass1": False, "gen_tokens_pass1": 20},
            {"item_id": "b", "correct_pass1": True, "gen_tokens_pass1": 30},
        ]
        candidate = [
            {"item_id": "b", "correct_pass1": True, "gen_tokens_pass1": 20},
            {"item_id": "a", "correct_pass1": True, "gen_tokens_pass1": 10},
        ]
        marginal = bootstrap_item_metrics(candidate, n_resamples=100, seed=0)
        paired = paired_bootstrap_difference(
            candidate, baseline, n_resamples=100, seed=0
        )
        assert marginal["accuracy_pct"] == 100.0
        assert marginal["accuracy_pct_wilson95"][0] < 100.0
        assert marginal["mean_tokens"] == 15.0
        assert paired["accuracy_delta_pp"] == 50.0
        assert paired["mean_token_delta"] == -10.0

        with pytest.raises(ValueError, match="identical item_id sets"):
            paired_bootstrap_difference(
                candidate[:1], baseline, n_resamples=10, seed=0
            )

    def test_wilson_interval_does_not_collapse_at_zero_accuracy(self):
        rows = [
            {"item_id": str(i), "correct_pass1": False, "gen_tokens_pass1": 10}
            for i in range(30)
        ]
        marginal = bootstrap_item_metrics(rows, n_resamples=10, seed=0)
        assert marginal["accuracy_pct_ci95"] == [0.0, 0.0]
        assert marginal["accuracy_pct_wilson95"][0] == pytest.approx(0.0)
        assert marginal["accuracy_pct_wilson95"][1] > 10.0


# ---------------------------------------------------------------------------
# FormatReward
# ---------------------------------------------------------------------------

class TestFormatReward:
    def setup_method(self):
        self.reward = FormatReward()

    def test_perfect_format(self):
        text = r"<think>Work through the steps.</think>Final answer: \boxed{5}"
        assert self.reward(_wrap(text)) == [1.0]

    def test_partial_no_think(self):
        text = r"Final answer: \boxed{5}"
        assert self.reward(_wrap(text)) == [0.5]

    def test_missing_tags(self):
        text = "这是一段普通文本，没有任何标签"
        assert self.reward(_wrap(text)) == [0.0]

    def test_invalid_answer_json(self):
        text = "<think>Reasoning without a final boxed answer.</think>"
        assert self.reward(_wrap(text)) == [0.3]

    def test_prefilled_protocol_accepts_prompt_owned_open_tag(self, monkeypatch):
        monkeypatch.setenv("TOPO_FORMAT_PROTOCOL", "prefilled_think")
        reward = FormatReward()
        text = r"Derive the result.</think>Final answer: \boxed{5}"
        assert reward(_wrap(text)) == [1.0]

    def test_prefilled_protocol_rejects_generated_open_tags(self, monkeypatch):
        monkeypatch.setenv("TOPO_FORMAT_PROTOCOL", "prefilled_think")
        reward = FormatReward()
        repeated = r"<think><think>Derive.</think>Final answer: \boxed{5}"
        assert reward(_wrap(repeated)) == [0.0]

    def test_full_protocol_requires_one_complete_block(self, monkeypatch):
        monkeypatch.setenv("TOPO_FORMAT_PROTOCOL", "full_think")
        reward = FormatReward()
        valid = r"<think>Derive the result.</think>Final answer: \boxed{5}"
        missing_open = r"Derive the result.</think>Final answer: \boxed{5}"
        assert reward(_wrap(valid)) == [1.0]
        assert reward(_wrap(missing_open)) == [0.0]


# ---------------------------------------------------------------------------
# TopoReward
# ---------------------------------------------------------------------------

class TestTopoReward:
    def setup_method(self):
        self.reward = TopoReward()

    def test_good_reasoning(self):
        think = "∵ x = 1\n∴ y = x + 1\n故 y = 2"
        text = f"<think>{think}</think><answer>ok</answer>"
        scores = self.reward(_wrap(text))
        assert scores[0] > 0.0

    def test_empty_think(self):
        text = "<think></think><answer>ok</answer>"
        assert self.reward(_wrap(text)) == [0.0]

    def test_no_tags(self):
        text = "普通文本没有think标签"
        assert self.reward(_wrap(text)) == [0.0]

    def test_acyclic_bonus(self):
        think = "∵ a = 1\n∴ b = 2\n故 c = 3"
        text = f"<think>{think}</think><answer>ok</answer>"
        scores = self.reward(_wrap(text))
        assert scores[0] >= 0.6  # base 0.4 + acyclic 0.2

    def test_formula_terms_consistency(self):
        think = "设 x=1\n由 x=1 得 y=2\n故 y=2"
        text = f"<think>{think}</think><answer>ok</answer>"
        scores = self.reward(_wrap(text))
        assert len(scores) == 1
        assert self.reward.last_diagnostics
        diag = self.reward.last_diagnostics[0]
        required = {
            "lambda_base", "lambda_acyclic", "lambda_orphan", "lambda_delta", "lambda_kappa",
            "rho_orphan", "delta", "kappa",
            "term_base", "term_acyclic", "term_orphan", "term_delta", "term_kappa",
            "denom", "r_topo",
        }
        assert required.issubset(set(diag.keys()))
        recomputed = (
            diag["term_base"]
            + diag["term_acyclic"]
            + diag["term_orphan"]
            + diag["term_delta"]
            + diag["term_kappa"]
        ) / max(diag["denom"], 1e-12)
        assert diag["r_topo"] == pytest.approx(recomputed, abs=1e-9)


# ---------------------------------------------------------------------------
# ContinuityReward
# ---------------------------------------------------------------------------

class TestContinuityReward:
    def setup_method(self):
        self.reward = ContinuityReward()

    def test_no_think_block(self):
        text = "没有think标签的文本"
        assert self.reward(_wrap(text)) == [0.0]

    def test_empty_think(self):
        text = "<think></think>"
        assert self.reward(_wrap(text)) == [0.0]

    def test_continuous_reasoning(self):
        think = "已知 $x=1$\n由题意 $y=x+1=2$\n故 答案为2"
        text = f"<think>{think}</think>"
        scores = self.reward(_wrap(text))
        assert len(scores) == 1

    def test_single_step(self):
        text = "<think>已知 x = 1</think>"
        scores = self.reward(_wrap(text))
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# LengthReward
# ---------------------------------------------------------------------------

class TestLengthReward:
    def setup_method(self):
        self.reward = LengthReward()

    def test_ideal_length(self):
        text = "短文本" * 10
        assert self.reward(_wrap(text)) == [1.0]

    def test_too_long(self):
        text = "x" * 5000
        assert self.reward(_wrap(text)) == [0.0]

    def test_boundary_low(self):
        text = "x" * 2000
        assert self.reward(_wrap(text)) == [1.0]

    def test_boundary_high(self):
        text = "x" * 4000
        assert self.reward(_wrap(text)) == [0.0]

    def test_linear_decay(self):
        text = "x" * 3000
        scores = self.reward(_wrap(text))
        assert scores[0] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# TopoCompositeReward
# ---------------------------------------------------------------------------

class TestTopoCompositeReward:
    def setup_method(self):
        self.reward = TopoCompositeReward()

    def test_output_length(self):
        completions = [
            [{"content": "text1"}],
            [{"content": "text2"}],
            [{"content": "text3"}],
        ]
        scores = self.reward(completions, solution=None)
        assert len(scores) == 3

    def test_values_in_range(self):
        text = '<think>∵ x=1\n∴ y=2</think><answer>{"学生得分": 5}</answer>'
        solution = json.dumps({"学生得分": 5})
        scores = self.reward(_wrap(text), solution=solution)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_weights_sum_to_one(self):
        assert sum(TopoCompositeReward.WEIGHTS.values()) == pytest.approx(1.0)


class TestTopoRewardDiagnostics:
    def test_empty_trace_preserves_batch_alignment(self):
        reward = TopoReward()
        scores = reward([
            [{"content": "Final answer: \\boxed{1}"}],
            [{"content": ""}],
        ])

        assert scores == [0.0, 0.0]
        assert len(reward.last_dags) == 2
        assert len(reward.last_diagnostics) == 2
        assert [row["empty_trace"] for row in reward.last_diagnostics] == [1.0, 1.0]
        assert all(row["r_topo"] == 0.0 for row in reward.last_diagnostics)

        channels = reward.last_source_scores()
        assert channels == {"q_dir": [0.0, 0.0], "q_acyc": [0.0, 0.0]}

    def test_independent_sources_reuse_last_graph_pass(self):
        reward = TopoReward()
        reward.last_diagnostics = [
            {"direction_consistency": 0.2, "acyclic": 0.8},
            {"direction_consistency": 1.2, "acyclic": -0.1},
        ]

        assert reward.last_source_scores() == {
            "q_dir": [0.2, 1.0],
            "q_acyc": [0.8, 0.0],
        }


class TestTopoChoquetReward:
    def setup_method(self):
        self.reward = TopoChoquetReward()

    def test_coefficients_are_normalized(self):
        total = sum(self.reward.singletons.values()) + sum(self.reward.interactions.values())
        assert total == pytest.approx(1.0)

    def test_boundary_and_monotonicity(self):
        zero = {name: 0.0 for name in self.reward.singletons}
        one = {name: 1.0 for name in self.reward.singletons}
        assert self.reward.aggregate_values(zero) == 0.0
        assert self.reward.aggregate_values(one) == pytest.approx(1.0)

        baseline = dict(zero, outcome=1.0, topology=0.2, continuity=0.8, length=0.8)
        improved = dict(baseline, topology=0.9)
        assert self.reward.aggregate_values(improved) > self.reward.aggregate_values(baseline)

    def test_topology_interaction_is_non_additive(self):
        values = {name: 0.0 for name in self.reward.singletons}
        values.update(outcome=1.0, topology=1.0)
        additive_part = (
            self.reward.singletons["outcome"] + self.reward.singletons["topology"]
        )
        assert self.reward.aggregate_values(values) > additive_part


class TestMatchedAggregationBaselines:
    def setup_method(self):
        self.additive = TopoMatchedAdditiveReward()
        self.equal = TopoEqualAdditiveReward()
        self.multiplicative = TopoMatchedMultiplicativeReward()

    def test_shapley_importance_is_normalized(self):
        assert sum(self.additive.source_importance().values()) == pytest.approx(1.0)

    def test_all_baselines_have_normalized_boundaries(self):
        zero = {name: 0.0 for name in self.additive.singletons}
        one = {name: 1.0 for name in self.additive.singletons}
        for reward in (self.additive, self.equal, self.multiplicative):
            assert reward.aggregate_values(zero) == pytest.approx(0.0)
            assert reward.aggregate_values(one) == pytest.approx(1.0)

    def test_matched_additive_removes_pairwise_synergy(self):
        values = {name: 1.0 for name in self.additive.singletons}
        values["topology"] = 0.0
        choquet = TopoChoquetReward()
        assert self.additive.aggregate_values(values) > choquet.aggregate_values(values)


class TestIndependentTopologyAggregation:
    def test_profile_is_explicit(self, monkeypatch):
        monkeypatch.delenv("TOPO_CHOQUET_CAPACITY_PROFILE", raising=False)
        with pytest.raises(ValueError, match="requires an approved"):
            TopoIndependentChoquetReward()

    def test_balanced_capacity_and_shapley_importance(self):
        reward = TopoIndependentChoquetReward(capacity_profile="balanced")
        assert sum(reward.singletons.values()) + sum(reward.interactions.values()) == pytest.approx(1.0)
        assert reward.source_importance() == pytest.approx({
            "outcome": 0.48,
            "format": 0.05,
            "direction": 0.1625,
            "acyclicity": 0.1625,
            "continuity": 0.145,
        })

    def test_dashboard_capacity_preview_matches_training_reward(self):
        values = {
            "outcome": 1.0,
            "format": 1.0,
            "direction": 0.833,
            "acyclicity": 0.417,
            "continuity": 0.75,
        }
        for name, profile in json_capacity_profiles().items():
            preview = sum(
                weight * values[source]
                for source, weight in profile["singletons"].items()
            )
            preview += sum(
                item["weight"]
                * min(values[item["sources"][0]], values[item["sources"][1]])
                for item in profile["interactions"]
            )
            reward = TopoIndependentChoquetReward(capacity_profile=name)
            assert preview == pytest.approx(reward.aggregate_values(values))

    def test_profile_tradeoff_is_explicit(self):
        balanced = TopoIndependentChoquetReward(capacity_profile="balanced")
        structure_forward = TopoIndependentChoquetReward(
            capacity_profile="structure_forward"
        )
        wrong_but_structured = {
            "outcome": 0.0,
            "format": 1.0,
            "direction": 1.0,
            "acyclicity": 1.0,
            "continuity": 1.0,
        }
        correct_but_structurally_empty = {
            "outcome": 1.0,
            "format": 1.0,
            "direction": 0.0,
            "acyclicity": 0.0,
            "continuity": 0.0,
        }
        assert balanced.aggregate_values(wrong_but_structured) == pytest.approx(0.46)
        assert balanced.aggregate_values(correct_but_structurally_empty) == pytest.approx(0.47)
        assert structure_forward.aggregate_values(wrong_but_structured) == pytest.approx(0.48)
        assert structure_forward.aggregate_values(correct_but_structurally_empty) == pytest.approx(0.45)

    def test_direction_and_acyclicity_are_not_averaged(self):
        reward = TopoIndependentChoquetReward(capacity_profile="balanced")
        base = {name: 0.0 for name in reward.singletons}
        direction_only = dict(base, direction=1.0)
        acyclicity_only = dict(base, acyclicity=1.0)
        both = dict(base, direction=1.0, acyclicity=1.0)
        assert reward.aggregate_values(direction_only) == pytest.approx(0.05)
        assert reward.aggregate_values(acyclicity_only) == pytest.approx(0.05)
        assert reward.aggregate_values(both) == pytest.approx(0.20)

    def test_direction_acyclicity_interaction_ablation_is_matched(
        self, monkeypatch
    ):
        full = TopoIndependentChoquetReward(capacity_profile="balanced")
        monkeypatch.setenv("TOPO_DISABLE_DIRECTION_ACYCLICITY_INTERACTION", "1")
        ablated = TopoIndependentChoquetReward(capacity_profile="balanced")
        assert ("direction", "acyclicity") not in ablated.interactions
        assert sum(ablated.singletons.values()) + sum(ablated.interactions.values()) == pytest.approx(1.0)
        assert ablated.source_importance() == pytest.approx(full.source_importance())
        direction_only = {name: 0.0 for name in ablated.singletons}
        direction_only["direction"] = 1.0
        both = dict(direction_only, acyclicity=1.0)
        assert ablated.aggregate_values(direction_only) == pytest.approx(0.10)
        assert ablated.aggregate_values(both) == pytest.approx(0.20)

    def test_additive_control_matches_shapley_weights(self):
        reward = TopoIndependentMatchedAdditiveReward(capacity_profile="balanced")
        values = {name: 1.0 for name in reward.singletons}
        assert reward.aggregate_values(values) == pytest.approx(1.0)

    def test_call_reuses_one_graph_pass_and_preserves_source_alignment(self):
        class StaticReward:
            def __init__(self, values):
                self.values = values

            def __call__(self, completions, **kwargs):
                assert len(completions) == len(self.values)
                return list(self.values)

        class TopologySources(StaticReward):
            calls = 0

            def __call__(self, completions, **kwargs):
                self.calls += 1
                return super().__call__(completions, **kwargs)

            def last_source_scores(self):
                return {"q_dir": [0.2, 0.9], "q_acyc": [0.8, 0.1]}

        reward = TopoIndependentChoquetReward(capacity_profile="balanced")
        reward._outcome = StaticReward([1.0, 0.0])
        reward._format = StaticReward([0.5, 0.5])
        reward._continuity = StaticReward([0.4, 0.6])
        reward._topo = TopologySources([0.0, 0.0])
        completions = [[{"content": "a"}], [{"content": "b"}]]

        observed = reward(completions, solution=["a", "b"])
        expected = [
            reward.aggregate_values({
                "outcome": 1.0,
                "format": 0.5,
                "direction": 0.2,
                "acyclicity": 0.8,
                "continuity": 0.4,
            }),
            reward.aggregate_values({
                "outcome": 0.0,
                "format": 0.5,
                "direction": 0.9,
                "acyclicity": 0.1,
                "continuity": 0.6,
            }),
        ]
        assert observed == pytest.approx(expected, abs=1e-6)
        assert reward._topo.calls == 1


class TestMatchedControls:
    class StaticReward:
        def __init__(self, values):
            self.values = values

        def __call__(self, completions, **kwargs):
            assert len(completions) == len(self.values)
            return list(self.values)

    class TrainerState:
        global_step = 20

    def test_laser_d_uses_group_difficulty_and_token_lengths(self, monkeypatch):
        monkeypatch.setenv("TOPO_LASER_MONITOR_SIZE", "1")
        monkeypatch.setenv("TOPO_LASER_MONITOR_MIN_GROUPS", "1")
        monkeypatch.setenv("TOPO_LASER_TARGET_LOWER", "2")
        monkeypatch.setenv("TOPO_LASER_TARGET_UPPER", "8")
        monkeypatch.setenv("TOPO_LASER_TARGET_INTERVAL", "2")
        reward = MatchedLaserDReward()
        reward._outcome = self.StaticReward([1.0, 1.0, 1.0, 0.0])
        completions = [[{"content": str(index)}] for index in range(4)]
        scores = reward(
            completions,
            solution=["x"] * 4,
            response_token_ids=[[1, 2], [1, 2, 3, 4], list(range(8)), list(range(8))],
            trainer_state=self.TrainerState(),
        )
        assert reward.targets["high"] == 4
        assert scores == [1.5, 1.5, 1.0, 0.0]

    def test_laser_d_rejects_length_proxies(self):
        reward = MatchedLaserDReward()
        reward._outcome = self.StaticReward([1.0] * 4)
        completions = [[{"content": "short text"}]] * 4
        with pytest.raises(RuntimeError, match="response_token_ids"):
            reward(completions, solution=["x"] * 4)

    def test_hero_preserves_verifier_strata_with_dense_topology(self):
        reward = TopoIndependentHeroReward(capacity_profile="balanced")
        batches = {
            "outcome": [0.0, 0.0, 1.0, 1.0],
            "format": [0.0, 1.0, 0.0, 1.0],
            "direction": [0.0, 1.0, 0.0, 1.0],
            "acyclicity": [0.0, 1.0, 0.0, 1.0],
            "continuity": [0.0, 1.0, 0.0, 1.0],
        }
        reward._source_batches = lambda *args, **kwargs: batches
        completions = [[{"content": str(index)}] for index in range(4)]
        scores = reward(completions, solution=["x"] * 4)
        assert scores[0] < scores[1] < scores[2] < scores[3]
        assert scores[1] < scores[2]

    def test_hero_rejects_partial_rollout_groups(self):
        reward = TopoIndependentHeroReward(capacity_profile="balanced")
        with pytest.raises(RuntimeError, match="complete rollout groups"):
            reward([[{"content": "x"}]] * 3, solution=["x"] * 3)


class TestOutcomeFormatLengthReward:
    def test_is_non_topological_and_rewards_correct_short_answers(self):
        reward = OutcomeFormatLengthReward()
        assert not hasattr(reward, "_topo")
        scores = reward(
            [_wrap(r"Work. \boxed{2}")[0], _wrap(r"Work. \boxed{3}")[0]],
            solution=["2", "2"],
        )
        assert scores[0] > scores[1]
