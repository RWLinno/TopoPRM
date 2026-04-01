from src.prm.model import VerifiableProcessRewardModel


def test_prm_model_score_trace_returns_fields() -> None:
    model = VerifiableProcessRewardModel()
    result = model.score_trace("已知 x=1\n由题意 y=x+1=2\n故答案为2")
    assert 0.0 <= result.topology_reward <= 1.0
    assert 0.0 <= result.continuity_reward <= 1.0
    assert result.dag_nodes >= 0
    assert result.dag_edges >= 0
