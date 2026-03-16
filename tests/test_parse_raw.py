import json

import pytest

from src.data.parse_raw import parse_single_record


def _make_payload(scp_value, score_info=None, sub_correct_infos=None,
                  topic_id="t1", topic_type="math"):
    """Build a valid status-2 payload dict with double-encoded content."""
    if score_info is None:
        score_info = {"score": 5.0, "procedureScore": 3.0}
    if sub_correct_infos is None:
        sub_correct_infos = []
    content_inner = {
        "data": {
            "topicId": topic_id,
            "topicType": topic_type,
            "stepCorrectInfo": {
                "extendInfos": [
                    {"key": "StepCorrProcess", "value": scp_value},
                ],
                "correctInfo": {"scoreInfo": score_info},
                "subCorrectInfos": sub_correct_infos,
            },
        },
    }
    return {
        "header": {"status": 2},
        "payload": {
            "output": {
                "content": json.dumps(content_inner, ensure_ascii=False),
            },
        },
    }


class TestParseSingleRecord:
    def test_valid_nested_json(self):
        scp = {
            "llm_stem": "求x的值",
            "llm_stdanswer": "x=5",
            "llm_user": "x=4",
            "llm_result": "部分正确",
        }
        raw = _make_payload(scp, score_info={"score": 3.0, "procedureScore": 2.0})
        result = parse_single_record(raw)

        assert result is not None
        assert result["stem"] == "求x的值"
        assert result["standard_answer"] == "x=5"
        assert result["student_answer"] == "x=4"
        assert result["llm_result"] == "部分正确"
        assert result["score"] == 3.0
        assert result["procedure_score"] == 2.0
        assert result["topic_id"] == "t1"
        assert result["topic_type"] == "math"

    def test_missing_payload(self):
        raw = {"header": {"status": 0}}
        assert parse_single_record(raw) is None

    def test_malformed_content(self):
        raw = {
            "header": {"status": 2},
            "payload": {"output": {"content": "{{{INVALID JSON}}}"}},
        }
        assert parse_single_record(raw) is None

    def test_empty_content_string(self):
        raw = {
            "header": {"status": 2},
            "payload": {"output": {"content": ""}},
        }
        assert parse_single_record(raw) is None

    def test_missing_step_corr_process(self):
        content_inner = {
            "data": {
                "stepCorrectInfo": {
                    "extendInfos": [
                        {"key": "OtherKey", "value": {}},
                    ],
                },
            },
        }
        raw = {
            "header": {"status": 2},
            "payload": {
                "output": {
                    "content": json.dumps(content_inner, ensure_ascii=False),
                },
            },
        }
        assert parse_single_record(raw) is None
