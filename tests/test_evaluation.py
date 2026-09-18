"""Evaluation CLI guard and JSON artifact regressions (no external services)."""
import json

import pytest

from scripts.evaluate_models import run_evaluation


@pytest.mark.parametrize("ratio,k", [(0, 5), (1, 5), (-0.1, 5), (0.2, 0)])
def test_invalid_evaluation_parameters(ratio, k):
    with pytest.raises(ValueError, match="Require"):
        run_evaluation(test_ratio=ratio, top_k=k)


def test_evaluation_json_report(tmp_path):
    destination = tmp_path / "evaluation.json"
    report = run_evaluation(output=destination, dataset_label="test database")
    assert json.loads(destination.read_text()) == report
    assert len(report["results"]) == 5
    dataset = report["dataset"]
    assert dataset["train_events"] + dataset["test_events"] == dataset["interactions"]
    assert dataset["evaluated_users"] > 0
    assert len(dataset["sha256"]) == 64
    for metrics in report["results"].values():
        assert metrics["EvaluatedUsers"] == dataset["evaluated_users"]
        assert 0 <= metrics["NDCG@5"] <= 1
