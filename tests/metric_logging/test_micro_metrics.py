# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import torch
from transformers import DynamicCache

from kvpress import KnormPress
from kvpress.metric_logging import (
    ATTENTION_OUTPUT_ERROR,
    ATTENTION_SPARSITY,
    MicroMetricLogger,
    bias_sparsity,
    cache_sparsity,
    masked_key_sparsity,
    relative_attention_output_error,
)
from tests.fixtures import unit_test_model  # noqa: F401


def setup_function():
    MicroMetricLogger.reset_for_testing()


def test_bias_sparsity():
    bias = torch.tensor([[[0.0, float("-inf"), 1.0, float("-inf")]]])
    assert abs(bias_sparsity(bias) - 0.5) < 1e-6


def test_cache_sparsity():
    assert abs(cache_sparsity(100, 25) - 0.75) < 1e-6
    assert cache_sparsity(0, 0) == 0.0


def test_masked_key_sparsity():
    idx = (torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long), torch.arange(10))
    assert abs(masked_key_sparsity(idx, batch_size=1, num_kv_heads=2, seq_len=10) - 0.5) < 1e-6


def test_relative_attention_output_error_zero_when_equal():
    x = torch.randn(1, 2, 3, 4)
    assert relative_attention_output_error(x, x) == 0.0


def test_relative_denominator_error():
    from kvpress.metric_logging import relative_denominator_error

    assert relative_denominator_error(1.0, 1.0) == 0.0
    assert abs(relative_denominator_error(1.2, 1.0) - 0.2) < 1e-9
    assert relative_denominator_error(0.0, 0.0) == 0.0
    assert relative_denominator_error(1.0, 0.0) == float("inf")


def test_micro_metric_logger_writes_jsonl(tmp_path):
    logger = MicroMetricLogger()
    logger.configure_logging(
        log_path=str(tmp_path),
        enabled_metrics=[ATTENTION_SPARSITY, ATTENTION_OUTPUT_ERROR],
    )
    logger.log(ATTENTION_SPARSITY, 0.75, metadata={"layer_idx": 0, "source": "cache"})
    logger.log(ATTENTION_OUTPUT_ERROR, 0.01, metadata={"layer_idx": 0, "source": "attention_bias"})
    logger.flush()

    path = tmp_path / "micro_metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    assert {e["metric"] for e in events} == {ATTENTION_SPARSITY, ATTENTION_OUTPUT_ERROR}
    assert events[0]["value"] == 0.75


def test_knorm_logs_cache_sparsity(unit_test_model, tmp_path):  # noqa: F811
    MicroMetricLogger.reset_for_testing()
    MicroMetricLogger().configure_logging(log_path=str(tmp_path), enabled_metrics=[ATTENTION_SPARSITY])

    press = KnormPress(compression_ratio=0.5)
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache())
    MicroMetricLogger().flush()

    events = [json.loads(line) for line in (tmp_path / "micro_metrics.jsonl").read_text().splitlines()]
    assert len(events) == unit_test_model.config.num_hidden_layers
    assert all(e["metric"] == ATTENTION_SPARSITY and e["metadata"]["source"] == "cache" for e in events)
    assert all(abs(e["value"] - 0.5) < 1e-5 for e in events)
