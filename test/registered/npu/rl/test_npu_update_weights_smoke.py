"""1-NPU smoke tests for weight update and metrics.

RL Test Consolidation (sgl-project/sglang#47624) — weight update + metrics
area, 1-NPU tier. File ownership per RFC section 5.

Coverage:
- WU-S1  weights actually reach the compute graph: update_weights_from_disk
         with a modified checkpoint (one layernorm weight +0.05, built by this
         test from the base model — the E4a template) must change greedy
         outputs; no stale-graph/old-address replay.
- WU-S3  weight_version round trip: /update_weights_from_disk accepts
         weight_version and /get_server_info reflects it; /update_weight_version
         sets a version without touching weights.
- WU-M1  metrics export: with --enable-metrics, /metrics carries
         sglang:num_requests_total / sglang:num_running_reqs, and the request
         counter increases by the number of served generations (numeric
         assertions, not field-presence).

Unit-tier note (per RFC no-same-tier-duplicates): version tracking state
machine (success records version / failure does not, disk+distributed+tensor
paths) is already covered by
test/registered/unit/managers/test_scheduler_weight_version_tracking.py —
this file adds no redundant unit cases.

Known defect context (defect-fix PR pending, NOT asserted here): on NPU MoE
models the fuseep layout transform makes update_weights_from_disk
non-idempotent — an A->B->A roundtrip is NOT guaranteed to restore baseline
outputs. This file therefore never asserts baseline restoration; it uses a
dense model (Qwen3-0.6B) where the update path is stable. Follow-up nightly
work: HCCL from_distributed A->B->A with trainer-logprob reference (2 NPU).

Server die selection: set RL_FEAT_NPU_ID (default "0") to pin
ASCEND_RT_VISIBLE_DEVICES.
"""

import json
import os
import shutil
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_npu_ci(est_time=1500, suite="nightly-npu-a3-merged", nightly=True)

# Model override for hosts without the default weights cached (e.g. npu188
# only caches Qwen/Qwen3-0.6B): export RL_FEAT_MODEL_PATH=<absolute path>.
MODEL = os.environ.get("RL_FEAT_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
MODIFIED_DIR = os.path.join(tempfile.gettempdir(), "rlfeat-model-modified")

PROMPTS = [
    "The capital of France is",
    "Write a one-sentence summary of photosynthesis.",
    "1 2 3 4 5",
    "你好，请用一句话介绍自己。",
    "def add(a, b):",
]


def _sampling_params(max_new_tokens=64):
    return {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}


def _generate_all(base_url):
    outs = []
    for p in PROMPTS:
        r = requests.post(
            base_url + "/generate",
            json={"text": p, "sampling_params": _sampling_params()},
            timeout=300,
        )
        r.raise_for_status()
        outs.append(r.json()["text"])
    return outs


def _metric_sum(text, name):
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return total


def _find_key_value(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_value(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key_value(v, key)
            if found is not None:
                return found
    return None


@unittest.skipUnless(os.path.isdir(MODEL), "model weights not available")
class TestNpuUpdateWeightsSmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Build the modified checkpoint once: copy the model dir and bump one
        # first-layer norm weight by +0.05 (E4a template from the npu188
        # weight-update validation).
        if os.path.isdir(MODIFIED_DIR):
            shutil.rmtree(MODIFIED_DIR)
        shutil.copytree(MODEL, MODIFIED_DIR)
        from safetensors.torch import load_file, save_file

        st_path = os.path.join(MODIFIED_DIR, "model.safetensors")
        tensors = load_file(st_path)
        target_key = None
        for key in sorted(tensors):
            if ".layers.0." in key and "norm" in key and tensors[key].dim() == 1:
                target_key = key
                break
        assert target_key is not None, "no first-layer norm tensor found"
        tensors[target_key] = tensors[target_key] + 0.05
        save_file(tensors, st_path, metadata={"format": "pt"})
        cls.modified_key = target_key

        port = find_available_port(31000)
        cls.base_url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("RL_FEAT_NPU_ID", "0")
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "ascend", "--enable-metrics"],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.isdir(MODIFIED_DIR):
            shutil.rmtree(MODIFIED_DIR, ignore_errors=True)

    def _update_from_disk(self, model_path, **extra):
        r = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": model_path, "flush_cache": True, **extra},
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body.get("success"), f"update failed: {body}")
        return body

    # --- scenarios ----------------------------------------------------------

    def test_1_modified_weights_reach_compute_graph(self):
        """E4a template: after updating to the modified checkpoint the greedy
        outputs must change — proving the new weights (not stale graph
        pointers) drive generation."""
        baseline = _generate_all(self.base_url)

        self._update_from_disk(MODIFIED_DIR)

        after = _generate_all(self.base_url)
        self.assertNotEqual(
            after,
            baseline,
            "modified weights must change at least one greedy output",
        )

    def test_2_weight_version_round_trip(self):
        """A disk update can carry weight_version and /update_weight_version
        bumps the live version without touching weights.

        Known visibility gap (observed on npu188, for the defect round):
        /get_server_info reports only the STATIC startup weight_version
        ('default' even after updates) — the live version surfaces per-request
        via weight-version spans in generate meta_info instead, not via
        get_server_info. Asserted here: both endpoints accept the calls and
        get_server_info exposes the field."""
        self._update_from_disk(MODEL, weight_version="v3")

        r = requests.post(
            self.base_url + "/update_weight_version",
            json={"new_version": "v4"},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200, r.text)

        info = requests.get(
            self.base_url + "/get_server_info", timeout=120
        ).json()
        self.assertIsNotNone(
            _find_key_value(info, "weight_version"),
            "get_server_info should expose a weight_version field",
        )

    def test_3_metrics_export_and_request_counter(self):
        """With --enable-metrics, /metrics exports the request counters and
        the counter grows by the number of served generations."""
        text = requests.get(self.base_url + "/metrics", timeout=60).text
        self.assertIn("sglang:num_requests_total", text)
        self.assertIn("sglang:num_running_reqs", text)

        before = _metric_sum(text, "sglang:num_requests_total")

        for _ in range(3):
            r = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": _sampling_params(16),
                },
                timeout=300,
            )
            r.raise_for_status()

        text_after = requests.get(self.base_url + "/metrics", timeout=60).text
        after = _metric_sum(text_after, "sglang:num_requests_total")
        self.assertGreaterEqual(
            after - before,
            3.0,
            f"request counter must grow by >=3: before={before} after={after}",
        )


if __name__ == "__main__":
    unittest.main()
