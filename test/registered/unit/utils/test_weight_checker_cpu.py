"""WeightChecker checksum on a non-CUDA device (CPU stands in for TPU)."""

import hashlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils.weight_checker import ChecksumInfo, WeightChecker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except Exception:  # pragma: no cover - non-NPU environments
    _NPU_AVAILABLE = False

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChecksumWithoutCuda(CustomTestCase):
    def test_checksum_does_not_require_cuda(self):
        """The checksum action must work on hosts without CUDA, e.g. TPU engines
        checked by an RL trainer after every weight update."""
        model = nn.Linear(4, 4)
        enter_scope(self, published_topology())
        checker = WeightChecker(get_model=lambda: model)
        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("no CUDA device")
        ):
            out = checker.handle("checksum", role="target")
        info = ChecksumInfo.model_validate(out)
        self.assertEqual(set(info.checksums), {"weight", "bias"})


class TestRawChecksumWithoutCuda(CustomTestCase):
    def test_raw_checksum_includes_loaded_buffers(self) -> None:
        """Hash persistent buffers while excluding derived rotary caches."""
        for names in (None, ["layer_scalar"]):
            with self.subTest(names=names):
                model = nn.Linear(2, 2)
                scalar = torch.tensor([1.5])
                model.register_buffer("layer_scalar", scalar)
                model.register_buffer("cos_sin_cache", torch.ones(2), persistent=False)
                with published_topology():
                    checker = WeightChecker(get_model=lambda: model)
                    result = checker.handle(
                        action="raw_checksum", names=names, role="target"
                    )

                expected_names = (
                    {"weight", "bias", "layer_scalar"} if names is None else set(names)
                )
                self.assertEqual(set(result["checksums"]), expected_names)
                self.assertEqual(
                    result["checksums"]["layer_scalar"],
                    hashlib.sha256(
                        scalar.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                )

    def test_missing_names_report_runner_and_all_missing_tensors(self) -> None:
        """Report unavailable names explicitly instead of leaking a bare KeyError."""
        with published_topology():
            checker = WeightChecker(get_model=lambda: nn.Linear(2, 2))
            with self.assertRaisesRegex(
                ValueError, "draft runner.*missing_a.*missing_b"
            ):
                checker.handle(
                    action="raw_checksum",
                    names=["weight", "missing_b", "missing_a"],
                    role="draft",
                )


class TestWeightCheckerLifecycle(CustomTestCase):
    """Handle-level snapshot/reset/compare lifecycle.

    Consolidation additions (sgl-project/sglang#47624, weight-checker tier):
    the snapshot is consumed by a successful compare (one-shot baseline),
    reset_tensors corrupts exactly the non-skipped tensors, and unknown
    actions fail loudly.
    """

    @staticmethod
    def _checker() -> WeightChecker:
        # One model instance for the whole lifecycle: get_model is called per
        # action, and a fresh nn.Linear would hand every action different
        # random weights. Weights are inference-style (requires_grad=False),
        # matching real models where reset_tensors does in-place copies.
        model = nn.Linear(4, 4)
        for p in model.parameters():
            p.requires_grad_(False)
        return WeightChecker(get_model=lambda: model)

    def setUp(self):
        # The compare path synchronizes the device per tensor; on CPU-only
        # hosts torch.cuda.synchronize raises. Unit tier runs on CPU tensors,
        # so stub the sync out.
        patcher = patch("torch.cuda.synchronize", lambda *a, **k: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        if not torch.cuda.is_available():
            # The comparator hardcodes .cuda() on comparison chunks. On an
            # NPU-only host redirect it to .npu() so the lifecycle keeps real
            # device semantics; with neither backend, skip.
            if _NPU_AVAILABLE:
                npu_patcher = patch.object(
                    torch.Tensor, "cuda", lambda self, *a, **k: self.npu()
                )
                npu_patcher.start()
                self.addCleanup(npu_patcher.stop)
            else:
                self.skipTest(
                    "compare path moves chunks to .cuda(); needs a CUDA or NPU host"
                )

    def test_handle_unknown_action_raises(self):
        with published_topology():
            checker = self._checker()
            with self.assertRaisesRegex(Exception, "[Uu]nsupported"):
                checker.handle(action="bogus", role="target")

    def test_compare_without_snapshot_raises(self):
        """compare must refuse to run without a snapshot instead of passing
        vacuously."""
        with published_topology():
            checker = self._checker()
            with self.assertRaises(AssertionError):
                checker.handle(action="compare", role="target")

    def test_snapshot_then_compare_passes_and_is_one_shot(self):
        """A matching compare consumes the snapshot: the second compare must
        fail loudly rather than silently re-validate stale state."""
        with published_topology():
            checker = self._checker()
            checker.handle(action="snapshot", role="target")
            checker.handle(action="compare", role="target")
            with self.assertRaises(AssertionError):
                checker.handle(action="compare", role="target")

    def test_reset_tensors_detected_by_compare(self):
        """reset_tensors randomizes the weights: compare against the
        pre-reset snapshot must fail with the equality-failure report."""
        with published_topology():
            checker = self._checker()
            checker.handle(action="snapshot", role="target")
            checker.handle(action="reset_tensors", role="target")
            with self.assertRaisesRegex(Exception, "check tensor equality failed"):
                checker.handle(action="compare", role="target")

    def test_reset_tensors_respects_skip_list(self):
        """reset skips exactly what the skip list names (substring match):
        with bias preserved, a compare that skips the randomized weight
        passes."""
        with published_topology():
            checker = self._checker()
            checker.handle(action="snapshot", role="target")
            checker.handle(
                action="reset_tensors", skip_tensor_list=["bias"], role="target"
            )
            checker.handle(
                action="compare", skip_tensor_list=["weight"], role="target"
            )

    def test_checksum_changes_after_reset(self):
        """The overall checksum is content-sensitive: resetting tensors must
        change it (evidence the checksum really hashes weights)."""
        with published_topology():
            checker = self._checker()
            before = checker.handle(action="checksum", role="target")
            checker.handle(action="reset_tensors", role="target")
            after = checker.handle(action="checksum", role="target")
        self.assertNotEqual(
            before["per_gpu_checksum"], after["per_gpu_checksum"]
        )


if __name__ == "__main__":
    unittest.main()
