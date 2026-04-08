# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile
import unittest

import torch

# Importing _torchwave registers the torchwave delegate handler with the
# nativert KernelFactory, so PyModelRunner can dispatch to it.
from velox.experimental.torchwave._torchwave import DelegateExecutor
from velox.experimental.torchwave.lower import (
    lower_to_torchwave,
    package_nativert_with_torchwave_delegate,
)
from velox.experimental.torchwave.tests.element_test import ElementTestPreproc


class DelegateExecutorTest(unittest.TestCase):
    """Tests the torchwave delegate executor pattern using ElementTestPreproc.

    Follows the AOTI delegate pattern:
    1. Define a module
    2. Export via torch.export
    3. Save as .pt2 archive
    4. Load via C++ DelegateExecutor and verify outputs match eager execution
    """

    def setUp(self) -> None:
        self.module = ElementTestPreproc()
        self.a = torch.arange(1, 10001, dtype=torch.long)
        self.b = torch.arange(1, 10001, dtype=torch.long)
        self.c = torch.arange(1, 10001, dtype=torch.long)
        self.d = torch.arange(1, 100001, dtype=torch.long)
        self.e = torch.arange(1, 100001, dtype=torch.long)
        self.f = torch.arange(1, 100001, dtype=torch.long)
        self.inputs = (self.a, self.b, self.c, self.d, self.e, self.f)

    def test_eager_execution(self) -> None:
        """Verifies that eager execution produces correct results."""
        outputs = self.module(*self.inputs)
        self.assertEqual(len(outputs), 5)
        # o1 = a + b - c
        self.assertTrue(torch.equal(outputs[0], self.a + self.b - self.c))
        # o3 = c - a
        self.assertTrue(torch.equal(outputs[2], self.c - self.a))

    def test_export_and_reload(self) -> None:
        """Exports the module to .pt2, reloads it, and verifies outputs match."""
        eager_outputs = self.module(*self.inputs)

        with torch.no_grad():
            exported = torch.export.export(
                self.module, self.inputs, strict=False
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "element_test.pt2")
            torch.export.save(exported, pt2_path)

            self.assertTrue(os.path.exists(pt2_path))
            self.assertGreater(os.path.getsize(pt2_path), 0)

            loaded = torch.export.load(pt2_path)
            loaded_outputs = loaded.module()(*self.inputs)

        for i, (eager, loaded) in enumerate(
            zip(eager_outputs, loaded_outputs)
        ):
            self.assertTrue(
                torch.equal(eager, loaded),
                f"Output {i} differs between eager and loaded",
            )

    def test_delegate_executor(self) -> None:
        """Exports to .pt2 and runs through the C++ DelegateExecutor."""
        eager_outputs = self.module(*self.inputs)

        with torch.no_grad():
            exported = torch.export.export(
                self.module, self.inputs, strict=False
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "element_test.pt2")
            torch.export.save(exported, pt2_path)

            executor = DelegateExecutor(pt2_path)
            wave_outputs = executor.run(list(self.inputs))

        self.assertEqual(len(wave_outputs), len(eager_outputs))
        for i, (eager, wave) in enumerate(zip(eager_outputs, wave_outputs)):
            self.assertTrue(
                torch.equal(eager.cpu(), wave.cpu()),
                f"Output {i} differs between eager and torchwave",
            )

    def test_nativert_delegate(self) -> None:
        """Lowers to torchwave delegate, packages as .pt2, runs via nativert
        ModelRunner to exercise the full delegate registration path."""
        from torch._C._nativert import PyModelRunner

        MODEL_NAME = "model"
        eager_outputs = self.module(*self.inputs)

        with torch.no_grad():
            original_ep = torch.export.export(
                self.module, self.inputs, strict=False
            )

        delegate_ep = lower_to_torchwave(original_ep, MODEL_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "nativert_test.pt2")
            package_nativert_with_torchwave_delegate(
                pt2_path,
                MODEL_NAME,
                original_ep,
                delegate_ep,
            )

            model_runner = PyModelRunner(
                pt2_path, f"{MODEL_NAME}-torchwave"
            )
            results = model_runner.run(*self.inputs)
            flat_results = torch.utils._pytree.tree_leaves(results)

        self.assertEqual(len(flat_results), len(eager_outputs))
        for i, (eager, result) in enumerate(
            zip(eager_outputs, flat_results)
        ):
            self.assertTrue(
                torch.equal(eager.cpu(), result.cpu()),
                f"Output {i} differs between eager and nativert+torchwave",
            )
