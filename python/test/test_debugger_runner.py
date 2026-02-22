# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import pyarrow
from pyvelox.arrow import to_velox
from pyvelox.plan_builder import PlanBuilder
from pyvelox.runner import (
    LocalDebuggerRunner,
)


class TestPyVeloxDebuggerRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._batch_size = 20
        self._num_batches = 10
        self._num_projections = 10

        array = pyarrow.array(list(range(self._batch_size)))
        # type: ignore
        batch = pyarrow.record_batch([array], names=["c0"])

        vector = to_velox(batch)
        vectors = [vector] * self._num_batches

        plan_builder = PlanBuilder().values(vectors)
        self._node_ids = []

        for _ in range(self._num_projections):
            plan_builder.project(["c0 * 10 as c0"])
            self._node_ids.append(plan_builder.get_plan_node().id())

        self._plan_node = plan_builder.get_plan_node()

    def _count_next(self, runner):
        total_size = 0
        it = runner.execute()

        while True:
            try:
                vector = it.next()
                total_size += vector.size()
            except StopIteration:
                break
        return total_size

    def _count_step(self, runner):
        total_size = 0
        it = runner.execute()

        while True:
            try:
                vector = it.step()
                total_size += vector.size()
            except StopIteration:
                break
        return total_size

    def test_undrained(self):
        """Ensure tasks won't hang or fail if they are not completely drained."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.execute()

    def test_next_no_breakpoints(self):
        runner = LocalDebuggerRunner(self._plan_node)
        self.assertEqual(self._count_next(runner), self._batch_size * self._num_batches)

    def test_step_no_breakpoints(self):
        runner = LocalDebuggerRunner(self._plan_node)
        self.assertEqual(self._count_step(runner), self._batch_size * self._num_batches)

    def test_next_with_breakpoints(self):
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[5])
        self.assertEqual(self._count_next(runner), self._batch_size * self._num_batches)

    def test_step_with_breakpoints(self):
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[5])
        runner.set_breakpoint(self._node_ids[8])
        self.assertEqual(
            self._count_step(runner), self._batch_size * self._num_batches * 4
        )

    def test_step_all_breakpoints(self):
        runner = LocalDebuggerRunner(self._plan_node)
        [runner.set_breakpoint(i) for i in self._node_ids]
        self.assertEqual(
            self._count_step(runner),
            self._batch_size * self._num_batches * (self._num_projections + 1),
        )

    def test_breakpoint_with_aggregate(self):
        """Set a breakpoint before aggregation to see pre-aggregated data."""
        batch_size = 100

        # Create data with some duplicates for aggregation.
        values = [i % 10 for i in range(batch_size)]
        array = pyarrow.array(values)
        batch = pyarrow.record_batch([array], names=["c0"])
        vector = to_velox(batch)

        # Produce the input vector 3 times.
        plan_builder = PlanBuilder().values([vector, vector, vector])

        plan_builder.aggregate(grouping_keys=["c0"], aggregations=["count(1) as cnt"])
        agg_node_id = plan_builder.get_plan_node().id()

        runner = LocalDebuggerRunner(plan_builder.get_plan_node())
        runner.set_breakpoint(agg_node_id)

        it = runner.execute()

        # Get aggregate first input, 100 records.
        it.step()
        self.assertEqual(it.current().size(), 100)

        # Get aggregate second input, 100 records.
        it.step()
        self.assertEqual(it.current().size(), 100)

        # Ignore next aggregate input and move to the task output, 10 records.
        it.next()
        self.assertEqual(it.current().size(), 10)

        # Should be done now.
        with self.assertRaises(StopIteration):
            it.next()

    def test_iterator_at(self):
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[8])

        it = runner.execute()
        self.assertEqual(it.at(), "")

        it.step()
        self.assertEqual(it.at(), self._node_ids[3])

        it.step()
        self.assertEqual(it.at(), self._node_ids[8])

        it.step()
        self.assertEqual(it.at(), "")

        it.step()
        self.assertEqual(it.at(), self._node_ids[3])

        it.next()
        self.assertEqual(it.at(), "")

        it.next()
        self.assertEqual(it.at(), "")

        it.step()
        self.assertEqual(it.at(), self._node_ids[3])

    def test_hook_always_stop(self):
        """Hook that always returns True should behave like set_breakpoint."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_hook(self._node_ids[3], lambda v: True)
        runner.set_hook(self._node_ids[5], lambda v: True)
        runner.set_hook(self._node_ids[8], lambda v: True)
        self.assertEqual(
            self._count_step(runner), self._batch_size * self._num_batches * 4
        )

    def test_hook_never_stop(self):
        """Hook that always returns False should skip the breakpoint."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_hook(self._node_ids[3], lambda v: False)
        runner.set_hook(self._node_ids[5], lambda v: False)
        runner.set_hook(self._node_ids[8], lambda v: False)
        # Since all hooks return False, step() should behave like next()
        self.assertEqual(self._count_step(runner), self._batch_size * self._num_batches)

    def test_hook_conditional(self):
        """Hook that conditionally stops based on vector content."""
        runner = LocalDebuggerRunner(self._plan_node)

        # Track how many times the hook is called.
        call_count = 0

        def conditional_hook(vector):
            nonlocal call_count
            call_count += 1
            # Stop only on odd calls.
            return call_count % 2 == 1

        runner.set_hook(self._node_ids[5], conditional_hook)

        # The hook is called for each batch (10 batches).
        # It stops on odd calls (1, 3, 5, 7, 9) = 5 stops.
        # Plus 10 task outputs = 15 total vectors.
        self.assertEqual(self._count_step(runner), self._batch_size * 15)
        self.assertEqual(call_count, self._num_batches)

    def test_hook_mixed_with_breakpoint(self):
        """Mix set_breakpoint and set_hook."""
        runner = LocalDebuggerRunner(self._plan_node)

        # set_breakpoint always stops.
        runner.set_breakpoint(self._node_ids[3])

        # set_hook that never stops.
        runner.set_hook(self._node_ids[5], lambda v: False)

        # set_hook that always stops.
        runner.set_hook(self._node_ids[8], lambda v: True)

        # node_ids[3] stops (10 batches), node_ids[8] stops (10 batches),
        # plus 10 task outputs = 30 total vectors.
        self.assertEqual(
            self._count_step(runner), self._batch_size * self._num_batches * 3
        )

    def test_hook_inspects_vector(self):
        """Verify the hook receives a valid vector."""
        runner = LocalDebuggerRunner(self._plan_node)

        received_sizes = []

        def inspect_hook(vector):
            received_sizes.append(vector.size())
            return True

        runner.set_hook(self._node_ids[5], inspect_hook)

        it = runner.execute()

        # Consume all output.
        while True:
            try:
                it.step()
            except StopIteration:
                break

        # Hook should have been called for each batch.
        self.assertEqual(len(received_sizes), self._num_batches)
        # Each vector should have batch_size rows.
        self.assertTrue(all(s == self._batch_size for s in received_sizes))

    def test_step_with_plan_id(self):
        """step(plan_id) should only stop at the matching breakpoint."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[5])
        runner.set_breakpoint(self._node_ids[8])

        it = runner.execute()

        # Step targeting node_ids[5] should skip node_ids[3] and stop at
        # node_ids[5].
        it.step(self._node_ids[5])
        self.assertEqual(it.at(), self._node_ids[5])

        # Step targeting node_ids[8] should skip node_ids[5] (remaining) and
        # stop at node_ids[8].
        it.step(self._node_ids[8])
        self.assertEqual(it.at(), self._node_ids[8])

        # Step with no filter (default) stops at the next breakpoint or task
        # output.
        it.step()
        self.assertEqual(it.at(), "")

    def test_step_with_plan_id_counts(self):
        """step(plan_id) should only produce vectors from the matching
        breakpoint plus task outputs."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[5])

        it = runner.execute()
        total_size = 0

        # Only step to node_ids[5], skipping node_ids[3].
        while True:
            try:
                vector = it.step(self._node_ids[5])
                total_size += vector.size()
            except StopIteration:
                break

        # We expect node_ids[5] breakpoint hits (num_batches) plus task outputs
        # (num_batches) = 2 * num_batches vectors, each of batch_size rows.
        self.assertEqual(total_size, self._batch_size * self._num_batches * 2)

    def test_step_with_plan_id_default_behavior(self):
        """step() with no argument should preserve original behavior."""
        runner = LocalDebuggerRunner(self._plan_node)
        runner.set_breakpoint(self._node_ids[3])
        runner.set_breakpoint(self._node_ids[5])

        # step() with no plan_id should hit both breakpoints + task output.
        self.assertEqual(
            self._count_step(runner), self._batch_size * self._num_batches * 3
        )
