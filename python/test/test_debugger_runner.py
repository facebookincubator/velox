# Copyright (c) Facebook, Inc. and its affiliates.
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
