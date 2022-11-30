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

import pyvelox.pyvelox as pv
import unittest

class TestVeloxVector(unittest.TestCase):
    def test_from_list(self):
        self.assertTrue(isinstance(pv.BaseVector.from_list([1, 2, 3]), pv.BaseVector))
        self.assertTrue(isinstance(pv.BaseVector.from_list([1, None, None]), pv.BaseVector))

    def test_to_string(self):
        self.assertEqual(str(pv.BaseVector.from_list([1, 2, 3])), '[FLAT BIGINT: 3 elements, no nulls]')
        self.assertEqual(str(pv.BaseVector.from_list([1, None, 3])), '[FLAT BIGINT: 3 elements, 1 nulls]')
