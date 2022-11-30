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
        self.assertTrue(isinstance(pv.BaseVector.from_list(['hello', 'world']), pv.BaseVector))
        with self.assertRaises(TypeError):
            pv.BaseVector.from_list(['hello', 3.14])
        with self.assertRaises(ValueError):
            pv.BaseVector.from_list([None, None, None])
        with self.assertRaises(ValueError):
            pv.BaseVector.from_list([])

    def test_to_string(self):
        self.assertEqual(str(pv.BaseVector.from_list([1, 2, 3])), '[FLAT BIGINT: 3 elements, no nulls]')
        self.assertEqual(str(pv.BaseVector.from_list([1, None, 3])), '[FLAT BIGINT: 3 elements, 1 nulls]')

    def test_get_item(self):
        ints = pv.BaseVector.from_list([1, 2, None, None, 3])
        self.assertEqual(ints[0], 1)
        self.assertEqual(ints[1], 2)
        self.assertEqual(ints[2], None)
        self.assertEqual(ints[3], None)
        self.assertEqual(ints[4], 3)

        strs = pv.BaseVector.from_list(['hello', 'world', None])
        self.assertEqual(strs[0], 'hello')
        self.assertEqual(strs[1], 'world')
        self.assertEqual(strs[2], None)
        self.assertNotEqual(strs[0], 'world')
        self.assertNotEqual(strs[2], 'world')
        
        with self.assertRaises(IndexError):
            ints[5]
        with self.assertRaises(IndexError):
            ints[-1]
        with self.assertRaises(IndexError):
            strs[1000]
        with self.assertRaises(IndexError):
            strs[-1000]

    def test_set_item(self):
        ints = pv.BaseVector.from_list([1, 2, None, None, 3])
        self.assertEqual(ints[2], None)
        ints[2] = 10
        self.assertEqual(ints[2], 10)

        strs = pv.BaseVector.from_list(['googly', 'doogly'])
        self.assertEqual(strs[1], 'doogly')
        strs[1] = 'moogly'
        self.assertEqual(strs[1], 'moogly')

        with self.assertRaises(IndexError):
            ints[5] = 10
        with self.assertRaises(IndexError):
            ints[-1] = 10
        with self.assertRaises(IndexError):
            strs[1000] = 'hi'
        with self.assertRaises(IndexError):
            strs[-1000] = 'bye'
        with self.assertRaises(TypeError):
            ints[3] = 'ni hao'
        with self.assertRaises(TypeError):
            strs[0] = 2

    def test_length(self):
        ints = pv.BaseVector.from_list([1, 2, None])
        self.assertEqual(len(ints), 3)

        strs = pv.BaseVector.from_list(['hi', 'bye'])
        self.assertEqual(len(strs), 2)

    def test_numeric_limits(self):
        bigger_than_int32 = pv.BaseVector.from_list([1 << 33])
        self.assertEqual(bigger_than_int32[0], 1 << 33)
        with self.assertRaises(RuntimeError):
            bigger_than_int64 = pv.BaseVector.from_list([1 << 63])
        smaller_than_int64 = pv.BaseVector.from_list([(1 << 62) + (1 << 62) - 1])
