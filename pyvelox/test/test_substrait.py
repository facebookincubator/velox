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
import json
import unittest
import shutil, tempfile
import os

def compare_vectors(v1, v2):
    N = len(v1)
    assert N == len(v2)
    assert v1.dtype == v2.dtype
    
    for idx in range(N):
        assert v1[idx] == v2[idx]


# Run the tests from the base path of the project

BASE_PATH = os.path.join(os.getcwd(), "velox/substrait/tests/data/")
class TestVeloxSubstrait(unittest.TestCase):
    def setUp(self):
        # create a temporary directory
        pv.initialize_substrait()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove the temporary directory
        pv.finalize_substrait()
        shutil.rmtree(self.test_dir)
        
    def test_make_row_vectors(self):
        vals = [2499109626526694126, 2342493223442167775, 4077358421272316858]
        expected_vectors = [
            pv.from_list(vals),
        ]
        
        def set_null(n: int) -> bool:
            return False
        
        rw = pv.row_vector(["a"], expected_vectors, set_null)
        
        for i in range(len(rw)):
            vec = rw[i]
            N = len(vec)
            for i in range(N):
                assert vec[i] == vals[i]

        file_name = "d1.orc"
        save_path = os.path.join(self.test_dir, file_name)
        pv.save_row_vector([rw], save_path)
        files = os.listdir(self.test_dir)
        assert os.path.exists(save_path)
        assert file_name in files
    
    def test_simple(self):
        other_path = BASE_PATH + "substrait_virtualTable.json"
        
        res = pv.run_substrait_query(other_path)
        
        expected_vectors = [
            pv.from_list([2499109626526694126, 2342493223442167775, 4077358421272316858]),
            pv.from_list([581869302, -708632711, -133711905]),
            pv.from_list([0.90579193414549275, 0.96886777112423139, 0.63235925003444637]),
            pv.from_list([True, False, False]),
        ]

        for i in range(len(res) - 1):
            vec = res[i]
            exp_vec = expected_vectors[i]
            N = len(vec)
            for i in range(N):
                assert vec[i] == exp_vec[i]

    def test_aggregates(self):
        substrait_plan = "q6_first_stage.json"
        plan_path = BASE_PATH + substrait_plan
        
        l_order_key_data = [
            4636438147,
            2012485446,
            1635327427,
            8374290148,
            2972204230,
            8001568994,
            989963396,
            2142695974,
            6354246853,
            4141748419
        ]
        
        l_part_key_data = [
            263222018,
            255918298,
            143549509,
            96877642,
            201976875,
            196938305,
            100260625,
            273511608,
            112999357,
            299103530 
        ]
        
        l_supp_key_data = [
            2102019,
            13998315,
            12989528,
            4717643,
            9976902,
            12618306,
            11940632,
            871626,
            1639379,
            3423588
        ]
        
        l_line_number_data = [4, 6, 1, 5, 1, 2, 1, 5, 2, 6]
        
        l_quantity_data = [6.0, 1.0, 19.0, 4.0, 6.0, 12.0, 23.0, 11.0, 16.0, 19.0]
        
        l_extended_price_data = [
            30586.05,
            7821.0,
            1551.33,
            30681.2,
            1941.78,
            66673.0,
            6322.44,
            41754.18,
            8704.26,
            63780.36
        ]
        
        l_discount_data = [0.05, 0.06, 0.01, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07]
        
        l_tax_data = [0.02, 0.03, 0.01, 0.0, 0.01, 0.01, 0.03, 0.07, 0.01, 0.04]
        
        l_return_flag_data = ["N", "A", "A", "R", "A", "N", "A", "A", "N", "R"]
        
        l_line_status_data = ["O", "F", "F", "F", "F", "O", "F", "F", "O", "F"]
        
        l_ship_date_new_data = [
            8953.666666666666,
            8773.666666666666,
            9034.666666666666,
            8558.666666666666,
            9072.666666666666,
            8864.666666666666,
            9004.666666666666,
            8778.666666666666,
            9013.666666666666,
            8832.666666666666
        ]
        
        l_commit_date_new_data = [
            10447.666666666666,
            8953.666666666666,
            8325.666666666666,
            8527.666666666666,
            8438.666666666666,
            10049.666666666666,
            9036.666666666666,
            8666.666666666666,
            9519.666666666666,
            9138.666666666666
        ]
        
        l_receipt_date_new_data = [
            10456.666666666666,
            8979.666666666666,
            8299.666666666666,
            8474.666666666666,
            8525.666666666666,
            9996.666666666666,
            9103.666666666666,
            8726.666666666666,
            9593.666666666666,
            9178.666666666666
        ]
        
        l_ship_instruct_data = [
            "COLLECT COD",
            "NONE",
            "TAKE BACK RETURN",
            "NONE",
            "TAKE BACK RETURN",
            "NONE",
            "DELIVER IN PERSON",
            "DELIVER IN PERSON",
            "TAKE BACK RETURN",
            "NONE"
        ]
        
        l_ship_mode_data = [
            "FOB",
            "REG AIR",
            "MAIL",
            "FOB",
            "RAIL",
            "SHIP",
            "REG AIR",
            "REG AIR",
            "TRUCK",
            "AIR" 
        ]
        
        l_comment_data = [
            " the furiously final foxes. quickly final p",
            "thely ironic",
            "ate furiously. even, pending pinto bean",
            "ackages af",
            "odolites. slyl",
            "ng the regular requests sleep above",
            "lets above the slyly ironic theodolites sl",
            "lyly regular excuses affi",
            "lly unusual theodolites grow slyly above",
            " the quickly ironic pains lose car"
        ]
        
        data = [
            pv.from_list(l_order_key_data), 
            pv.from_list(l_part_key_data), 
            pv.from_list(l_supp_key_data), 
            pv.from_list(l_line_number_data), 
            pv.from_list(l_quantity_data), 
            pv.from_list(l_extended_price_data),
            pv.from_list(l_discount_data), 
            pv.from_list(l_tax_data), 
            pv.from_list(l_return_flag_data), 
            pv.from_list(l_line_status_data), 
            pv.from_list(l_ship_date_new_data),
            pv.from_list(l_commit_date_new_data),
            pv.from_list(l_receipt_date_new_data),
            pv.from_list(l_ship_instruct_data),
            pv.from_list(l_ship_mode_data),
            pv.from_list(l_comment_data)
        ]
        
        def set_null(n: int) -> bool:
            return False

        names = [
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment"
        ]
        row_vec = pv.row_vector(names, data, set_null)
        row_vec_exp = pv.row_vector(["sum#43"], [pv.from_list([13613.1921])], set_null)
        
        pv.save_row_vector([row_vec], self.test_dir + "/mock_lineitem.orc")
        
        assert os.path.exists(self.test_dir + "/mock_lineitem.orc")

        res = pv.run_substrait_query(plan_path, True, self.test_dir)


        for i in range(len(res)):
            vec = res[i]
            exp_vec = row_vec_exp[i]
            N = len(vec)
            for i in range(N):
                assert vec[i] == exp_vec[i]

    def test_file_type_failure(self):
        other_path = self.test_dir + "dummy_plan.txt"
        with open(other_path, "w") as fp:
            fp.write("no json")

        with self.assertRaises(ValueError) as cm:
            pv.run_substrait_query(other_path)
        self.assertEqual('plan should be path to a plan in JSON format.', str(cm.exception))

    def test_plan_str_json_format(self):
        other_path = BASE_PATH + "substrait_virtualTable.json"
        with open(other_path, 'r') as f:
            data = f.read()

        res = pv.run_substrait_query(data)

        expected_vectors = [
            pv.from_list([2499109626526694126, 2342493223442167775, 4077358421272316858]),
            pv.from_list([581869302, -708632711, -133711905]),
            pv.from_list([0.90579193414549275, 0.96886777112423139, 0.63235925003444637]),
            pv.from_list([True, False, False]),
        ]

        for i in range(len(res) - 1):
            vec = res[i]
            exp_vec = expected_vectors[i]
            N = len(vec)
            for i in range(N):
                assert vec[i] == exp_vec[i]
