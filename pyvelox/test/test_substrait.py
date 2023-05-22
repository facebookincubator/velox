print("In Python")
import pyvelox.pyvelox as pv
import unittest

def compare_vectors(v1, v2):
    N = len(v1)
    assert N == len(v2)
    assert v1.dtype == v2.dtype
    
    for idx in range(N):
        assert v1[idx] == v2[idx]


BASE_PATH = "/home/asus/github/fork/velox/velox/substrait/tests/data/"
class TestVeloxSubstrait(unittest.TestCase):
    def test_simple(self):
        other_path = BASE_PATH + "substrait_virtualTable.json"
        print("*" * 80)
        res = pv.run_substrait_query(other_path, False)
        print(res)
        print(dir(res))
        print(len(res))
        
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
        substrait_plan = "substrait_virtualTable.json"
        substrait_other_plan = "q6_first_stage.json"
        plan_path = BASE_PATH + substrait_other_plan
        print("*" * 80)
        res = pv.run_substrait_query(plan_path, True)
        print(res)
        print(dir(res))
        print(len(res))
        
        # expected_vectors = [
        #     pv.from_list([2499109626526694126, 2342493223442167775, 4077358421272316858]),
        #     pv.from_list([581869302, -708632711, -133711905]),
        #     pv.from_list([0.90579193414549275, 0.96886777112423139, 0.63235925003444637]),
        #     pv.from_list([True, False, False]),
        # ]

        # for i in range(len(res) - 1):
        #     vec = res[i]
        #     exp_vec = expected_vectors[i]
        #     N = len(vec)
        #     for i in range(N):
        #         # assert vec[i] == exp_vec[i]
        #         print(vec[i])
        