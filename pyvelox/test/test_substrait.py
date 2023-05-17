print("In Python")
import pyvelox.pyvelox as pv
import unittest

def compare_vectors(v1, v2):
    N = len(v1)
    assert N == len(v2)
    assert v1.dtype == v2.dtype
    
    for idx in range(N):
        assert v1[idx] == v2[idx]

class TestVeloxSubstrait(unittest.TestCase):
    def test_simple(self):
        plan_path = "/home/asus/github/fork/velox/velox/substrait/tests/data/substrait_virtualTable.json"
        print("*" * 80)
        res = pv.run_substrait_query(plan_path)
        print(res)
        print(dir(res))
        print(len(res))
        
        """
        RowVectorPtr expectedData = makeRowVector(
      {makeFlatVector<int64_t>(
           {2499109626526694126, 2342493223442167775, 4077358421272316858}),
       makeFlatVector<int32_t>({581869302, -708632711, -133711905}),
       makeFlatVector<double>(
           {0.90579193414549275, 0.96886777112423139, 0.63235925003444637}),
       makeFlatVector<bool>({true, false, false}),
       makeFlatVector<int32_t>(3, nullptr, nullEvery(1))

      });
        """
        
        expected_vectors = [
            pv.from_list([2499109626526694126, 2342493223442167775, 4077358421272316858]),
            pv.from_list([581869302, -708632711, -133711905]),
            pv.from_list([0.90579193414549275, 0.96886777112423139, 0.63235925003444637]),
            pv.from_list([True, False, False]),
            pv.from_list([3, None, None]),
        ]
        
        for i in range(len(res)):
            vec = res[i]
            exp_vec = expected_vectors[i]
            N = len(vec)
            for i in range(N):
                v1 = vec[i]
                v2 = exp_vec[i]
                print(v1, v2)
        