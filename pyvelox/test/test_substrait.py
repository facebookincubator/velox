print("In Python")
import pyvelox.pyvelox as pv
import unittest

class TestVeloxSubstrait(unittest.TestCase):
    def test_simple(self):

        plan_path = "/home/asus/github/fork/velox/velox/substrait/tests/data/substrait_virtualTable.json"
        #res = pv.run_substrait_query(plan_path)
        #print(res)
        print("*" * 80)
        res = pv.run_substrait_query_with_builder(plan_path)
        print(type(res))
        print(len(res))
        print(res)
