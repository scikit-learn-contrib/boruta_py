import unittest
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


class BorutaTestCases(unittest.TestCase):

    def test_get_tree_num(self):
        rfc = RandomForestClassifier(max_depth=10)
        bt = BorutaPy(rfc)
        self.assertEqual(bt._get_tree_num(10),44,"Tree Est. Math Fail")
        self.assertEqual(bt._get_tree_num(100),141,"Tree Est. Math Fail")



if __name__ == '__main__':
    unittest.main()


