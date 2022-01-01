import ml_lib.utils.combos as combos
import unittest


class CombosTest(unittest.TestCase):

    def test_power_set(self):
        test_in = (1, 2, 3)
        ret = combos.power_set(test_in, skip_empty_set=True)
        self.assertEqual(7, len(set(ret)))
        ret = combos.power_set(test_in, skip_empty_set=False)
        self.assertEqual(8, len(set(ret)))
