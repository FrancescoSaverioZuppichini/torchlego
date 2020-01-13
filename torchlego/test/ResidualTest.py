import unittest
import torch
from torchlego.blocks import Residual, Lambda

class ResidualTest(unittest.TestCase):
    def test_add(self):
        x = torch.tensor(1)
        add_one = Lambda(lambda x: x + 1)
        adder = Residual([add_one])
        # 2 + 1
        self.assertEqual(adder(x), 3)
        adder = Residual([add_one, add_one])
        self.assertEqual(adder(x), 7)

    def test_concat(self):
        x = torch.tensor([[1]])
        a = Lambda(lambda x: x)
        catter = Residual([a], mode='cat')
        self.assertEqual(catter(x).sum(), 2)
        catter = Residual([a, a], mode='cat')
        self.assertEqual(catter(x).sum(), 4)




if __name__ == '__main__':
    unittest.main()
