import unittest

import torch.nn as nn

from scripts.finetune_model import trainable_parameters


class FineTuneConfigTest(unittest.TestCase):
    def test_trainable_parameters_returns_only_requires_grad_parameters(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        params = list(trainable_parameters(model))

        self.assertEqual(params, list(model[1].parameters()))


if __name__ == "__main__":
    unittest.main()
