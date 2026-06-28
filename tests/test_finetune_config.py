import unittest

import torch
import torch.nn as nn

from scripts.finetune_model import (
    build_optimizer,
    build_run_dir,
    keep_frozen_modules_eval,
    trainable_parameters,
)


class FineTuneConfigTest(unittest.TestCase):
    def test_trainable_parameters_returns_only_requires_grad_parameters(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False

        params = list(trainable_parameters(model))

        self.assertEqual(params, list(model[1].parameters()))

    def test_build_optimizer_rejects_model_without_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(2, 1))
        for parameter in model.parameters():
            parameter.requires_grad = False

        with self.assertRaisesRegex(ValueError, "No trainable parameters"):
            build_optimizer(model, lr=1e-5)

    def test_keep_frozen_modules_eval_preserves_batchnorm_stats(self):
        model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
        )
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        model.train()

        keep_frozen_modules_eval(model)
        before = model[0].running_mean.clone()
        model(torch.ones(4, 2))

        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        self.assertTrue(torch.equal(model[0].running_mean, before))

    def test_build_run_dir_includes_trainable_mode(self):
        self.assertEqual(
            build_run_dir("outputs/runs_improved", "resnet18", "head"),
            "outputs/runs_improved/resnet18_head",
        )
        self.assertEqual(
            build_run_dir("outputs/runs_improved/resnet18_head", "resnet18", "head"),
            "outputs/runs_improved/resnet18_head",
        )


if __name__ == "__main__":
    unittest.main()
