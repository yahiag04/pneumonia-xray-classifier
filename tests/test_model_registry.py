import unittest

import torch
import torch.nn as nn

from thesis.model_registry import (
    available_models,
    build_model,
    configure_trainable_layers,
    freeze_backbone,
)


class DummyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(2, 1))


class ModelRegistryTest(unittest.TestCase):
    def test_available_models_contains_recommended_comparison_set(self):
        names = set(available_models())

        self.assertIn("pneumonia_net", names)
        self.assertIn("resnet18", names)
        self.assertIn("resnet50", names)
        self.assertIn("densenet121", names)
        self.assertIn("efficientnet_b0", names)
        self.assertIn("mobilenet_v3_large", names)

    def test_pneumonia_net_outputs_single_logit(self):
        model = build_model("pneumonia_net", pretrained=False)
        model.eval()

        with torch.no_grad():
            out = model(torch.zeros(2, 1, 224, 224))

        self.assertEqual(tuple(out.shape), (2, 1))

    def test_freeze_backbone_keeps_resnet_classifier_trainable(self):
        model = build_model("resnet18", pretrained=False)

        freeze_backbone(model, "resnet18")

        frozen_backbone = [p.requires_grad for name, p in model.named_parameters() if not name.startswith("fc.")]
        trainable_head = [p.requires_grad for name, p in model.named_parameters() if name.startswith("fc.")]
        self.assertTrue(frozen_backbone)
        self.assertTrue(trainable_head)
        self.assertTrue(all(not flag for flag in frozen_backbone))
        self.assertTrue(all(trainable_head))


class TrainableModeTest(unittest.TestCase):
    def test_configure_trainable_layers_head_mode_only_unfreezes_classifier(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "head")

        self.assertFalse(any(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_configure_trainable_layers_last_block_unfreezes_last_feature_block_and_classifier(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "last_block")

        self.assertFalse(any(parameter.requires_grad for parameter in model.features[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.features[-1].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_configure_trainable_layers_all_mode_unfreezes_everything(self):
        model = DummyEfficientNet()

        configure_trainable_layers(model, "efficientnet_b0", "all")

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
