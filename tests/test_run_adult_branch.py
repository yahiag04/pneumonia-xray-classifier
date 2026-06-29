import unittest

from scripts.run_adult_branch import build_model_list, evaluation_output_name


class RunAdultBranchTest(unittest.TestCase):
    def test_build_model_list_defaults_to_all_comparison_models(self):
        self.assertEqual(
            build_model_list(None),
            [
                "pneumonia_net",
                "resnet18",
                "mobilenet_v3_large",
                "efficientnet_b0",
                "densenet121",
            ],
        )

    def test_evaluation_output_name_includes_model_and_dataset(self):
        self.assertEqual(
            evaluation_output_name("resnet18", "chittagong"),
            "resnet18_chittagong.json",
        )


if __name__ == "__main__":
    unittest.main()
