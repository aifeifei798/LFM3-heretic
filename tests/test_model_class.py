# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import tempfile
import unittest

import torch.nn
import transformers.integrations

from heretic.model import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    get_model_class,
)


def write_config(config: dict) -> str:
    directory = tempfile.mkdtemp()
    with open(f"{directory}/config.json", "w") as file:
        json.dump(config, file)
    return directory


class GetModelClassTests(unittest.TestCase):
    def test_native_vision_model(self) -> None:
        model_class = get_model_class(write_config({"vision_config": {}}))
        self.assertIs(model_class, AutoModelForImageTextToText)

    def test_native_audio_model(self) -> None:
        model_class = get_model_class(write_config({"audio_config": {}}))
        self.assertIs(model_class, AutoModelForMultimodalLM)

    def test_text_model(self) -> None:
        model_class = get_model_class(write_config({}))
        self.assertIs(model_class, AutoModelForCausalLM)

    def test_vision_model_with_matching_auto_map(self) -> None:
        model_class = get_model_class(
            write_config(
                {
                    "vision_config": {},
                    "auto_map": {
                        "AutoModel": "modeling.CustomModel",
                        "AutoModelForImageTextToText": "modeling.CustomModelForImageTextToText",
                    },
                }
            )
        )
        self.assertIs(model_class, AutoModelForImageTextToText)

    def test_vision_model_with_only_automodel_registered(self) -> None:
        # Custom-code checkpoints like the D1 family register only AutoModel,
        # because their class doesn't fit the standard task classes.
        model_class = get_model_class(
            write_config(
                {
                    "vision_config": {},
                    "auto_map": {
                        "AutoModel": "modeling.D1Model",
                        "AutoConfig": "configuration.CustomConfig",
                    },
                }
            )
        )
        self.assertIs(model_class, AutoModel)

    def test_audio_model_with_only_automodel_registered(self) -> None:
        model_class = get_model_class(
            write_config(
                {
                    "audio_config": {},
                    "auto_map": {
                        "AutoModel": "modeling.D1Model",
                        "AutoConfig": "configuration.CustomConfig",
                    },
                }
            )
        )
        self.assertIs(model_class, AutoModel)


class TolerantUseKernelizedFuncTests(unittest.TestCase):
    # heretic.model replaces transformers' use_kernelized_func with a version
    # that falls back to the plain PyTorch implementation when the kernels
    # library rejects the decorated function (vendored modeling code from newer
    # Transformers versions passes unregistered functions).

    def test_unregistered_function_falls_back_to_pytorch(self) -> None:
        def some_op(x):
            return x

        decorator = transformers.integrations.use_kernelized_func(some_op)

        class Layer(torch.nn.Module):
            pass

        decorated = decorator(Layer)
        self.assertFalse(hasattr(decorated(), "_kernel_funcs"))

    def test_registered_function_still_uses_kernels(self) -> None:
        def some_op(x):
            return x

        some_op.kernel_layer_name = "SomeOp"

        decorator = transformers.integrations.use_kernelized_func(some_op)

        class Layer(torch.nn.Module):
            pass

        decorated = decorator(Layer)
        layer = decorated()
        self.assertTrue(hasattr(layer, "_kernel_funcs"))


if __name__ == "__main__":
    unittest.main()
