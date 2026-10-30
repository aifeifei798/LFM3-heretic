# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import re
import tempfile
import unittest

import torch.nn
import transformers.integrations
from transformers.core_model_loading import WeightRenaming

from heretic.model import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    _make_dollar_safe_process_source_pattern,
    _make_dollar_safe_process_target_pattern,
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


# Stand-ins for `process_target_pattern` / `process_source_pattern` as they behave
# in 5.21.0+ (the `$`-anchor handling was dropped) and in 5.18.0-5.20.0 (reference).
def fake_521_process_target_pattern(pattern: str) -> tuple[str, str | None]:
    processed = pattern.removeprefix("^")
    processed = re.sub(r"\(\?.+?\)?\)", "", processed)
    processed = processed.replace(r"\.", ".")
    capturing_group_match = re.search(r"\(.+?\)", processed)
    return processed, capturing_group_match.group(0) if capturing_group_match else None


def fake_518_process_target_pattern(pattern: str) -> tuple[str, str | None]:
    processed, captured_group = fake_521_process_target_pattern(pattern)
    return processed.removesuffix("$"), captured_group


def fake_521_process_source_pattern(source_pattern: str, target_pattern: str) -> str:
    if target_pattern.startswith("^") and not source_pattern.startswith("^"):
        source_pattern = f"^{source_pattern}"
    return source_pattern


def fake_518_process_source_pattern(source_pattern: str, target_pattern: str) -> str:
    source_pattern = fake_521_process_source_pattern(source_pattern, target_pattern)
    if target_pattern.endswith("$") and not source_pattern.endswith("$"):
        source_pattern = f"{source_pattern}$"
    return source_pattern


class DollarSafeReverseRenamingPatchTests(unittest.TestCase):
    def test_target_wrapper_strips_leaked_dollar(self) -> None:
        wrapper = _make_dollar_safe_process_target_pattern(fake_521_process_target_pattern)
        self.assertEqual(wrapper("^mask_embedding$"), ("mask_embedding", None))

    def test_target_wrapper_idempotent_on_fixed_version(self) -> None:
        wrapper = _make_dollar_safe_process_target_pattern(fake_518_process_target_pattern)
        self.assertEqual(wrapper("^mask_embedding$"), ("mask_embedding", None))

    def test_source_wrapper_restores_end_anchor(self) -> None:
        wrapper = _make_dollar_safe_process_source_pattern(fake_521_process_source_pattern)
        self.assertEqual(wrapper("model.probe", "^probe$"), "^model.probe$")

    def test_source_wrapper_idempotent_on_fixed_version(self) -> None:
        wrapper = _make_dollar_safe_process_source_pattern(fake_518_process_source_pattern)
        self.assertEqual(wrapper("model.probe", "^probe$"), "^model.probe$")

    def test_real_reverse_rename_round_trip(self) -> None:
        # The contract through the real (patched at heretic.model import time)
        # transformers functions, for the rule that exposed the bug.
        transform = WeightRenaming(
            source_patterns=r"^mask_embedding$", target_patterns="model.depth_decoder.mask_embedding"
        )
        renamed_key, _ = transform.reverse_transform().rename_source_key(
            "model.depth_decoder.mask_embedding"
        )
        self.assertEqual(renamed_key, "mask_embedding")


if __name__ == "__main__":
    unittest.main()
