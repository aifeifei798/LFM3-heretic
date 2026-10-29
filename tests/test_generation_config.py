# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import tempfile
import unittest

from transformers import GenerationConfig

from heretic.model import sanitize_generation_config

# The generation config shipped with the LiquidAI LFM3-Audio-2B checkpoint, which
# Transformers 5 refuses to save (temperature/top_k set while do_sample is disabled).
LFM3_AUDIO_GENERATION_CONFIG = {
    "bos_token_id": 124894,
    "eos_token_id": [124900, 125014],
    "pad_token_id": 124893,
    "use_cache": True,
    "output_attentions": False,
    "output_hidden_states": False,
    "do_sample": False,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_k": 1,
    "audio_temperature": 1.0,
    "audio_top_k": 128,
    "audio_repetition_penalty": 1.1,
    "audio_repetition_window": 4096,
}


class SanitizeGenerationConfigTests(unittest.TestCase):

    def saved_generation_config(self, generation_config: GenerationConfig) -> dict:
        # save_pretrained validates with strict=True and raises on invalid configs.
        with tempfile.TemporaryDirectory() as directory:
            generation_config.save_pretrained(directory)
            with open(f"{directory}/generation_config.json") as file:
                return json.load(file)

    def test_valid_config_untouched(self) -> None:
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_k=50,
            top_p=0.9,
        )

        self.assertEqual(sanitize_generation_config(generation_config), [])
        self.assertEqual(generation_config.temperature, 0.1)
        self.assertEqual(
            self.saved_generation_config(generation_config)["temperature"],
            0.1,
        )

    def test_removes_sampling_flags_from_greedy_config(self) -> None:
        generation_config = GenerationConfig(**LFM3_AUDIO_GENERATION_CONFIG)

        removed = sanitize_generation_config(generation_config)

        self.assertEqual(removed, ["temperature", "top_k"])
        self.assertEqual(generation_config.do_sample, False)
        # Unrelated flags, including custom ones like the audio sampling
        # parameters, must survive untouched.
        self.assertEqual(generation_config.pad_token_id, 124893)
        self.assertEqual(generation_config.audio_temperature, 1.0)
        self.assertEqual(generation_config.audio_top_k, 128)
        saved = self.saved_generation_config(generation_config)
        self.assertNotIn("temperature", saved)
        self.assertNotIn("top_k", saved)
        self.assertEqual(saved["do_sample"], False)
        self.assertEqual(saved["audio_top_k"], 128)

    def test_removes_beam_flags_from_single_beam_config(self) -> None:
        generation_config = GenerationConfig(
            num_beams=1,
            length_penalty=2.0,
            early_stopping=True,
        )

        removed = sanitize_generation_config(generation_config)

        self.assertEqual(removed, ["early_stopping", "length_penalty"])
        self.saved_generation_config(generation_config)

    def test_keeps_beam_flags_when_beam_search_selected(self) -> None:
        generation_config = GenerationConfig(
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

        self.assertEqual(sanitize_generation_config(generation_config), [])
        self.saved_generation_config(generation_config)

    def test_partially_fixable_config(self) -> None:
        # The sampling flags are removed, but the config remains invalid for a
        # reason the sanitization doesn't handle; it must not crash and must
        # still report the flags it did remove.
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            pad_token_id=-1,
        )

        with self.assertRaises(ValueError):
            generation_config.validate(strict=True)

        removed = sanitize_generation_config(generation_config)

        self.assertEqual(removed, ["temperature"])
        with self.assertRaises(ValueError):
            generation_config.validate(strict=True)


if __name__ == "__main__":
    unittest.main()
