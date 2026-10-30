# LFM3-heretic

[Heretic](https://github.com/p-e-w/heretic) — fully automatic censorship removal
for language models — with first-class support for the **LiquidAI LFM3 family**,
the **D1 family**, and hardening against several Transformers 5.x regressions in
the export and evaluation paths.

This is a maintained fork of [p-e-w/heretic](https://github.com/p-e-w/heretic)
(currently tracking upstream `master`, `2.0.0.dev0`). Everything upstream
documents about [how Heretic works](https://github.com/p-e-w/heretic#how-heretic-works)
(directional ablation + TPE parameter optimization) applies here unchanged.

## Supported checkpoints

| Model | Modalities | Loaded as | Notes |
|---|---|---|---|
| [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | text → text | `AutoModelForCausalLM` | |
| [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) | image/video → text | `AutoModelForImageTextToText` | |
| [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | audio → text/speech | `AutoModelForMultimodalLM` | speech output never requested; text responses suffice for scoring |
| [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) | image/video/audio → text/speech | `AutoModelForImageTextToText` | |
| D1 family (`d1-2B`, `d1-Omni-2B`) | like VL / Omni | `AutoModel` | checkpoints register only `AutoModel` in their `auto_map` |

All of these share the same 24-layer hybrid text decoder, so abliteration
targets the same components in every variant: `conv.out_proj` (short-conv
layers), `self_attn.o_proj` (attention layers), and `feed_forward.w2` (both).
The vision tower, audio encoder, and speech depth decoder are never modified.

## Changes compared to upstream

- **LFM3 family support** — audio-text and speech-output checkpoints: correct
  model class detection, processor loading, custom `generate()` handling
  (Heretic falls back to a plain forward pass for residual/logit analysis),
  and a [`config.lfm3.toml`](config.lfm3.toml) covering the whole family.
- **D1 family support** — checkpoints whose `auto_map` registers only
  `AutoModel` fall back to it instead of failing; modeling code vendored from
  unreleased Transformers versions (with `@use_kernelized_func(<plain function>)`)
  imports cleanly via a tolerant wrapper that falls back to the PyTorch
  implementation when a hub kernel is not registered.
- **Export hardening: generation config** — Transformers 5.x validates
  generation configs strictly at save time and rejects configs that set
  sampling flags next to `do_sample: false` (as shipped by LFM3-Audio/Omni).
  Merged-model export now sanitizes such flags (they are no-ops in the
  configured mode) instead of failing.
- **Export hardening: reverse key renaming** — Transformers 5.21.0 dropped the
  `$`-anchor handling from `process_target_pattern()`/`process_source_pattern()`,
  corrupting saved keys for models with end-anchored `WeightRenaming` rules
  (`mask_embedding$`, which then silently reloads as random weights). Both
  functions are probed at import time and wrapped with the 5.18.0–5.20.0
  behavior on affected versions. See
  [huggingface/transformers#50177](https://github.com/huggingface/transformers/issues/50177),
  fix pending in
  [#50178](https://github.com/huggingface/transformers/pull/50178).
- **`--evaluate-model` fix** — the evaluation flow could never evaluate
  custom-code checkpoints (the reload didn't trust the evaluated path). The
  same trust policy as model loading now applies: explicit
  `trust_remote_code = true`, or unset + local path.
- **`trust_remote_code` setting** — explicit override added; by default,
  local paths are trusted (the files were obtained deliberately beforehand)
  while Hub IDs still prompt.

## Usage

Same workflow as upstream Heretic, installed from this repository rather than
PyPI:

```sh
git clone https://github.com/aifeifei798/LFM3-heretic
cd LFM3-heretic
uv sync
uv run heretic LiquidAI/LFM3-Omni-2B
```

For the LFM3 family, start from [`config.lfm3.toml`](config.lfm3.toml):

```sh
cp config.lfm3.toml config.toml   # then set `model` to the checkpoint you want
uv run heretic                    # or: uv run heretic LiquidAI/LFM3-Audio-2B
```

Local checkpoint paths work directly (and are trusted automatically):

```toml
model = "/path/to/LFM3-Omni-2B"
```

Interrupted or completed runs are kept in `checkpoints/` and resumed
automatically; re-running a finished study lets you export a different trial
of the same run without re-optimizing. To score an existing decensored model
against its original with Heretic's own scorers (Refusals + KL divergence):

```toml
model = "/path/to/LFM3-Audio-2B"
evaluate_model = "/path/to/LFM3-Audio-2B-heretic"
```

## Verified results

Best Pareto trial per model, exported and re-evaluated on the exported
checkpoints with Heretic's own scorers (100 harmful prompts for Refusals,
100 harmless prompts for KL divergence against the original model;
transformers 5.18.0):

| Model | Baseline refusals | Heretic refusals | KL divergence |
|---|---|---|---|
| LFM3-1.7B | 91/100 | **0/100** | 3.62 |
| LFM3-VL-2B | 95/100 | **0/100** | **0.23** |
| LFM3-Omni-2B | 99/100 | **0/100** | 2.16 |
| LFM3-Audio-2B | 86/100 | **0/100** | 3.55 |

Refusal 0/100 means every harmful prompt was answered; KL divergence measures
how far the decensored model's token distributions drift from the original on
harmless prompts (lower is closer to the original's intelligence). The Pareto
front contains more conservative points trading a small number of refusals for
lower KL — any trial of a completed run can be exported without re-optimizing.

## Transformers compatibility

| Version | Status |
|---|---|
| 5.18.0 – 5.20.0 | clean, verified |
| 5.21.0 – 5.22.x | two save-time regressions; both auto-patched at import |
| any version | D1-family modeling code (vendored from unreleased Transformers) imports via the tolerant kernel wrapper |

## License

Copyright &copy; 2025-2026  Philipp Emanuel Weidmann (<pew@worldwidemann.com>) + contributors
Fork changes &copy; 2026 the LFM3-heretic contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

**By contributing to this project, you agree to release your
contributions under the same license.**
