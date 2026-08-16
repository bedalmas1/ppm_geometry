"""LUPIN (A6) model architecture - activity-only, text-narrative + BERT
fine-tuning, multi-head direct suffix classification.

**LICENSE NOTICE**: this module is a from-scratch reimplementation
referencing Pasquadibisceglie, Appice & Malerba, "LUPIN: A LLM Approach for
Activity Suffix Prediction in Business Process Event Logs" (ICPM 2024),
https://github.com/vinspdb/LUPIN, which is licensed **CC BY-NC-SA 4.0**
(non-commercial, share-alike) - unlike every other model in this project's
roster (Apache-2.0/MIT, or no-license-disclosed for A3). Per this project's
license-isolation policy (same handling as A7 MLMME's GPL-3.0, see
STATUS.md), `src/models/lupin/` (this file + adapter.py) must be treated as
a self-contained, separately-licensed unit: do not merge or mix this
directory's code into the rest of the codebase's otherwise permissively
licensed modules for redistribution, and do not use it for commercial
purposes. See LICENSE-lupin.txt (verbatim copy of the original repo's
LICENSE.md) alongside this file.

**Original architecture** (`neural_network/llamp_multiout.py`'s
`BertMultiOutputClassificationHeads`, `main.py`): encode the ENTIRE running
process instance as a natural-language "story" (one templated sentence per
event, built from Jinja2 templates in `utility/log_config.py` that weave in
activity, resource, elapsed-time-since-case-start, and several event/trace
attributes specific to each log), tokenize it with a pretrained LLM's own
tokenizer, fine-tune the pretrained encoder (`prajjwal1/bert-medium`, a
small 8-layer/512-hidden BERT variant from Turc et al. 2019's "Well-Read
Students Learn Better" distillation family - already a deliberately small
pretrained checkpoint in the original paper itself, not something this
project needs to shrink further), and attach ONE independent linear
classification head per suffix position (`output_layers`, a
`nn.ModuleList`), each trained with its own `CrossEntropyLoss` against the
activity-class label at that position. This is architecturally a "direct"
(non-autoregressive) suffix model in the same family as A5 CRTP-LSTM (one
forward pass predicts every future position at once), but via K independent
classification heads on a single pooled representation rather than a
per-timestep recurrent readout.

**Scope decision, consistent with every other roster model**: the original
templates weave in resource, timesincecasestart, and several dataset-specific
event/trace attributes (see the real `utility/log_config.py`'s `'helpdesk'`
entry: activity, resource, timesincecasestart, servicelevel, servicetype,
workgroup, product, customer, supportsection, responsiblesection). This
project's common schema (`src/data/schema.py`) carries only
case_id/activity/timestamp, and every other model in the roster (A1-A5) was
scoped to activity-only. LUPIN follows the same rule: `adapter.py`'s
`build_prefix_text` emits a minimal narrative template using ONLY the
activity sequence ("Activity X was performed. Then activity Y was
performed. ..."), dropping every other attribute. This is a much larger
proportional cut to LUPIN's own core idea (semantic-rich text stories) than
for any other model in the roster - LUPIN's principal contribution is
precisely the rich attribute-to-text encoding this project must drop for
cross-model comparability - and is disclosed explicitly in
paper/phase3_baseline_reproduction.md's A6 section, not hidden.

**A genuine architectural difference worth documenting, not a bug**: unlike
A1-A5's closed, project-specific activity vocabulary/tokenization, LUPIN
requires the pretrained LLM's OWN subword (WordPiece) tokenizer for its
*input* text (see adapter.py) - this is not optional, since fine-tuning a
pretrained BERT checkpoint only makes sense with the tokenizer it was
pretrained against. The *output* class vocabulary (the discrete activity
classes each classification head predicts) still reuses this project's own
closed activity vocabulary (`models.sutran.adapter.Vocab`/`build_vocab`,
built from the TRAIN split only, same as every other full-suffix model) -
so LUPIN is the only roster model with two DIFFERENT vocabularies in play:
a subword input vocabulary (BERT's own, ~30k WordPiece tokens, frozen/
pretrained) and a closed output class vocabulary (this project's own, built
per-dataset from TRAIN activities only).

Hyperparameters: `bert_model_name='prajjwal1/bert-medium'`, `learning_rate=
1e-5`, and early-stop `patience=5` are taken directly from the repo's own
`main.py`, not guessed. `batch_size`, `epochs`, and `max_token_length` are
NOT taken from the repo's own values (`batch_size=8`, `epochs<=50`,
`MAX_LEN=512`) - these are deliberate, MEASURED, disclosed compute-budget
deviations (see `configs/models/lupin.yaml`'s comments and
`paper/phase3_baseline_reproduction.md`'s A6 section): at the repo's own
batch_size=8 with this project's activity-only text at its true (uncapped)
TRAIN-split length, one epoch measured ~59 minutes on this project's
CPU-only hardware, making the repo's own 50-epoch budget impractical.
`batch_size=32` (better CPU throughput) and `max_token_length` CAPPED at 48
tokens (covering ~92-93% of TRAIN prefixes without any truncation - only
the longest tail loses its oldest events, via `truncation_side='left'`,
which is a genuine, disclosed, small accuracy-relevant deviation, unlike a
purely-efficiency-motivated cap that never truncates real content) bring
this to ~14 minutes/epoch; `epochs` capped at 15 (patience=5 kept) is
informed by this roster's own empirical fast-convergence pattern for
full-suffix models on Helpdesk (A5's best checkpoint at epoch 1, A3's at
epoch 2).

**A real compatibility bug caught before any training run** (this project's
now-standard pre-training smoke-test practice): `prajjwal1/bert-medium`'s
HuggingFace repo predates the modern `AutoConfig`/`AutoTokenizer`/
`AutoModel` registry convention - its `config.json` has no `model_type` key
and it ships no `tokenizer_config.json`/`tokenizer.json`. Under this
project's installed `transformers` (5.15.0, the `torch-hf` optional-
dependency group), `AutoModel.from_pretrained(...)` raises `ValueError:
Unrecognized model ... Should have a model_type key`, and
`AutoTokenizer.from_pretrained(...)` raises a sentencepiece-related
`ValueError` while trying to guess a slow-to-fast tokenizer conversion path.
Both are worked around by using the concrete `BertModel`/`BertTokenizerFast`
classes directly instead of the `Auto*` wrappers - confirmed working via a
smoke test (forward pass, correct `pooler_output` shape) before any real
training. Not a bug in this project's code or in the original repo (which
predates `transformers` 5.x entirely) - a genuine old-checkpoint/new-library
compatibility gap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class LupinActivityOnly(nn.Module):
    """Fine-tuned BERT encoder + one independent linear classification head
    per suffix position (`num_heads` == this project's shared `window_size`,
    see adapter.py's `get_window_size`, reused directly from A4's adapter).

    Reimplements `neural_network/llamp_multiout.py`'s
    `BertMultiOutputClassificationHeads` (encode -> pooler_output -> K
    independent `nn.Linear` heads), adapted to reuse this project's own
    (history, suffix) prefix/target definitions instead of the original's
    own per-position label-dict construction in `preprocessing/
    log_to_history.py`.
    """

    def __init__(
        self,
        num_classes: int,
        num_heads: int,
        bert_model_name: str = "prajjwal1/bert-medium",
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.num_heads = num_heads
        self.output_heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_heads)])

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns BERT's pooled [CLS] representation (batch, hidden_size) -
        this project's z_t extraction point for LUPIN (Phase 4). Unlike
        every other roster model (one vector per PREFIX LENGTH per forward
        pass), LUPIN encodes one whole prefix's text in a single sequence
        input, so this is naturally already one vector per prefix - no
        length-indexed readout convention is needed, architecturally closest
        to A3 RLHGNN's single pooled-per-prefix vector, though via a
        pretrained-LLM [CLS]/pooler mechanism rather than graph pooling."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Direct (non-autoregressive) suffix prediction: one forward pass
        returns logits for every suffix position at once (batch, num_heads,
        num_classes) - used identically at train and eval time, same
        convention as A5 CRTP-LSTM."""
        pooled = self.encode(input_ids, attention_mask)
        logits = torch.stack([head(pooled) for head in self.output_heads], dim=1)
        return logits
