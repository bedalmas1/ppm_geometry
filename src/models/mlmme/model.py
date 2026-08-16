"""MLMME (A7) model architecture - activity-only, adversarially-trained
(GAN) LSTM encoder-decoder, full-suffix prediction.

Taymouri, La Rosa & Erfani, "A Deep Adversarial Model for Suffix and
Remaining Time Prediction of Event Sequences," SDM 2021 (arXiv:2102.07298,
fetched and read in full - not relied on from memory or the abstract),
https://github.com/farbodtaymouri/MLMME (cloned to a scratch location and
read in full: network.py, main.py, preparation.py, suffix.py - never added
to this repo). The paper itself names the method "MLMME" = Maximum
Likelihood Min-Max Estimation (its Algorithm 1).

**LICENSE NOTICE**: MLMME's source is licensed **GPL-3.0**
(https://github.com/farbodtaymouri/MLMME/blob/main/LICENSE) - stronger
copyleft than any other model in this project's roster (Apache-2.0/MIT for
most, CC BY-NC-SA 4.0 for A6 LUPIN, no-license-disclosed for A3 RLHGNN). Per
this project's license-isolation policy (identical handling to A6 LUPIN's,
see STATUS.md and src/models/lupin/'s equivalent notice), `src/models/mlmme/`
(this file + adapter.py) is a from-scratch reimplementation against the
actual paper equations (Sec. 4) and the actual repo code, not a verbatim
copy, and is kept isolated: do not merge or mix this directory's code into
the rest of this project's otherwise permissively licensed modules for
redistribution purposes. See LICENSE-mlmme.txt (verbatim copy of the
original repo's GPL-3.0 LICENSE) alongside this file.

======================================================================
Resolved architectural questions (hands-on, from the real paper + code,
not guessed - see paper/related_work_model_audit.md and
paper/phase3_baseline_reproduction.md's A7 section for the full write-up)
======================================================================

(a) Generator architecture: an encoder-decoder where BOTH the encoder and
    decoder are plain (unidirectional) 5-layer LSTMs with 200 hidden units
    per layer (main.py: `hidden_size=200, num_layers=5`), operating
    directly on ONE-HOT event vectors (no learned embedding layer at all -
    preparation.py's `__event_to_one_hot` builds these one-hot columns
    directly from the raw log). The decoder's start-of-sequence input is a
    vector of literal 1.0s across every column (network.py's
    `begin_symbol = torch.ones(...)`, not a proper one-hot token) -
    replicated here exactly as found, despite being an unusual choice.
    Each decoder step's raw LSTM output passes through ONE shared Linear
    layer (`fc_out`) producing all output columns at once, followed by an
    element-wise ReLU applied to the WHOLE output vector (network.py's
    `Decoder.forward`: `prediction = self.relu(self.fc_out(output))`) -
    this is a genuine, verified PAPER/CODE DISCREPANCY: the paper's Eq. 4.1
    states the activity distribution uses Softmax while only the duration
    time uses ReLU, but the actual code applies ReLU indiscriminately to
    every output column BEFORE the (softmax-internal) cross-entropy loss is
    computed on those same columns. Per this project's established practice
    (e.g. A3 RLHGNN's max-pooling-vs-"last node" discrepancy), the CODE's
    actual behavior is replicated here, since that is what produced the
    paper's reported numbers - `fc_out` output is ReLU'd before
    `F.cross_entropy` is applied to it as "logits."

(b) Discriminator architecture: also a 5-layer, 200-hidden-unit
    unidirectional LSTM, followed by one Linear(200, 1) layer
    (network.py's `Discriminator`) producing a raw realism score at EVERY
    timestep (not one pooled scalar per sequence) - the adversarial loss
    averages `logsigmoid` over every timestep and every batch element
    (network.py's `train_gan`: `ll1 = -mean(logsigmoid(pr))`, etc.).

(c) Training procedure - NOT a separate pretrain-then-adversarial-fine-tune
    schedule, and NOT REINFORCE/policy-gradient. Confirmed hands-on (both
    the paper's Algorithm 1 and the actual `network.py::train_gan`): the
    generator and discriminator are trained JOINTLY from initialization,
    every mini-batch, in two sub-steps:
      1. Discriminator update: real sequences are the ground-truth suffix
         one-hots, LABEL-SMOOTHED (0.9 on the true class, 0.1/(m-1) spread
         over the rest - paper's Sec. 4.2 / Eq. 4.5's description of the
         "real" side) and passed through a near-discrete Gumbel-softmax
         (fixed tau=0.001, network.py's `one_hot_to_gumble_soft`) as a
         small stochastic perturbation ("instance noise"). Fake sequences
         are the generator's own raw per-step output, passed through
         Gumbel-softmax (Eq. 4.5) at an EXPONENTIALLY ANNEALED temperature
         `tau = max(0.9**epoch, floor)` (network.py: `t = np.power(.9, i)`;
         the `floor` is THIS project's own added numerical-stability
         safeguard - see configs/models/mlmme.yaml's comment). This is the
         genuine mechanism resolving the "how does a GAN train over
         DISCRETE sequences" question: Gumbel-softmax continuous
         relaxation, not REINFORCE, not a hard sampling+policy-gradient
         estimator.
      2. Generator update: minimizes the SUM of (i) the adversarial
         "fool the discriminator" loss (paper's Eq. 4.6, second line) and
         (ii) the standard supervised cross-entropy loss over the full
         suffix (paper's Eq. 4.2/4.4) - exactly Algorithm 1's lines 4-5
         ("update theta_g by minimizing L(G;D) + L_supervised"). The
         reference code computes this as two sequential `.backward()`
         calls without an intervening `optimizer.zero_grad()` (relying on
         PyTorch's default gradient accumulation to sum the two losses'
         gradients before one `optimizer.step()`) - this project computes
         the equivalent sum explicitly (`total_g_loss = adv_loss +
         mle_loss`, one `.backward()`) for clarity; the trained objective
         is unchanged.
      Decoder self-feeding during training uses a stochastic mix
      (`teacher_forcing_ratio=0.1`, network.py/main.py): 10% of steps feed
      the true previous suffix token, 90% feed the decoder's own previous
      CONTINUOUS (ReLU'd, non-discretized) output vector back in - i.e.
      "open-loop" training in the paper's own terminology is a soft/
      differentiable self-feeding loop, not a hard-argmax one. Hard-argmax,
      one-hot closed-loop generation (matching `suffix.py::suffix_generate`,
      the paper's actual Table 2 evaluation protocol, beam size 1) is used
      only at inference time (see `generate()` below) - this project's
      reported val/test DL-similarity uses that same hard-argmax greedy
      protocol, consistent with every other full-suffix model in this
      roster (A4/A5/A6).

(d) z_t for Phase 4: `encode()` returns the ENCODER's final top-layer
    hidden state (batch, hidden_size=200) - the classic seq2seq bottleneck
    vector summarizing the whole prefix, computed once before the decoder
    or either prediction pathway ever runs. Architecturally, this is the
    roster's cleanest "one vector per prefix" case: unlike A4 SuTraN (a
    sequence of per-position encoder states) or A5 CRTP-LSTM (a
    left-padded BiLSTM's last-position state), MLMME's encoder genuinely
    collapses the entire prefix into a single fixed-size context vector by
    architectural design (a real RNN encoder bottleneck, not a
    length-indexed readout convention) - closest in spirit to A3 RLHGNN's/
    A6 LUPIN's single pooled-per-prefix vectors, though via a classical
    seq2seq mechanism rather than graph- or attention-pooling.

======================================================================
Scope decision: activity-only, remaining-time head AND input dropped
======================================================================
Same project-wide rule as A1-A6: this project's common schema
(src/data/schema.py) carries only case_id/activity/timestamp, so every
roster model predicts activities only. The original architecture's
duration-time output column (and its corresponding input column, part of
the same one-hot-plus-scalar event vector `e^(i) = (a^(i), t_i)`) is
dropped entirely - since MLMME (unlike A4 SuTraN's NDA variant) does not
autoregressively re-consume its own timestamp PREDICTIONS as later
timestamp INPUTS (the original's duration-time input at each step is a
precomputed feature straight from the log, not the model's own prior
output), dropping it does not create the "broken inputs" issue A4 had to
work around - it is simply omitted, consistent with A3/A5's equivalent
"drop the numeric time-proxy feature entirely" cuts.

A second, necessary consequence of this scope cut, disclosed explicitly
(not a footnote): dropping the duration-time column means the generator's
per-step output tensor now consists ENTIRELY of the "events" (activity)
columns. In the original repo's own `train_gan`, the generator's fake
sequence for the adversarial loss (`suffix_fake = y_pred`, after replacing
`y_pred[:,:,events]` with a `.detach()`-ed-then-Gumbel-softened copy) is
reused, UN-detached-again, for computing the generator's own adversarial
loss (`pf = rnnD(suffix_fake); gl = ...; gl.backward()`). Because the
`.detach()` call happens on `y_pred[:,:,events].detach()` specifically, in
the ORIGINAL two-headed (activity + time) design, gradient can still leak
back into the shared decoder/encoder via the (non-detached) time column of
the same underlying `y_pred` tensor and `fc_out` weight matrix - a subtle,
partial gradient path. Once the time column is removed (this project's
activity-only scope), replicating the `.detach()` verbatim would leave
NOTHING un-detached, silently making the entire adversarial loss a
no-gradient no-op for the generator - directly contradicting the paper's
stated contribution (adversarial training measurably improves suffix
quality, its Table 5's statistical tests). This project therefore does
NOT replicate that specific `.detach()` call: the generator's own
adversarial-loss computation uses a NON-detached copy of its fake sequence
(gradient flows from the discriminator's realism judgment into the
generator's activity-prediction weights, matching the paper's actual
described mechanism, Eq. 4.6), while the DISCRIMINATOR's own parameter
update still correctly uses a `.detach()`-ed copy (so D's loss never
back-propagates into G's parameters) - i.e., this project keeps the
standard/correct GAN training pattern rather than an artifact of the
original's multi-task output design that this project's own activity-only
scope would otherwise silently defeat.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """5-layer, 200-hidden unidirectional LSTM over one-hot activity
    vectors (network.py's `Encoder`, `dropout=0.3`, `hidden_size=200`,
    `num_layers=5`, taken directly from main.py's construction call, not
    guessed). Uses `pack_padded_sequence` so padding never contaminates the
    final hidden state - a correctness improvement over the original
    repo's own same-length-bucket batching trick (`preparation.py`'s
    `__log_partition`, not replicated here), not an architectural change."""

    def __init__(self, num_classes: int, hidden_size: int = 200, num_layers: int = 5, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=num_classes, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )

    def forward(self, x_onehot: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            x_onehot, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, c) = self.lstm(packed)
        return h, c  # each (num_layers, batch, hidden_size)


class Decoder(nn.Module):
    """5-layer, 200-hidden unidirectional LSTM + one shared Linear producing
    every output column, followed by an indiscriminate ReLU (network.py's
    `Decoder.forward` - see model.py's docstring section (a) for the
    verified paper/code discrepancy this replicates: the paper's Eq. 4.1
    describes Softmax for activities, the actual code applies ReLU to the
    whole output vector before it is used as cross-entropy "logits")."""

    def __init__(self, num_classes: int, hidden_size: int = 200, num_layers: int = 5, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_classes, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, step_input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        output, (hidden, cell) = self.lstm(step_input, (hidden, cell))
        prediction = self.relu(self.fc_out(output))  # (batch, 1, num_classes) - ReLU'd, see docstring
        return prediction, hidden, cell


class Discriminator(nn.Module):
    """5-layer, 200-hidden unidirectional LSTM + Linear(hidden, 1),
    producing one raw realism score PER TIMESTEP (network.py's
    `Discriminator` - not a single pooled scalar per sequence)."""

    def __init__(self, num_classes: int, hidden_size: int = 200, num_layers: int = 5, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_classes, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(seq)
        return self.fc_out(output)  # (batch, seq_len, 1)


class MLMMEGenerator(nn.Module):
    """The encoder-decoder ("generator" in the GAN sense) - activity-only,
    remaining-time head dropped (see model.py's module docstring)."""

    def __init__(self, num_classes: int, hidden_size: int = 200, num_layers: int = 5, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(num_classes, hidden_size, num_layers, dropout)
        self.decoder = Decoder(num_classes, hidden_size, num_layers, dropout)

    def encode(self, prefix_onehot: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
        """z_t for Phase 4 - see model.py's docstring section (d): the
        encoder's final TOP-LAYER hidden state, one vector per prefix."""
        h, _ = self.encoder(prefix_onehot, prefix_lengths)
        return h[-1]  # (batch, hidden_size)

    def forward(
        self,
        prefix_onehot: torch.Tensor,
        prefix_lengths: torch.Tensor,
        suffix_onehot: torch.Tensor,
        teacher_forcing_ratio: float = 0.1,
    ) -> torch.Tensor:
        """Teacher-forced / soft-self-feeding training forward pass
        (network.py's `Seq2Seq.forward`): returns per-step predictions,
        shape (batch, window_size, num_classes), ReLU'd raw "logits" (see
        Decoder's docstring). `suffix_onehot` is the target suffix itself
        (aligned so that decoder step i predicts `suffix_onehot[:, i, :]`);
        the decoder's start-of-sequence input is a vector of literal 1.0s
        (see module docstring (a)), and - with probability
        `teacher_forcing_ratio` at each step - the NEXT step's input is the
        true target at the CURRENT step (`suffix_onehot[:, i, :]`);
        otherwise it is the decoder's own previous (continuous, non-
        discretized) output, exactly matching the original's "open-loop"
        self-feeding mechanism (see module docstring (c))."""
        batch_size, window_size = suffix_onehot.size(0), suffix_onehot.size(1)
        hidden, cell = self.encoder(prefix_onehot, prefix_lengths)
        inp = torch.ones(batch_size, 1, self.num_classes, device=suffix_onehot.device)

        outputs = []
        for i in range(window_size):
            out, hidden, cell = self.decoder(inp, hidden, cell)
            outputs.append(out)
            teacher_force = random.random() < teacher_forcing_ratio
            inp = suffix_onehot[:, i : i + 1, :] if teacher_force else out
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def generate(self, prefix_onehot: torch.Tensor, prefix_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Hard-argmax, one-hot closed-loop greedy generation (beam size 1),
        matching the original repo's ACTUAL inference-time mechanism
        (`suffix.py::suffix_generate`'s per-step `F.one_hot(argmax(softmax(
        ...)))` self-feeding - not the soft/continuous feeding used during
        `forward()`'s training loop, see module docstring (c)). This is the
        protocol that produced the paper's reported Table 2 SDL numbers
        (beam size 1), and the protocol this project uses for its own
        reported val/test DL-similarity, consistent with A4/A5/A6."""
        self.eval()
        batch_size = prefix_onehot.size(0)
        hidden, cell = self.encoder(prefix_onehot, prefix_lengths)
        inp = torch.ones(batch_size, 1, self.num_classes, device=prefix_onehot.device)

        pred_classes = []
        for _ in range(max_len):
            out, hidden, cell = self.decoder(inp, hidden, cell)
            pred_class = out.argmax(dim=-1)  # (batch, 1)
            pred_classes.append(pred_class)
            inp = F.one_hot(pred_class, num_classes=self.num_classes).float()
        return torch.cat(pred_classes, dim=1)  # (batch, max_len)


def label_smooth_gumbel(onehot: torch.Tensor, tau: float = 0.001) -> torch.Tensor:
    """The "real"/target-side continuous relaxation used by the
    discriminator (network.py's `one_hot_to_gumble_soft`): label-smooth the
    one-hot vector (0.9 on the true class, 0.1/(m-1) spread over the rest -
    paper's Sec. 4.2 description of Eq. 4.5's real-instance construction),
    then apply a near-discrete Gumbel-softmax (fixed tau=0.001) as a small
    stochastic perturbation ("instance noise"), matching the original
    exactly (including feeding the smoothed values into `gumbel_softmax` AS
    IF they were raw logits, which is what the reference code literally
    does)."""
    num_classes = onehot.size(-1)
    smoothed = torch.where(
        onehot > 0.5,
        torch.full_like(onehot, 0.9),
        torch.full_like(onehot, 0.1 / (num_classes - 1)),
    )
    return F.gumbel_softmax(smoothed, tau=tau, dim=-1)


def anneal_gumbel_temperature(epoch: int, base: float = 0.9, floor: float = 1e-3) -> float:
    """The generator's OWN annealed Gumbel-softmax temperature (paper's
    Sec. 4.2 / Eq. 4.5: "exponentially anneal the temperature tau ... from
    0.9 to 0"; network.py's `t = np.power(.9, i)` where `i` is the epoch
    index). `floor` is THIS project's own added numerical-stability
    safeguard, not present in the original code: `0.9**epoch` underflows
    towards machine-epsilon well before a 500-epoch budget completes, and
    `gumbel_softmax` at tau=0 divides by zero - disclosed in
    configs/models/mlmme.yaml's comments."""
    return max(base**epoch, floor)


def discriminator_adversarial_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Paper's Eq. 4.6, first line; network.py's `train_gan`:
    `ll1 = -mean(logsigmoid(pr)); ll2 = -mean(logsigmoid(1-pf))`."""
    return -torch.mean(F.logsigmoid(real_scores)) - torch.mean(F.logsigmoid(1.0 - fake_scores))


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Paper's Eq. 4.6, second line; network.py's `train_gan`:
    `gl = -mean(logsigmoid(pf) - logsigmoid(1-pf))`."""
    return -torch.mean(F.logsigmoid(fake_scores) - F.logsigmoid(1.0 - fake_scores))
