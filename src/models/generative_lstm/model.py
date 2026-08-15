"""GenerativeLSTM (A2) model architecture - next-activity path only.

Adapted from Camargo, Dumas & Gonzalez-Rojas, "Learning Accurate LSTM Models
of Business Processes" (BPM 2019), https://github.com/AdaptiveBProcess/GenerativeLSTM
(Apache-2.0). See paper/related_work_model_audit.md Section A and
paper/phase3_baseline_reproduction.md for the full audit/reproduction notes.

The original repo's `model_shared_cat` architecture is multi-task: three
input branches (activity, role, time) feeding two stacked LSTM layers each,
predicting next-activity, next-role, AND next-time/remaining-time jointly,
with activity/role embeddings PRETRAINED separately (embedding_training.py,
a word2vec-style skip-gram phase) before the main model is trained, and its
exact hyperparameters chosen via a Bayesian search over model_type in
{shared_cat, concatenated} (dg_training.py).

This project implements only the **activity branch** of that architecture,
consistent with how A1 (ProcessTransformer) was scoped to next-event
prediction only (spec's Family A next-event objective, not the paper's full
multi-task setup):
  - No role branch: this project's common event-log schema (src/data/schema.py)
    deliberately carries only case_id/activity/timestamp, not resource/role,
    to keep every model's input data identical and comparable (spec Sec.6) -
    adding a role branch just for this one model would need resource data no
    other roster model uses.
  - No time-regression branch/head: matches A1's scope decision (next-event
    only, not next-time/remaining-time).
  - Trainable (not separately-pretrained/frozen) activity embeddings: avoids
    reproducing the word2vec-style pretraining stage, which is a substantial
    separate undertaking not central to this project's geometry study.
  - Fixed hyperparameters instead of the repo's Bayesian search over them:
    l_size=100, embed_dim=10 (both from the repo's own dg_training.py search
    space {50,100} and {5,10,15} respectively - middle/upper values, not
    guessed), dropout=0.2 (hardcoded in the original, not searched),
    optimizer=Nadam(lr=0.002) (the original's own hardcoded Nadam config).

What IS kept faithful: the core "embed -> LSTM(return_sequences=True) ->
BatchNormalization -> LSTM(return_sequences=False) -> Dense(softmax)"
structure, exactly matching `model_shared_cat.py`'s activity-prediction path
(layers l1_c1, batch1, l2_c1, act_output).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def get_next_activity_model(
    max_case_length, vocab_size, output_dim, embed_dim=10, lstm_size=100, dropout=0.2
):
    inputs = layers.Input(shape=(max_case_length,), name="ac_input")
    x = layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, input_length=max_case_length, name="ac_embedding"
    )(inputs)
    x = layers.LSTM(
        lstm_size, kernel_initializer="glorot_uniform", return_sequences=True, dropout=dropout
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(
        lstm_size,
        kernel_initializer="glorot_uniform",
        return_sequences=False,
        dropout=dropout,
        name="prefix_representation",
    )(x)
    outputs = layers.Dense(output_dim, activation="softmax", kernel_initializer="glorot_uniform")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="generative_lstm_next_activity")
