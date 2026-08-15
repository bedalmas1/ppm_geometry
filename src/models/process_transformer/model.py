"""ProcessTransformer (A1) model architecture.

Vendored and lightly adapted (Keras 3 compatibility only - no architectural
changes) from Bukhsh, Saeed & Dijkman, "ProcessTransformer: Predictive
Business Process Monitoring with Transformer Network" (arXiv:2104.00721),
https://github.com/Zaharah/processtransformer (Apache-2.0). See
paper/related_work_model_audit.md Section A for the full audit.

This file intentionally does NOT include a z_t extraction hook - Phase 4
(representation extraction) builds a new functional model re-using this
trained model's `global_average_pooling1d` layer output, without needing to
touch this file. See paper/related_work_model_audit.md's note on the
GlobalAveragePooling1D output as the z_t extraction point.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layernorm_b(out_a + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def get_next_activity_model(
    max_case_length, vocab_size, output_dim, embed_dim=36, num_heads=4, ff_dim=64
):
    inputs = layers.Input(shape=(max_case_length,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D(name="prefix_representation")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="next_activity_transformer")
