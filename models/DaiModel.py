from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Reshape,
    Bidirectional,
    Embedding,
    GRU,
    AdditiveAttention,
    Concatenate,
    Dense,
)


def build_dai_model(img_shape, vocab_size, max_seq_len):
    """CNN + Bidirectional GRU Encoder -> GRU Decoder + Bahdanau Attention."""
    # --- 1. Encoder (Conv2D + BatchNorm + MaxPool -> Bidirectional GRU) ---
    img_input = Input(shape=img_shape, name="image_input")  # (28, 224, 1)

    x = Conv2D(16, (3, 3), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=1, padding="same")(x)    # (28, 224, 16)

    x = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)                               # (14, 112, 32)

    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)                               # (7, 56, 64)

    x = Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((7, 2))(x)                               # (1, 28, 128)

    x = Reshape((28, 128))(x)                                 # (28 timesteps, 128 features)
    enc_out = Bidirectional(GRU(64, return_sequences=True))(x)  # (28, 128)

    # --- 2. GRU Decoder + Attention ---
    dec_input = Input(shape=(max_seq_len,), name="decoder_input")
    dec_emb = Embedding(vocab_size, 32)(dec_input)
    dec_out = GRU(128, return_sequences=True)(dec_emb)

    context = AdditiveAttention()([dec_out, enc_out])
    combined = Concatenate()([dec_out, context])
    outputs = Dense(vocab_size, activation="softmax")(combined)

    return Model(inputs=[img_input, dec_input], outputs=outputs, name="Dai_CRNN_Seq2Seq")
