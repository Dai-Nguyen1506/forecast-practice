from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Permute,
    Reshape,
    Embedding,
    LSTM,
    AdditiveAttention,
    Concatenate,
    Dense,
)


def build_dai_model(img_shape, vocab_size, max_seq_len):
    """Mô hình mẫu CNN Encoder + LSTM Decoder + Attention."""
    # --- Encoder (CNN + LSTM) ---
    img_input = Input(shape=img_shape, name="image_input")  # (28, 224, 1)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    x = MaxPooling2D((2, 2))(x)                             # (14, 112, 32)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)                             # (7, 56, 64)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((7, 2))(x)                             # (1, 28, 128)

    x = Permute((2, 1, 3))(x)                               # (28, 1, 128)
    x = Reshape((28, 128))(x)                               # 28 bước thời gian từ trái sang phải
    enc_out, state_h, state_c = LSTM(128, return_sequences=True, return_state=True)(x)

    # --- Decoder (LSTM + Attention) ---
    dec_input = Input(shape=(max_seq_len,), name="decoder_input")
    dec_emb = Embedding(vocab_size, 64)(dec_input)
    dec_out = LSTM(128, return_sequences=True)(dec_emb, initial_state=[state_h, state_c])

    context = AdditiveAttention()([dec_out, enc_out])
    combined = Concatenate()([dec_out, context])
    outputs = Dense(vocab_size, activation="softmax")(combined)

    return Model(inputs=[img_input, dec_input], outputs=outputs)
