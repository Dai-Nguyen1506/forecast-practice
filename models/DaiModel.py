from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Permute,
    Reshape,
    Bidirectional,
    Embedding,
    LSTM,
    GRU,
    AdditiveAttention,
    Concatenate,
    Dense,
)


def build_model_1(img_shape, vocab_size, max_seq_len):
    # Decoder
    img_input = Input(shape=img_shape, name="image_input")

    x = Conv2D(16, (3, 3), padding="same", use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=1, padding="same")(x)

    x = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((7, 2))(x)

    x = Reshape((28, 128))(x)
    enc_out = Bidirectional(GRU(64, return_sequences=True))(x)

    # Encoder
    dec_input = Input(shape=(max_seq_len,), name="decoder_input")
    dec_emb = Embedding(vocab_size, 32)(dec_input)
    dec_out = GRU(128, return_sequences=True)(dec_emb)

    # Attention mechanism
    context = AdditiveAttention()([dec_out, enc_out])
    combined = Concatenate()([dec_out, context])

    # Output layer
    outputs = Dense(vocab_size, activation="softmax")(combined)

    return Model(inputs=[img_input, dec_input], outputs=outputs, name="Dai_CRNN_Seq2Seq")

def build_model_2(img_shape, vocab_size, max_seq_len):
    enc_inputs = Input(shape=img_shape)

    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same', use_bias=False)(enc_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=1, padding="same")(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Reshape((-1, 128))(x)
    enc_out, f_h, f_c, b_h, b_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(x)
    state_h = Concatenate()([f_h, b_h])
    state_c = Concatenate()([f_c, b_c])

    # Decoder
    dec_input = Input(shape=(max_seq_len,), name='decoder_input')
    x_dec = Embedding(vocab_size, 128)(dec_input)
    dec_out = LSTM(128, return_sequences=True)(x_dec, initial_state=[state_h, state_c])

    # Attention mechanism
    context  = AdditiveAttention()([dec_out, enc_out])
    combined = Concatenate()([context, dec_out])

    # Output layer
    output = Dense(vocab_size, activation='softmax')(combined)

    return Model(inputs=[enc_inputs, dec_input], outputs=output)

def build_model_3(img_shape, vocab_size, max_seq_len):
    # Encoder
    enc_inputs = Input(shape=img_shape, name='encoder_input')
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(enc_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)                             # (14, 112, 32)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)                             # (7, 56, 64)

    x = Permute((2, 1, 3))(x)                               # (56, 7, 64)
    x = Reshape((56, 7 * 64))(x)                            # (56 timesteps, 448 features)
    enc_out, f_h, b_h = Bidirectional(GRU(64, return_sequences=True, return_state=True))(x)
    enc_state = Concatenate()([f_h, b_h])

    # Decoder
    dec_input = Input(shape=(max_seq_len,), name='decoder_input')
    dec_emb = Embedding(vocab_size, 128)(dec_input)
    dec_out = GRU(128, return_sequences=True)(dec_emb, initial_state=enc_state)

    # Attention mechanism
    context  = AdditiveAttention()([dec_out, enc_out])
    combined = Concatenate()([context, dec_out])

    # Output layer
    output = Dense(vocab_size, activation='softmax')(combined)

    return Model(inputs=[enc_inputs, dec_input], outputs=output)

