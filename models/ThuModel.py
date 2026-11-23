from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Permute, Reshape, Dense,
                                     Embedding, LSTM, AdditiveAttention, Concatenate,
                                     BatchNormalization, Dropout)



def build_thu_model_1(IMG_SHAPE, MAX_OUTPUT_LENGTH, VOCAB_SIZE):
    units = 256

    # Encoder (CNN + LSTM)
    img_input = Input(shape=IMG_SHAPE)          # (28, 224, 1)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)    # (28, 224, 32)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)                         # (14, 112, 32)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)            # (14, 112, 64)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)                         # (7, 56, 64)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)           # (7, 56, 128)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)                         # (3, 28, 128)

    # Chuyển feature map thành chuỗi theo chiều ngang
    # feature map (H', W', C) -> chuỗi W' vector, mỗi vector mô tả một "cột" của ảnh
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = Permute((2, 1, 3))(x)
    x = Reshape((w, h * c))(x)                          # (28, 3*128) = (28, 384)  

    enc_out, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(x)

    # Decoder (LSTM + Additive Attention)
    dec_input = Input(shape=(MAX_OUTPUT_LENGTH,))
    emb = Embedding(VOCAB_SIZE, 64)(dec_input)
    dec_out = LSTM(units, return_sequences=True)(emb, initial_state=[state_h, state_c])

    context = AdditiveAttention()([dec_out, enc_out])
    merged = Concatenate()([dec_out, context])
    z = Dense(256, activation='relu')(merged)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(z)

    model = Model(inputs=[img_input, dec_input], outputs=outputs)
    return model