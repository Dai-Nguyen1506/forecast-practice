import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_SHAPE = (28, 224, 1)  # (hàng, cột, channel)
PAD_TOKEN, START_TOKEN, END_TOKEN = 0, 1, 2
VOCAB_SIZE = 13
MAX_SEQ_LEN = 9           # tối đa 8 chữ số + 1 token


def encode_labels(labels):
    """Mã hóa nhãn số thành (decoder_input, decoder_target)."""
    n = len(labels)
    dec_in = np.zeros((n, MAX_SEQ_LEN), dtype=np.int32)
    dec_out = np.zeros((n, MAX_SEQ_LEN), dtype=np.int32)

    for i, val in enumerate(labels):
        digits = [int(c) + 3 for c in str(int(val))]
        dec_in[i, : len(digits) + 1] = [START_TOKEN] + digits
        dec_out[i, : len(digits) + 1] = digits + [END_TOKEN]

    return dec_in, dec_out


def decode_tokens(tokens):
    """Chuyển chuỗi token dự đoán về số (dừng khi gặp <END>)."""
    digits = []
    for t in tokens:
        if t == END_TOKEN:
            break
        if t >= 3:
            digits.append(str(t - 3))
    return float("".join(digits)) if digits else 0.0


def load_data(csv_path="../datasets/labels.csv", npy_path="../datasets/images_gray.npy"):
    """Đọc dữ liệu từ .npy và chia cố định Train / Val / Test (70 / 15 / 15)."""
    labels = pd.read_csv(csv_path)["label"].values
    images = np.load(npy_path)

    idx_train, idx_vt = train_test_split(np.arange(len(labels)), test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_vt, test_size=0.5, random_state=42)

    X_train = images[idx_train].astype(np.float32) / 255.0
    X_val   = images[idx_val].astype(np.float32) / 255.0
    X_test  = images[idx_test].astype(np.float32) / 255.0

    train_in, train_out = encode_labels(labels[idx_train])
    val_in, val_out     = encode_labels(labels[idx_val])
    y_test              = labels[idx_test]

    return (X_train, train_in, train_out), (X_val, val_in, val_out), (X_test, y_test)
