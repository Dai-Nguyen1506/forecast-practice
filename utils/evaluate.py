import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data import MAX_SEQ_LEN, START_TOKEN, decode_tokens


def predict_sequences(model, X_test, batch_size=256):
    """Sinh dự đoán bằng Greedy Decoding và trả về mảng số."""
    preds_num = []
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i : i + batch_size]
        dec_in = np.zeros((len(X_batch), MAX_SEQ_LEN), dtype=np.int32)
        dec_in[:, 0] = START_TOKEN

        for t in range(MAX_SEQ_LEN - 1):
            preds = model([X_batch, dec_in], training=False).numpy()
            dec_in[:, t + 1] = np.argmax(preds[:, t, :], axis=-1)

        preds_num.extend(decode_tokens(seq) for seq in dec_in[:, 1:])
    return np.asarray(preds_num, dtype=np.float64)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Đánh giá bằng R2, NSE, MAE, RMSE."""
    y_true_num = np.asarray(y_true, dtype=np.float64)
    y_pred_num = np.asarray(y_pred, dtype=np.float64)

    mae     = mean_absolute_error(y_true_num, y_pred_num)
    rmse    = np.sqrt(mean_squared_error(y_true_num, y_pred_num))
    r2      = r2_score(y_true_num, y_pred_num)
    denom   = np.sum((y_true_num - np.mean(y_true_num)) ** 2)
    nse     = 1.0 - np.sum((y_true_num - y_pred_num) ** 2) / denom if denom else np.nan

    print(f"=== {model_name} ===")
    print(f"R2: {r2:.4f} | NSE: {nse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return {"model": model_name, "R2": r2, "NSE": nse, "MAE": mae, "RMSE": rmse}
