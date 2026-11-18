# TỔNG HỢP & GIẢI THÍCH YÊU CẦU ĐỒ ÁN CUỐI KỲ (FINAL PROJECT 2026)

## 1. Quy định chung
* **Hình thức làm việc:** Làm việc theo nhóm, mỗi nhóm gồm **4 sinh viên**.
* **Yêu cầu dữ liệu:** Mỗi nhóm phải tự chọn bộ dữ liệu riêng biệt, **không được trùng lặp** với các nhóm khác trong lớp.
* **Khối lượng công việc:** Mỗi nhóm cần hoàn thành đầy đủ **3 nhiệm vụ (Tasks)** dưới đây.

---

## 2. Chi tiết các nhiệm vụ (Tasks)

### Task 1: Dự báo một thời điểm duy nhất trong tương lai (Single-step Ahead Forecasting)
* **Dữ liệu & Biến mục tiêu:** 
  * Chọn một bộ dữ liệu chuỗi thời gian (Time Series).
  * Chọn 1 thuộc tính làm biến mục tiêu cần dự đoán (ký hiệu là `pred_column`).
* **Yêu cầu mô hình:** 
  * Xây dựng mạng nơ-ron hồi quy (**RNN**, có thể dùng biến thể như LSTM, GRU) để dự đoán giá trị của `pred_column` ở **2 bước thời gian tiếp theo** ($t + 2$) so với thời điểm hiện tại ($t$).
  * *Ví dụ:* Nếu dữ liệu ghi nhận theo từng giờ và hiện tại là 10:00 AM, mô hình cần dự đoán giá trị tại lúc 12:00 PM.
  * Tích hợp thêm **cơ chế Attention** vào mô hình ở vị trí phù hợp.
* **Chỉ số đánh giá:** $R^2$, NSE (Nash-Sutcliffe Efficiency), MAE, và RMSE.

---

### Task 2: Dự báo nhiều thời điểm liên tiếp (Multi-step / Sequence-to-Sequence Forecasting)
* **Dữ liệu:** Sử dụng **chính bộ dữ liệu** và biến mục tiêu `pred_column` đã chọn ở Task 1.
* **Yêu cầu mô hình:** 
  * Xây dựng kiến trúc **Encoder-Decoder (Seq2Seq)** để dự đoán giá trị của `pred_column` cho **5 bước thời gian liên tiếp tiếp theo** tính từ thời điểm hiện tại (từ $t+1$ đến $t+5$).
  * Tích hợp thêm **cơ chế Attention** (ví dụ: Bahdanau Attention hoặc Luong Attention) vào kiến trúc Encoder-Decoder.
* **Chỉ số đánh giá:** $R^2$, NSE, MAE, và RMSE. 
  * *Lưu ý:* Nhóm cần tự nghiên cứu cách tính và áp dụng các chỉ số này cho bài toán dự báo chuỗi nhiều bước (ví dụ: tính riêng cho từng bước thời gian $t+1 \dots t+5$ hoặc tính trung bình trên toàn bộ chuỗi dự đoán).

---

### Task 3: Dự đoán chuỗi từ hình ảnh (Image-to-Sequence Prediction)
* **Dữ liệu:** Chọn một bộ dữ liệu hình ảnh mà mỗi mẫu gồm **1 ảnh đầu vào** và **1 chuỗi đầu ra** tương ứng. Đề bài gợi ý 3 chủ đề chính:
  1. **Nhận diện chuỗi chữ số viết tay (Handwritten Digit Sequence Recognition):** Đầu vào là ảnh chứa nhiều chữ số viết tay $\rightarrow$ Đầu ra là dãy số tương ứng.
  2. **Đọc chỉ số công tơ điện (Electricity Meter Reading):** Đầu vào là ảnh chụp đồng hồ điện $\rightarrow$ Đầu ra là dãy số biểu thị chỉ số tiêu thụ điện.
  3. **Nhận diện biển số xe (License Plate Recognition):** Đầu vào là ảnh biển số xe $\rightarrow$ Đầu ra là chuỗi chữ cái và chữ số trên biển số.
* **Yêu cầu mô hình:** 
  * Xây dựng mô hình Học sâu kết hợp **CNN Encoder** (để trích xuất đặc trưng từ ảnh) với **RNN/LSTM Decoder** (để sinh ra chuỗi ký tự/số đầu ra).
  * Có thể tích hợp thêm **cơ chế Attention** nếu phù hợp.
* **Chỉ số đánh giá:** $R^2$, NSE, MAE, và RMSE.
  * *Ghi chú quan trọng cho nhóm:* Các chỉ số này vốn dành cho bài toán hồi quy (dữ liệu số liên tục). Do đó:
    * Nếu nhóm chọn bài toán xuất ra giá trị số (như *Đọc chỉ số công tơ điện* hoặc *Chuỗi chữ số viết tay* ghép thành một con số), nhóm có thể chuyển chuỗi dự đoán thành kiểu số (numeric) để tính trực tiếp $R^2$, NSE, MAE, RMSE.
    * Nếu chọn bài toán có chứa chữ cái (như *Biển số xe*), nhóm nên trình bày rõ cách quy đổi/đánh giá (hoặc bổ sung thêm các chỉ số đặc thù của bài toán nhận diện chuỗi như *Sequence Accuracy*, *Character Error Rate - CER*) trong báo cáo để đảm bảo tính chặt chẽ.

---

## 3. Tóm tắt công thức các chỉ số đánh giá (Evaluation Metrics)

Giả sử $y_i$ là giá trị thực tế, $\hat{y}_i$ là giá trị mô hình dự đoán, $\bar{y}$ là giá trị trung bình của tập dữ liệu thực tế, và $N$ là tổng số mẫu:

1. **MAE (Mean Absolute Error):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

2. **RMSE (Root Mean Squared Error):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

3. **$R^2$ (Coefficient of Determination):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

4. **NSE (Nash-Sutcliffe Efficiency):**
   $$\text{NSE} = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
   *(Lưu ý: Trong thống kê hồi quy cơ bản, công thức tính NSE giống hệt $R^2$ khi $R^2$ được tính theo tổng bình phương sai số so với giá trị trung bình, tuy nhiên NSE thường được dùng phổ biến trong các bài toán chuỗi thời gian thủy văn/môi trường).*