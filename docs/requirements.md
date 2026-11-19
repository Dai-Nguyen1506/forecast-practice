# Task 3: Dự đoán chuỗi từ hình ảnh (Image-to-Sequence Prediction)
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

# Tóm tắt công thức các chỉ số đánh giá (Evaluation Metrics)

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