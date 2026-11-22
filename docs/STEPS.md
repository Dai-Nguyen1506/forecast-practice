## Bước 1: Download data từ kaggle

1. Tạo folder `datasets/` trong repo
2. Tải Dataset bằng cách chạy file `kaggle_download.py`
3. Chạy `utils/save_data_array.py` để lưu dataset dưới dạng array (kết quả là 2 file `.npy` trong `datasets/`).

## Bước 2: Thiết kế và thử nghiệm model

Mỗi người có thể thử nghiệm nhiều kiến trúc Model khác nhau. Với mỗi kiến trúc, cần ghi lại kiến trúc model ra `models/` vaf kết quả csv ra `output/`.

**THEO FILE MẪU `BASE_MODEL.IPYNB` NÀY, PHẦN PREDICTION LÀ DỰ ĐOÁN THEO KIỂU AUTOREGRESSIVE, NẾU PREDICT TOÀN BỘ MỘT LẦN NHƯ BAN ĐẦU CỦA HUY  THÌ SẼ KHÁC, CÓ THỂ SỬA TÙY Ý**

1. **Clone và đổi tên file từ Base_Model:** Mở Terminal tại thư mục gốc forecast-practice, chạy:

```
# Nhớ đổi tên file ipynb vào đây
cp notebooks/Base_Model.ipynb notebooks/<Name>_Model.ipynb
```

2. Để ý những phần đã note cần thay tên trong Notebook
3. Phần `4. Train Model` thì thay các kiến trúc model khác nhau. Sau đó copy vào các hàm trong `models/<Name>Model.py` như của Đại. Sau này chỉ cần mở ra đổi chiếu
4. Sau khi chạy hết file notebook, kết quả sẽ được lưu ra file csv trong `output/`. Sau chỉ cần mở lên đối chiếu
