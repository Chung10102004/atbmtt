# Phương pháp thực nghiệm đặc trưng AML

## Bài toán và đơn vị dự đoán

Một mẫu là một giao dịch `(thời gian, tài khoản gửi, tài khoản nhận, số tiền, tiền tệ, hình thức thanh toán)`. Nhãn là `Is Laundering` của bộ mô phỏng IBM, không phải kết luận điều tra thực tế. XGBoost trả điểm nghi ngờ để xếp hạng giao dịch và lập danh sách cảnh báo.

Tại thời điểm `t`, mô hình có thuộc tính của giao dịch đang xét và lịch sử không nhãn `H(t) = {giao dịch j: t_j < t}`. Dữ liệu chỉ chính xác đến phút, vì vậy tất cả giao dịch cùng phút dùng chung ảnh chụp lịch sử. ID tài khoản ghép với ID ngân hàng chỉ dùng tra cứu, không đưa trực tiếp vào mô hình.

## Bốn nhóm đặc trưng

| Nhóm | Số cột | Nội dung |
|---|---:|---|
| Giao dịch | 11 | Log số tiền gửi/nhận, giờ, thứ, hình thức, tiền tệ, cùng ngân hàng, tự chuyển, đổi tiền, số tiền tròn |
| Lịch sử | 17 | 8 thống kê mỗi tài khoản gửi/nhận; thêm số giao dịch cặp gửi→nhận trước đó |
| Hoạt động | 20 | Mỗi tài khoản: 4 số đếm theo hướng/cửa sổ, 4 tổng tiền theo hướng/cửa sổ, 2 độ lệch số tiền so với lịch sử cùng tiền tệ |
| Quan hệ | 15 | 8 số đối tác duy nhất theo hướng/cửa sổ, lịch sử cặp đảo chiều, tuổi/độ gần quan hệ, 2 số đếm cặp theo cửa sổ, 2 tỷ trọng đối tác trong 24h |

Cửa sổ `W` là `[t-W,t)` với `W∈{1 giờ,24 giờ}`; mốc trái được tính, mốc phải không tính. Giao dịch tự chuyển tạo hai sự kiện vào/ra, chỉ cập nhật sau khi mọi giao dịch cùng thời điểm được truy vấn.

**Tiền tệ:** tổng tiền mới được phân theo tài khoản và tiền tệ. Không cộng USD với EUR khi chưa có tỷ giá. Bên gửi dùng Amount Paid/Payment Currency; bên nhận dùng Amount Received/Receiving Currency. Tổng sau đó lấy `log(1+x)`. Dùng cộng bù sai số và đặt tổng rỗng về đúng0 để tránh dư số học khi các khoản tiền chênh lệch rất lớn.

**Độ lệch số tiền:** `log(1+số tiền hiện tại) − mean(log(1+số tiền lịch sử cùng tiền tệ/hướng))`. Đây không phải z-score. Các mean-log-amount thuộc nhóm kiểm soát lịch sử giữ cách gộp tiền tệ của pipeline trước, không diễn giải chúng là số dư hay trung bình tiền đã quy đổi.

**Quan hệ:** đối tác duy nhất được đếm riêng hướng vào/ra. Tuổi quan hệ là khoảng cách từ giao dịch đầu tiên trước đó của đúng cặp gửi→nhận; độ gần là khoảng cách từ giao dịch gần nhất. Tỷ trọng đối tác là số giao dịch cặp trong 24h chia số lượt gửi của bên gửi hoặc lượt nhận của bên nhận trong 24h. Đây là thống kê cục bộ, chưa phải phát hiện vòng/chuỗi nhiều bước.

Không có lịch sử: số đếm/tổng bằng0; mean, tuổi và khoảng cách chưa xác định giữ NaN để XGBoost xử lý. Nhãn không tham gia tính đặc trưng.

## So sánh có kiểm soát

F0 = giao dịch; F1 = F0+lịch sử; F2 = F1+hoạt động; F3 = F1+quan hệ; F4 = F1+hoạt động+quan hệ. Chạy tất cả với seed 0/1/2, cùng cấu hình XGBoost, cùng chia thời gian 60/20/20, không thêm mẫu hoặc tìm siêu tham số. Early stopping dùng validation; ngưỡng F1 chọn trên validation rồi giữ nguyên khi đánh giá test.

AP là chỉ số chính. Precision@1000/Recall@1000 mô tả1000 cảnh báo có điểm cao nhất trên toàn test, không phải1000 cảnh báo mỗi ngày. F1 bổ sung góc nhìn theo ngưỡng. Accuracy không phải tiêu chí chính vì lớp rửa tiền hiếm.

Các hiệu số F2−F1/F3−F1 đo lợi ích khi đã có lịch sử; F4−F2/F4−F3 kiểm tra đóng góp bổ sung của nhóm còn lại. Báo cáo cả trường hợp không tăng; không chọn cấu hình rồi chỉ công bố điểm tốt nhất. Độ lệch chuẩn của3seed thể hiện biến động huấn luyện, không phải khoảng tin cậy trên quần thể giao dịch độc lập.

## Chẩn đoán và giới hạn

Test HI-Small đã được dùng trong thí nghiệm augmentation trước. Nghiên cứu này là thăm dò; không gọi đó là holdout xác nhận mới. Lịch sử theo timestamp nghiêm ngặt khác giả định xử lý theo thứ tự dòng trong cùng phút của baseline v3 cũ.

Sau khi định trước thí nghiệm, thống kê theo ngày phát hiện lưu lượng giảm mạnh và tỷ lệ dương tăng từ11/09. Phân tích tách giai đoạn là chẩn đoán bổ sung, giữ nguyên mọi mô hình và phép đánh giá chính. Không so sánh AP giữa hai giai đoạn có tỷ lệ nhãn khác nhau để khẳng định độ khó; chỉ so sánh các cấu hình trong cùng cohort.

Thời gian dựng toàn bảng đặc trưng không chứng minh độ trễ xử lý trực tuyến. Importance của cây không chứng minh nhân quả. Hiệu quả trên một bộ dữ liệu mô phỏng không tự chứng minh hiệu quả với dữ liệu ngân hàng thật.

Nghiên cứu nền: [Graph Feature Preprocessor, ICAIF 2024](https://research.ibm.com/publications/graph-feature-preprocessor-real-time-subgraph-based-feature-extraction-for-financial-crime-detection). Đóng góp của đồ án là thực nghiệm có kiểm soát và phân tích, không tuyên bố phát minh hướng khai thác đặc trưng đồ thị cho AML.
