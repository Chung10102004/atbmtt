# Kịch bản thuyết trình: Phát hiện rửa tiền bằng machine learning

Kịch bản cho bộ slide [slide/AML_thuyet_trinh.pptx](../slide/AML_thuyet_trinh.pptx). Cùng nội dung đã được ghi vào
speaker notes của từng slide (bật Presenter View để xem khi nói). Câu hỏi có thể gặp và gợi ý trả lời nằm ở
[thuyet-trinh-va-cau-hoi.md](thuyet-trinh-va-cau-hoi.md).

## 1. Mục đích của bài thuyết trình

Bài thuyết trình cần làm hội đồng tin bốn điều:

1. **Nhóm hiểu bài toán dưới góc nhìn an toàn thông tin.** Giám sát giao dịch là một hệ thống phát hiện, giống IDS:
   xếp hạng một luồng sự kiện lớn, mỗi cảnh báo sai tốn thời gian chuyên viên.
2. **Thí nghiệm được thiết kế đúng.** Chia dữ liệu theo thời gian, không nhìn trước tương lai, cùng một model và cùng
   cách đánh giá cho mọi cấu hình, nên khác biệt về kết quả là do dữ liệu hoặc feature.
3. **Kết luận chính rõ ràng.** Thêm dữ liệu giả (kể cả bằng LLM) không giúp; thêm bối cảnh hành vi giúp rất nhiều.
4. **Nhóm trung thực về giới hạn.** Không có thuật toán mới; nhóm tự chỉ ra điểm yếu của chính đánh giá.

**Thông điệp một câu** (nói ở slide 1, nhắc lại ở slide cuối):

> Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn bổ sung dữ liệu giả.

**Ba con số cả nhóm phải thuộc:**

| Con số | Ý nghĩa |
|---|---|
| AP 0,171 → 0,684 | Thêm 63 feature hành vi, cùng một model XGBoost |
| 915 / 1.000 | Số alert đúng là rửa tiền trong 1.000 alert cao nhất (F1 chỉ 712) |
| Discriminator AUC ≥ 0,98 | Dữ liệu CTGAN và LLM sinh ra bị phân biệt gần như hoàn toàn với dữ liệu thật |

**Không được nói:** "đây là phương pháp mới", "tốt hơn GFP/GNN" như một kết luận chắc chắn, "có ý nghĩa thống kê",
"áp dụng được ngay cho ngân hàng thật".

## 2. Phân vai và mốc thời gian

Tổng thời lượng **13:45**, còn khoảng 30 giây dự phòng trong giới hạn 15 phút.

| Người nói | Họ tên (nhóm tự điền) | Slide | Thời lượng |
|---|---|---|---|
| 1 | … | 1–7 | 4:40 |
| 2 | … | 8–12 | 5:05 |
| 3 | … | 13–17 | 4:00 |

Thay `[tên người nói 2]`, `[tên người nói 3]` trong lời nói bằng tên thật. Nên phân người theo phần mình đã trực tiếp
làm để trả lời câu hỏi sâu được.

**Mốc kiểm tra thời gian** (cột "Kết thúc lúc"): nếu đến cuối slide 7 đã quá 5:15, hoặc cuối slide 12 đã quá 10:30,
hãy rút gọn các slide sau theo mục 4.

| # | Slide | Người nói | Thời lượng | Kết thúc lúc |
|---|---|---|---|---|
| 1 | Phát hiện rửa tiền bằng machine learning | 1 | 0:20 | 0:20 |
| 2 | Nội dung trình bày | 1 | 0:20 | 0:40 |
| 3 | Giám sát giao dịch hoạt động như một IDS cho dòng tiền | 1 | 0:50 | 1:30 |
| 4 | Ba khó khăn khiến bài toán dễ bị đánh giá sai | 1 | 0:50 | 2:20 |
| 5 | Thêm dữ liệu hay thêm bối cảnh? Hai hướng, một protocol | 1 | 0:40 | 3:00 |
| 6 | Đóng góp: một phép so sánh có kiểm soát, không phải thuật toán mới | 1 | 0:50 | 3:50 |
| 7 | IBM AML HI-Small: 5,08 triệu giao dịch, chia theo thời gian | 1 | 0:50 | 4:40 |
| 8 | Cùng model, cùng split, cùng cách chọn threshold | 2 | 0:45 | 5:25 |
| 9 | Dữ liệu tổng hợp không giúp; dữ liệu LLM làm AP giảm mạnh nhất | 2 | 1:20 | 6:45 |
| 10 | Chỉ nhìn quá khứ: lịch sử nghiêm ngặt theo timestamp | 2 | 0:50 | 7:35 |
| 11 | 63 feature hành vi chia thành bốn nhóm | 2 | 0:50 | 8:25 |
| 12 | Lịch sử tạo bước nhảy lớn nhất; kết hợp ba nhóm cho AP cao nhất | 2 | 1:20 | 9:45 |
| 13 | Cùng 1.000 cảnh báo, F4 tìm thêm khoảng 200 ca rửa tiền | 3 | 0:50 | 10:35 |
| 14 | Kiểm tra lại tính hợp lệ của chính đánh giá | 3 | 1:10 | 11:45 |
| 15 | Detector là thành phần phòng thủ, và cũng có bề mặt tấn công | 3 | 0:50 | 12:35 |
| 16 | Giới hạn và hướng phát triển | 3 | 0:40 | 13:15 |
| 17 | Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn bổ sung dữ liệu giả. | 3 | 0:30 | 13:45 |
| 18 | Bảng số liệu đầy đủ (dự phòng) | – | – | chỉ mở khi được hỏi |

## 3. Kịch bản từng slide

### Slide 1. Phát hiện rửa tiền bằng machine learning
*Người nói 1 · ~20 giây*

- **Mục đích:** Chào, giới thiệu nhóm và nêu câu hỏi trung tâm để người nghe biết cả bài đi về đâu.
- **Chỉ vào:** Hình mạng tài khoản bên phải.
- **Nhấn mạnh:** Câu hỏi trung tâm: thêm dữ liệu hay thêm bối cảnh?

> Kính chào thầy cô và các bạn. Nhóm em gồm Nguyễn Thế Vĩnh, Hoàng Đức Chung và Nguyễn Hải Đăng, xin trình bày đồ án Phát hiện rửa tiền bằng machine learning. Câu hỏi trung tâm của nhóm là: muốn phát hiện rửa tiền tốt hơn, nên thêm dữ liệu tổng hợp, hay cho model thấy bối cảnh hành vi của tài khoản? Như hình bên phải, tín hiệu nằm ở chuỗi giao dịch giữa các tài khoản, không ở một giao dịch đơn lẻ.

**Chuyển:** Bài trình bày gồm sáu phần.

### Slide 2. Nội dung trình bày
*Người nói 1 · ~20 giây*

- **Mục đích:** Cho người nghe bản đồ của bài nói và giới thiệu ai trình bày phần nào.
- **Chỉ vào:** Lần lượt sáu thẻ, nhấn vào thẻ 3 và 4 (hai hướng A, B).
- **Nhấn mạnh:** Hai hướng A và B là trục chính của bài.

> Đầu tiên là bài toán và dữ liệu. Phần chính là hai hướng cải thiện: hướng A thêm dữ liệu, hướng B thêm bối cảnh. Sau đó nhóm tự kiểm tra tính hợp lệ của đánh giá, liên hệ với an toàn bảo mật, và kết luận. Em trình bày phần một và hai, [tên người nói 2] trình bày hai hướng, và [tên người nói 3] trình bày phần còn lại.

**Chuyển:** Trước hết, vì sao đây là bài toán của môn an toàn thông tin?

### Slide 3. Giám sát giao dịch hoạt động như một IDS cho dòng tiền
*Người nói 1 · ~50 giây*

- **Mục đích:** Kết nối đề tài với môn An toàn và bảo mật hệ thống thông tin: đây là một bài toán phát hiện.
- **Chỉ vào:** Sơ đồ bốn khối hàng trên, rồi hàng IDS tương ứng bên dưới.
- **Nhấn mạnh:** Đây là bài toán phát hiện, giống phát hiện xâm nhập.

> Rửa tiền là hợp thức hóa tiền bất hợp pháp qua chuỗi giao dịch để che giấu nguồn gốc. Ngân hàng có nghĩa vụ giám sát và báo cáo giao dịch đáng ngờ theo khuyến nghị FATF và Luật Phòng, chống rửa tiền 2022. Nhìn từ góc độ an toàn thông tin, hệ thống giám sát giao dịch hoạt động như một IDS: luồng giao dịch giống luồng gói tin, detector chấm điểm, chỉ một số alert đầu danh sách được xem, và chuyên viên điều tra đóng vai SOC analyst. Mỗi false positive tốn thời gian chuyên viên; mỗi false negative là một dòng tiền bẩn lọt qua.

**Chuyển:** Bài toán này có ba khó khăn khiến rất dễ đánh giá sai.

### Slide 4. Ba khó khăn khiến bài toán dễ bị đánh giá sai
*Người nói 1 · ~50 giây*

- **Mục đích:** Nêu ba bẫy của bài toán; mỗi bẫy dẫn tới một quyết định thiết kế của nhóm.
- **Chỉ vào:** Ba con số lớn: 0,102%, AP 0,171, > 300.
- **Nhấn mạnh:** Accuracy 99,9% là vô nghĩa.

> Thứ nhất, positive class cực hiếm: chỉ 0,102% giao dịch là rửa tiền. Đoán tất cả là bình thường đã có accuracy 99,9%, nên nhóm dùng Average Precision. Thứ hai, tín hiệu nằm ở bối cảnh: nếu chỉ nhìn thuộc tính của từng giao dịch, AP chỉ đạt 0,171. Thứ ba, dễ rò rỉ thời gian: timestamp chỉ đến phút, mỗi phút có hơn 300 giao dịch; feature vô tình dùng tương lai hoặc thứ tự dòng trong file sẽ cho điểm cao giả tạo. Ba khó khăn này dẫn tới ba quyết định của nhóm: dùng AP, xây feature hành vi, và time policy nghiêm ngặt.

**Chuyển:** Từ đó nhóm đặt câu hỏi nghiên cứu.

### Slide 5. Thêm dữ liệu hay thêm bối cảnh? Hai hướng, một protocol
*Người nói 1 · ~40 giây*

- **Mục đích:** Đưa ra khung của toàn bài: hai hướng cải thiện được so sánh trên cùng một protocol.
- **Chỉ vào:** Hai cột Hướng A, Hướng B, rồi ô tối 'Protocol cố định'.
- **Nhấn mạnh:** So sánh có kiểm soát.

> Hướng A bổ sung giao dịch rửa tiền tổng hợp vào train set: oversampling, SMOTENC, CTGAN và LLM. Hướng B mô tả hành vi theo thời gian của tài khoản bằng ba nhóm feature. Điểm quan trọng nhất là ô phía dưới: hai hướng được đánh giá trên cùng một protocol, cùng temporal split, cùng XGBoost cố định, cùng cách chọn threshold, ba seed. Nhờ vậy mọi khác biệt chỉ đến từ dữ liệu hoặc feature.

**Chuyển:** Nhóm xin nói rõ đóng góp của đồ án.

### Slide 6. Đóng góp: một phép so sánh có kiểm soát, không phải thuật toán mới
*Người nói 1 · ~50 giây*

- **Mục đích:** Định vị trung thực: không có thuật toán mới; đóng góp nằm ở thiết kế thí nghiệm và tính đúng của đánh giá.
- **Chỉ vào:** Bốn đóng góp bên trái, rồi bảng công trình liên quan bên phải.
- **Nhấn mạnh:** Không phải thuật toán mới; là so sánh có kiểm soát.

> Đồ án không đề xuất thuật toán mới. Đóng góp là bốn điểm bên trái: một bộ 63 feature tính nghiêm ngặt theo thời gian; ablation có kiểm soát với 5 cấu hình định trước; đánh giá 4 phương pháp augmentation, kể cả LLM, có chẩn đoán chất lượng dữ liệu; và phân tích tính hợp lệ của chính đánh giá. So với công trình liên quan: nhóm dùng lại dataset AMLworld của IBM; khác Graph Feature Preprocessor ở chỗ dùng feature cục bộ, dễ giải thích và time policy nghiêm ngặt hơn; khác Multi-GNN ở chỗ dùng XGBoost rẻ và giải thích được; và thử LLM sinh dữ liệu cho bài toán AML cực mất cân bằng.

**Chuyển:** Tiếp theo là dữ liệu.

### Slide 7. IBM AML HI-Small: 5,08 triệu giao dịch, chia theo thời gian
*Người nói 1 · ~50 giây*

- **Mục đích:** Giới thiệu dữ liệu và cách chia đúng theo thời gian, làm nền cho các con số phía sau.
- **Chỉ vào:** Thanh Train / Val / Test và bảng chia tập.
- **Nhấn mạnh:** Chia theo thời gian, không chia ngẫu nhiên.

> Dữ liệu là IBM AML HI-Small: hơn 5 triệu giao dịch mô phỏng từ ngày 1 đến 18 tháng 9 năm 2022, trong đó 5.177 giao dịch rửa tiền. Nhóm chia theo thời gian 60/20/20 như thanh màu phía trên, không chia ngẫu nhiên, vì thực tế model học từ quá khứ để dự đoán tương lai. Điểm cắt được dịch về đầu phút để một phút không bị tách sang hai tập. Nhãn của validation và test không tham gia tính feature, và ID tài khoản chỉ dùng để tra cứu lịch sử, không đưa vào model.

**Chuyển:** Tiếp theo, [tên người nói 2] sẽ trình bày thiết lập thực nghiệm và hai hướng cải thiện.

### Slide 8. Cùng model, cùng split, cùng cách chọn threshold
*Người nói 2 · ~45 giây*

- **Mục đích:** Chứng minh phép so sánh công bằng: mọi cấu hình dùng cùng model và cùng cách đánh giá.
- **Chỉ vào:** Ba thẻ Model, Đánh giá, Augmentation.
- **Nhấn mạnh:** Không tuning riêng: điểm tăng đến từ feature hoặc dữ liệu.

> Cảm ơn bạn. Mọi cấu hình dùng cùng một XGBoost, tối đa 1.000 cây, early stopping theo AUC-PR trên validation. Nhóm cố ý không tuning riêng từng cấu hình, để điểm tăng chắc chắn đến từ feature hoặc dữ liệu, không phải từ tuning. Chỉ số chính là AP; threshold chọn trên validation rồi giữ nguyên khi chạy test. Precision và Recall trong 1.000 alert cao nhất mô phỏng năng lực của chuyên viên. Với augmentation, dữ liệu tổng hợp chỉ thêm vào train, 1.148 mẫu mỗi lượt; validation và test luôn là dữ liệu thật.

**Chuyển:** Bắt đầu với hướng A: thêm dữ liệu.

### Slide 9. Dữ liệu tổng hợp không giúp; dữ liệu LLM làm AP giảm mạnh nhất
*Người nói 2 · ~1 phút 20 giây*

- **Mục đích:** Trình bày kết quả âm của hướng A có giải thích nguyên nhân; đây là lý do chuyển sang hướng B.
- **Chỉ vào:** Biểu đồ trái (ΔAP), rồi biểu đồ phải (discriminator AUC).
- **Nhấn mạnh:** Dữ liệu giả không giống thật nên làm model học sai.

> Biểu đồ trái là AP thay đổi bao nhiêu so với không augmentation, mốc AP 0,670. SMOTENC tăng 0,003, nằm trong dao động giữa các seed, tức là không cải thiện thật. Oversampling và CTGAN giảm nhẹ. Hai biến thể LLM giảm mạnh nhất, khoảng 0,06 đến 0,07, kể cả khi đưa kiến thức về các kiểu rửa tiền vào prompt. Biểu đồ phải giải thích vì sao. Nhóm huấn luyện một classifier để phân biệt dữ liệu giả với dữ liệu thật: AUC 0,5 nghĩa là không phân biệt được. Với CTGAN và LLM, AUC từ 0,98 trở lên, tức dữ liệu giả lộ rõ là giả. Mẫu giả vẫn mang nhãn rửa tiền, nên model học những vùng không tồn tại trong thực tế và bỏ sót ca thật. Lưu ý thí nghiệm này dùng pipeline cũ, nên chỉ so các phương pháp với nhau.

**Chuyển:** Vì thêm dữ liệu không giúp, nhóm chuyển sang hướng B: thêm bối cảnh.

### Slide 10. Chỉ nhìn quá khứ: lịch sử nghiêm ngặt theo timestamp
*Người nói 2 · ~50 giây*

- **Mục đích:** Trình bày nền tảng tính đúng của hướng B: không nhìn trước tương lai, không phụ thuộc thứ tự dòng.
- **Chỉ vào:** Công thức H(t), rồi hình timeline (cùng phút chung snapshot, tương lai không dùng).
- **Nhấn mạnh:** Không nhìn trước tương lai, không phụ thuộc thứ tự dòng.

> Nền tảng của hướng B là time policy. Lịch sử của giao dịch tại thời điểm t chỉ gồm các giao dịch có timestamp nhỏ hơn t. Vì timestamp chỉ đến phút và không biết thứ tự thật trong cùng phút, mọi giao dịch cùng phút dùng chung một snapshot lịch sử: như hình, các giao dịch tại t không thấy nhau, và tương lai không bao giờ được dùng. Cửa sổ thời gian là từ t trừ W đến t, với W là 1 giờ và 24 giờ. Tổng tiền tách theo currency vì không có tỷ giá. Có 43 regression test kiểm tra đúng các quy tắc này.

**Chuyển:** Trên nền đó, nhóm xây 63 feature.

### Slide 11. 63 feature hành vi chia thành bốn nhóm
*Người nói 2 · ~50 giây*

- **Mục đích:** Giải thích bốn nhóm feature bằng ví dụ dễ hiểu và cách thiết kế 5 cấu hình F0–F4.
- **Chỉ vào:** Bốn thẻ màu, rồi bảng cấu hình F0–F4.
- **Nhấn mạnh:** Mỗi nhóm một ví dụ đời thường.

> 63 feature chia bốn nhóm. Transaction là thuộc tính của chính giao dịch. History là lịch sử tài khoản, ví dụ tài khoản đã nhận bao nhiêu lần, giao dịch gần nhất cách bao lâu. Activity là nhịp độ trong 1 giờ và 24 giờ, và số tiền hiện tại lệch bao nhiêu so với thói quen. Relationship là quan hệ đối tác, ví dụ cặp tài khoản đã giao dịch với nhau bao lâu, có bao nhiêu đối tác khác nhau. Bảng phía dưới là 5 cấu hình F0 đến F4 định trước khi chạy, để mỗi hiệu số đo đúng đóng góp của một nhóm.

**Chuyển:** Và đây là kết quả.

### Slide 12. Lịch sử tạo bước nhảy lớn nhất; kết hợp ba nhóm cho AP cao nhất
*Người nói 2 · ~1 phút 20 giây*

- **Mục đích:** Kết quả chính của đồ án: thêm bối cảnh hành vi làm AP tăng mạnh và nhất quán.
- **Chỉ vào:** Các cột F0 → F4 ở biểu đồ trái, rồi dòng ΔAP phía dưới.
- **Nhấn mạnh:** History tạo bước nhảy lớn nhất; F4 đạt AP 0,684; 3/3 seed đều tăng.

> Chỉ dùng thuộc tính giao dịch, F0, AP là 0,171. Thêm History, AP nhảy lên 0,534, đây là bước nhảy lớn nhất. Khi đã có History, Activity đóng góp nhiều hơn Relationship: 0,662 so với 0,610. Kết hợp cả ba nhóm, F4 đạt 0,684, gấp bốn lần F0 với cùng một model. Recall@1000 bên phải có cùng xu hướng. Dòng dưới cùng là hiệu số theo cặp cùng seed: mọi bước bổ sung đều dương ở cả ba seed, kể cả F4 so với F2, nghĩa là Relationship vẫn mang thông tin bổ sung. Ba seed cho thấy tính nhất quán, nhưng chưa đủ để khẳng định ý nghĩa thống kê.

**Chuyển:** Phần tiếp theo, [tên người nói 3] sẽ trình bày ý nghĩa của kết quả này.

### Slide 13. Cùng 1.000 cảnh báo, F4 tìm thêm khoảng 200 ca rửa tiền
*Người nói 3 · ~50 giây*

- **Mục đích:** Dịch AP sang ngôn ngữ vận hành, nêu đánh đổi precision/recall và diễn giải feature quan trọng.
- **Chỉ vào:** Cột F1 và F4 ở biểu đồ trái, rồi top feature bên phải.
- **Nhấn mạnh:** Cùng khối lượng công việc, bắt thêm khoảng 200 ca; nhưng vẫn sót gần một nửa.

> Cảm ơn bạn. Vậy con số này có ý nghĩa gì với một ngân hàng? Giả sử chuyên viên chỉ xử lý được 1.000 alert. Với F1, khoảng 712 alert là rửa tiền thật; với F4 là 915, precision 91,5%. Cùng khối lượng công việc, tìm thêm khoảng 200 ca. Nhưng recall chỉ 50,9%: vẫn còn 883 ca nằm ngoài danh sách, muốn bắt nhiều hơn phải chấp nhận nhiều alert hơn. Bên phải là các feature quan trọng nhất: số lượt nhận trước đó của sender, thời gian từ giao dịch gần nhất của receiver, độ lệch số tiền so với lịch sử. Đó là dấu hiệu money mule: tài khoản vừa nhận tiền rồi chuyển đi với số tiền bất thường. Gain importance không phải quan hệ nhân quả, nhưng giúp giải thích alert cho chuyên viên.

**Chuyển:** Một kết quả tốt chỉ có giá trị khi cách đánh giá đúng, nên nhóm tự kiểm tra lại.

### Slide 14. Kiểm tra lại tính hợp lệ của chính đánh giá
*Người nói 3 · ~1 phút 10 giây*

- **Mục đích:** Thể hiện tư duy phản biện: chủ động nêu điểm yếu của đánh giá trước khi bị hỏi.
- **Chỉ vào:** Hai biểu đồ theo ngày, rồi ba mục a, b, c.
- **Nhấn mạnh:** Nhóm tự tìm ra điểm yếu của chính đánh giá.

> Có ba điểm. Một, distribution shift: từ ngày 11 tháng 9, lưu lượng giảm từ hơn 200 nghìn xuống vài trăm giao dịch mỗi ngày, còn tỷ lệ rửa tiền vọt lên gần 60%. Phát hiện này đến sau khi chạy, nên nhóm không xóa dữ liệu mà tách riêng để kiểm tra: trên phần trước 11 tháng 9, F4 vẫn hơn F1 rõ rệt, 0,528 so với 0,333. Hai, time policy ảnh hưởng mạnh: pipeline cũ xử lý cùng phút theo thứ tự dòng đạt AP 0,670, trong khi pipeline nghiêm ngặt với 28 feature tương ứng chỉ 0,534; đây không phải so sánh có kiểm soát, nhưng cho thấy cách xử lý thời gian có thể thổi phồng điểm số. Ba, test set đã được dùng trong thí nghiệm trước, nên kết quả mang tính khám phá.

**Chuyển:** Những điều này liên hệ trực tiếp với môn an toàn bảo mật.

### Slide 15. Detector là thành phần phòng thủ, và cũng có bề mặt tấn công
*Người nói 3 · ~50 giây*

- **Mục đích:** Trả lời trực tiếp câu hỏi 'đồ án liên quan gì đến môn An toàn và bảo mật hệ thống thông tin'.
- **Chỉ vào:** Bốn thẻ theo thứ tự: phòng thủ, leakage, evasion, bảo vệ dữ liệu.
- **Nhấn mạnh:** Leakage là lỗ hổng phương pháp; evasion là bề mặt tấn công.

> Có bốn ý. Một, detector là thành phần phòng thủ: hiệu quả đo bằng số ca tìm được trong alert budget, và feature hành vi giải thích được bằng ngôn ngữ nghiệp vụ. Hai, data leakage là một lỗ hổng phương pháp: hệ thống có điểm cao, trông an toàn, nhưng thực ra không. Ba, chính feature hành vi tạo ra bề mặt tấn công: kẻ rửa tiền có thể chia nhỏ giao dịch, giãn thời gian để thoát cửa sổ, hoặc dùng money mule; đây là evasion attack mà nhóm chưa đo được. Bốn, bảo vệ dữ liệu và bí mật: account ID không phải feature, repository không chứa API key hay dữ liệu thô.

**Chuyển:** Cuối cùng là giới hạn và hướng phát triển.

### Slide 16. Giới hạn và hướng phát triển
*Người nói 3 · ~40 giây*

- **Mục đích:** Nói rõ phạm vi của kết luận và chỉ ra bước tiếp theo.
- **Chỉ vào:** Cột Giới hạn, rồi cột Hướng phát triển.
- **Nhấn mạnh:** Kết quả có giá trị so sánh tương đối.

> Dữ liệu là mô phỏng; test set đã được xem trước; ba seed chưa đủ cho ý nghĩa thống kê; hyperparameter cố định; feature quan hệ mới là cục bộ. Vì vậy kết quả có giá trị so sánh tương đối giữa các phương pháp, chưa khẳng định hiệu năng ở ngân hàng thật. Hướng phát triển: so sánh với GFP dưới cùng time policy, phân tích theo tám kiểu rửa tiền, đánh giá khả năng chống né, và kiểm chứng trên dataset khác.

**Chuyển:** Xin tóm lại.

### Slide 17. Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn bổ sung dữ liệu giả.
*Người nói 3 · ~30 giây*

- **Mục đích:** Chốt thông điệp một câu và ba con số, mời hội đồng đặt câu hỏi.
- **Chỉ vào:** Ba con số lớn.
- **Nhấn mạnh:** Thông điệp một câu.

> Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn thêm dữ liệu giả. Ba con số: feature hành vi nâng AP thêm 0,513; precision 91,5% trong 1.000 alert; còn dữ liệu do CTGAN và LLM sinh ra bị phân biệt với dữ liệu thật ở mức AUC từ 0,98. Toàn bộ mã nguồn và số liệu có trên GitHub. Nhóm em xin cảm ơn thầy cô và các bạn đã lắng nghe, và sẵn sàng nhận câu hỏi.

**Chuyển:** (Hỏi đáp; mở slide dự phòng khi được hỏi số liệu chi tiết.)

## 4. Mẹo khi nói

- **Nói theo ý, không đọc thuộc lòng.** Mỗi slide chỉ cần nhớ mục đích, chỗ cần chỉ và câu nhấn mạnh; lời nói ở trên
  là gợi ý.
- **Chỉ vào slide khi nói con số** để hội đồng nhìn đúng chỗ.
- **Câu chuyển là chỗ hay bị quên.** Đặc biệt câu chuyển ở slide 9 ("vì thêm dữ liệu không giúp, nhóm chuyển sang hướng
  B") là lý do của cả phần sau.
- **Nếu bị lố giờ:** bỏ slide 2 (nội dung) và slide 6 (đóng góp); ở slide 14 chỉ nói ý (a); slide 15 nói trong 30 giây.
- **Nếu bị hỏi giữa chừng:** trả lời ngắn một câu, nói "nhóm sẽ trình bày kỹ ở slide …" nếu câu hỏi thuộc phần sau,
  rồi quay lại mạch.
- **Khi bị hỏi số liệu chi tiết:** mở slide 18 (dự phòng).
- **Nếu bị hỏi "so với các paper khác thì sao?":** xem câu A4 trong [thuyet-trinh-va-cau-hoi.md](thuyet-trinh-va-cau-hoi.md).
