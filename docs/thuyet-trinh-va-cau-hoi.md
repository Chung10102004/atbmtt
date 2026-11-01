# Chuẩn bị bảo vệ đồ án: Phát hiện rửa tiền bằng machine learning

Tài liệu nội bộ của nhóm để chuẩn bị thuyết trình và trả lời câu hỏi. Số liệu lấy từ bài báo
([paper/main.pdf](../paper/main.pdf), [paper/main_en.pdf](../paper/main_en.pdf)) và `reports/evidence/`.

---

## 1. Đồ án này khác gì các công trình đã có?

### Thông điệp một câu

> Nhóm **không đề xuất thuật toán mới**. Đóng góp là một **phép so sánh có kiểm soát** giữa hai hướng cải thiện
> detector (data augmentation, kể cả bằng LLM, và behavioral feature engineering) trên **cùng một protocol nghiêm
> ngặt về thời gian**, kèm phân tích tính hợp lệ của evaluation.

Khi bị hỏi về tính mới, hãy nói thẳng như trên rồi chỉ ra các điểm khác biệt cụ thể dưới đây. Không nói "đây là công
trình đầu tiên"; nói "theo khảo sát của nhóm, chưa thấy công trình nào làm đủ các điểm này cùng lúc".

### So với các công trình đã công bố

| Công trình | Họ làm gì | Đồ án này khác ở đâu |
|---|---|---|
| **Altman et al., NeurIPS 2023** (tạo ra dataset IBM AML) | Công bố bộ sinh dữ liệu AMLworld và 6 dataset, kèm baseline GNN và gradient boosting với feature cơ bản | Nhóm dùng lại dataset của họ. Trọng tâm của nhóm là feature engineering và data augmentation, không phải tạo dữ liệu |
| **Graph Feature Preprocessor (GFP), Blanuša et al., ICAIF 2024** | Trích pattern đồ thị nhiều bước (fan-in/out, scatter-gather, cycle) + vertex statistics, đưa vào XGBoost/LightGBM. Trên HI-Small đạt F1 khoảng 0,63–0,65 | (1) Nhóm dùng **feature hành vi cục bộ, dễ giải thích** (history, activity theo window, relationship), không dò cycle nhiều bước. (2) **Time policy nghiêm ngặt**: chỉ dùng giao dịch có timestamp nhỏ hơn, các giao dịch cùng phút dùng chung một snapshot lịch sử. GFP xử lý theo batch 128/2048 cạnh. (3) **Tổng tiền tách theo currency**. (4) **Ablation từng feature group** + paired difference theo seed. (5) Đánh giá theo **alert budget** (Precision/Recall@1000) |
| **Multi-GNN (Egressy et al., AAAI 2024), FraudGT, TeMP-TraG** | Graph Neural Network cho directed multigraph | Nhóm dùng XGBoost + feature giải thích được: rẻ hơn nhiều (tính feature khoảng 35 giây, train 15 model khoảng 10 phút trên Kaggle), dễ giải thích cho chuyên viên điều tra |
| **Deprez et al., INFORMS J. Data Science 2024** | Tổng quan 97 bài; thí nghiệm chủ yếu trên Elliptic (Bitcoin) | Nhóm làm trên dữ liệu ngân hàng mô phỏng (IBM AML), có so sánh với data augmentation |
| **Effendi & Chattopadhyay 2024** | Ghi nhận XGBoost có thể không cần graph feature ở một số cấu hình | Phù hợp với kết quả của nhóm: feature hành vi cục bộ đã cho kết quả cao |
| **GReaT (ICLR 2023), CLLM (ICML 2024)**: LLM sinh dữ liệu dạng bảng | Cho thấy LLM có thể sinh tabular data hữu ích khi ít dữ liệu | Nhóm thử LLM cho **bài toán AML cực kỳ mất cân bằng** và có **chẩn đoán chất lượng** (discriminator AUC). Kết quả **âm**: LLM làm AP giảm, và thêm domain knowledge vào prompt cũng không giúp |

### So với các lỗi thường gặp trong bài làm trên dataset này

Nhiều bài làm mẫu trên dataset IBM AML mắc các lỗi dưới đây. Đồ án tránh tất cả, và đây là điểm nên nhấn mạnh với
môn An toàn bảo mật:

| Lỗi thường gặp | Đồ án làm thế nào |
|---|---|
| Chia train/test **ngẫu nhiên**, nên model "thấy tương lai" | Chia **theo thời gian** 60/20/20; không tách một phút sang hai tập |
| Báo cáo **accuracy** (được khoảng 99,9% chỉ bằng cách đoán toàn bộ là "bình thường") | Dùng **AP**, F1, **Precision/Recall@1000** |
| **SMOTE trên toàn bộ dữ liệu trước khi chia** (rò rỉ sang test) | Augmentation chỉ dùng positive sample của **train**; validation/test luôn là dữ liệu thật |
| Đưa **account ID** vào model | ID chỉ dùng để tra cứu lịch sử, không phải feature |
| **Chọn threshold trên test** | Threshold chọn trên validation, giữ nguyên khi áp vào test |
| Feature lịch sử tính theo **thứ tự dòng trong file** | Snapshot theo timestamp, bất biến khi đổi thứ tự dòng (có test kiểm tra) |

### So với đề cương ban đầu của nhóm

- **Ban đầu:** dùng LLM để làm giàu dữ liệu train (data augmentation), rồi benchmark lại.
- **Kết quả:** cả 4 phương pháp augmentation (random oversampling, SMOTENC, CTGAN, LLM) đều không cải thiện. LLM còn
  kém nhất (AP 0,670 xuống 0,609 và 0,597).
- **Chuyển hướng:** sang feature engineering, giúp AP tăng từ 0,171 lên 0,684.
- **Cách trình bày khi bị hỏi:** đây là kết quả thực nghiệm dẫn tới chuyển hướng, không phải bỏ đề cương. Phần LLM
  vẫn được báo cáo đầy đủ như một **kết quả âm có phân tích nguyên nhân**.

---

## 2. Flow thuyết trình (10–15 phút)

Mục tiêu khoảng **14 phút** nói, còn khoảng 1 phút dự phòng. Có 3 người nói: người 1 khoảng 4 phút (slide 1–5), người 2 khoảng 5,5 phút (slide 6–9), người 3 khoảng 4,5 phút (slide 10–13). Nên phân người theo
phần mình đã trực tiếp làm, để trả lời câu hỏi sâu được.

| # | Slide | Thời gian | Người nói | Nội dung chính | Câu cần nói được |
|---|---|---|---|---|---|
| 1 | Tiêu đề | 0:20 | 1 | Tên đề tài, thành viên, môn học | – |
| 2 | Bối cảnh | 1:00 | 1 | Rửa tiền; nghĩa vụ báo cáo giao dịch đáng ngờ (FATF, Luật PCRT 2022). **Transaction monitoring giống một IDS**: xếp hạng luồng sự kiện lớn, mỗi false positive tốn thời gian chuyên viên | "Đây là bài toán phát hiện, giống phát hiện xâm nhập" |
| 3 | Ba khó khăn | 1:00 | 1 | (1) Positive class chỉ khoảng 0,1%; (2) một giao dịch đơn lẻ không có tín hiệu, tín hiệu nằm ở bối cảnh; (3) dễ temporal leakage | "Accuracy 99,9% là vô nghĩa" |
| 4 | Dataset và split | 1:00 | 1 | IBM AML HI-Small: 5.078.345 giao dịch, 5.177 rửa tiền (0,102%). Bảng split theo thời gian (train 2.297 / val 1.082 / test 1.798 positive) | "Chia theo thời gian, không random" |
| 5 | Câu hỏi nghiên cứu | 0:40 | 1 | Hai hướng: **thêm dữ liệu** (augmentation, LLM) và **thêm bối cảnh** (feature). Cùng XGBoost, cùng split, cùng cách chọn threshold, 3 seed | "So sánh có kiểm soát" |
| 6 | Data augmentation | 2:00 | 2 | 6 cấu hình E0, ER, E1, E2, E3a, E3b. Bảng AP + discriminator AUC. Thông điệp: **không cái nào tốt hơn E0**; CTGAN và LLM sinh dữ liệu bị phân biệt gần như hoàn toàn (0,98–0,999) | "Dữ liệu giả không giống thật nên làm model học sai" |
| 7 | Time policy | 1:00 | 2 | Công thức H(t) = {j : t_j < t}; giao dịch cùng phút dùng chung snapshot; window [t−W, t); tổng tiền tách theo currency | "Không nhìn trước tương lai, không phụ thuộc thứ tự dòng" |
| 8 | 4 feature group | 1:00 | 2 | Transaction 11, History 17, Activity 20, Relationship 15 = 63. Mỗi nhóm cho 1–2 ví dụ dễ hiểu | – |
| 9 | Kết quả ablation | 1:30 | 2 | Hình AP/Recall@1000 F0 đến F4: 0,171, 0,534, 0,662 / 0,610, 0,684. Mọi bước thêm đều tăng AP ở 3/3 seed | "History tạo bước nhảy lớn nhất" |
| 10 | Ý nghĩa vận hành | 1:00 | 3 | Trong 1.000 alert: F1 bắt đúng khoảng 712, F4 bắt đúng 915 (precision 91,5%). Nhưng recall 50,9%: vẫn sót khoảng một nửa. Feature importance: tài khoản vừa nhận tiền rồi chuyển đi bất thường là dấu hiệu money mule | "Cùng khối lượng công việc, bắt được nhiều hơn khoảng 200 ca" |
| 11 | Tính hợp lệ | 1:30 | 3 | (a) Distribution shift từ 11/09 (Hình daily profile); trước 11/09 F4 vẫn hơn F1 (0,528 so với 0,333). (b) Pipeline cũ xử lý theo thứ tự dòng đạt AP 0,670, còn F1 nghiêm ngặt đạt 0,534, nên time policy ảnh hưởng mạnh. (c) Test set đã dùng lại nên kết quả mang tính exploratory | "Nhóm tự tìm ra điểm yếu của chính đánh giá của mình" |
| 12 | Góc nhìn ATBMHTTT | 1:00 | 3 | Detector là thành phần phòng thủ; feature giải thích được cho chuyên viên; **attack surface**: structuring, giãn thời gian, money mule (evasion); bảo vệ dữ liệu và API key (repo không chứa key, dữ liệu thô) | – |
| 13 | Giới hạn, hướng phát triển, kết luận | 1:00 | 3 | 3–4 giới hạn chính; future work: so sánh GFP cùng time policy, phân tích theo typology, đánh giá evasion; câu kết | Câu kết: "Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn thêm dữ liệu giả" |
| | **Tổng** | **14:00** | | | |

**Mẹo:**
- Slide 6 và 9 là hai slide "kết quả", nên dùng hình hoặc bảng lớn, ít chữ. Hình có sẵn trong `paper/figures/`.
- Nếu bị giới hạn 10 phút: gộp slide 7 và 8, rút slide 11 còn ý (a), bỏ slide 12 vào phần trả lời câu hỏi.
- Chuẩn bị sẵn 1 slide dự phòng (backup) chứa bảng đầy đủ F0–F4 và bảng augmentation để mở khi bị hỏi số liệu.

### Bảng số cần thuộc

| Chỉ số | Giá trị |
|---|---|
| Tổng giao dịch / rửa tiền | 5.078.345 / 5.177 (0,102%) |
| Positive train / val / test | 2.297 / 1.082 / 1.798 |
| Số feature F0 / F1 / F2 / F3 / F4 | 11 / 28 / 48 / 43 / 63 |
| AP test F0 → F4 | 0,171 → 0,534 → 0,662 / 0,610 → **0,684** |
| F4: F1-score, Precision@1000, Recall@1000 | 0,685, 91,5%, 50,9% |
| Augmentation AP: E0 / SMOTENC / CTGAN / LLM-a / LLM-b | 0,670 / 0,673 / 0,648 / 0,609 / 0,597 |
| Discriminator AUC: ER / SMOTENC / CTGAN / LLM-a / LLM-b | 0,50 / 0,65 / 0,999 / 0,983 / 0,996 |
| Trước 11/09: AP F1 / F4 | 0,333 / 0,528 |
| Thời gian: tính feature / train 15 model / toàn bộ | khoảng 35 giây / 620 giây / 733 giây |
| Kiểm thử | 43 regression test; xác minh lại 30 prediction file |

---

## 3. Các câu hỏi thầy có thể đặt (kèm gợi ý trả lời)

### A. Tính mới và định vị

**A1. Điểm mới của đồ án là gì?**
Không có thuật toán mới. Đóng góp gồm: (1) so sánh có kiểm soát giữa data augmentation (kể cả LLM) và feature
engineering trên cùng protocol; (2) pipeline feature nghiêm ngặt theo thời gian, có 43 test; (3) kết quả âm có giải
thích cho LLM augmentation, bằng discriminator AUC; (4) phát hiện distribution shift của HI-Small từ 11/09 và ảnh hưởng
của time policy lên điểm số.

**A2. Khác gì GFP của IBM? Sao không dùng GFP?**
GFP dò pattern nhiều bước (cycle, scatter-gather); nhóm dùng thống kê hành vi cục bộ dễ giải thích và time policy
nghiêm ngặt hơn. Nhóm đã thử chạy GFP sơ bộ và thấy hai điểm cần cẩn thận. Thứ nhất, gọi `fit_transform` trên toàn
bộ dữ liệu một lần thì feature chứa thông tin tương lai, nên phải đưa dữ liệu vào tuần tự theo phút. Thứ hai, khi chạy
theo phút, các giao dịch cùng phút nhìn thấy nhau, khác với time policy của nhóm. So sánh công bằng với GFP là hướng
phát triển, chưa nằm trong báo cáo.

**A3. Sao không dùng Graph Neural Network?**
XGBoost với feature tốt là baseline mạnh. Bài GFP cho thấy GBDT + feature vượt GNN tiêu chuẩn trên chính dataset này.
Ngoài ra, chi phí thấp (khoảng 12 phút cho 15 model) và feature giải thích được cho chuyên viên, điều quan trọng trong
AML.

**A4. F1 của nhóm 0,685, GFP 0,63–0,65. Vậy nhóm tốt hơn?**
**Không được kết luận như vậy.** Cách chọn threshold và xử lý thời gian khác nhau; bài báo đã ghi rõ không so sánh
trực tiếp.

### B. Dữ liệu

**B1. Dữ liệu mô phỏng thì kết quả có ý nghĩa gì với ngân hàng thật?**
Dữ liệu ngân hàng thật không công bố được. IBM AML là benchmark chuẩn và có nhãn đầy đủ (dữ liệu thật thì nhãn thiếu).
Kết quả có giá trị **so sánh tương đối** giữa các phương pháp, chưa khẳng định hiệu năng trên ngân hàng thật. Điều này
đã ghi trong phần giới hạn.

**B2. Tại sao chia theo thời gian mà không random?**
Thực tế model học từ quá khứ để dự đoán tương lai. Chia random thì lịch sử tương lai lọt vào train (temporal leakage)
và điểm cao giả tạo.

**B3. Giao dịch cùng phút xử lý thế nào, tại sao quan trọng?**
Timestamp chỉ có độ phân giải phút. Dữ liệu có khoảng 15 nghìn mốc phút, trung bình hơn 300 giao dịch mỗi phút, nên
quyết định này ảnh hưởng gần như mọi dòng. Nhóm cho mọi giao dịch cùng phút dùng chung một snapshot lịch sử, để kết
quả không phụ thuộc thứ tự dòng tùy ý trong file. Bằng chứng ảnh hưởng: pipeline cũ (xử lý theo thứ tự dòng) đạt AP
0,670 với 28 feature, còn F1 nghiêm ngặt chỉ đạt 0,534.

**B4. Sau 11/09 dữ liệu bất thường, sao không bỏ đi?**
Phát hiện này đến sau khi đã chạy (post-hoc). Bỏ dữ liệu sau khi thấy kết quả là "chỉnh dữ liệu theo kết quả". Nhóm
giữ nguyên đánh giá chính và tách riêng để chẩn đoán: trước 11/09, F4 vẫn hơn F1.

**B5. Tại sao HI-Small mà không phải bộ lớn hơn?**
Vừa đủ lớn (5 triệu giao dịch) để có ý nghĩa, vừa chạy được trên Kaggle GPU miễn phí. Kiểm chứng trên HI-Medium hoặc
LI-Small là hướng phát triển.

### C. Phương pháp và đánh giá

**C1. Tại sao dùng AP mà không dùng accuracy hay ROC-AUC?**
Positive class khoảng 0,1%, nên đoán toàn "bình thường" đã có accuracy 99,9%. ROC-AUC cũng cao với mọi cấu hình
(F0 đã đạt khoảng 0,95) nên phân biệt kém. AP tập trung vào khả năng xếp ca rửa tiền lên đầu danh sách, đúng với nhu
cầu của chuyên viên.

**C2. Precision@1000 / Recall@1000 nghĩa là gì?**
Giả sử chuyên viên chỉ kiểm tra được 1.000 alert có score cao nhất. Precision@1000 là tỷ lệ alert đúng; Recall@1000 là
tỷ lệ tổng số ca rửa tiền nằm trong danh sách đó.

**C3. Threshold chọn thế nào?**
Chọn threshold tối đa F1 trên validation của từng lần chạy, áp nguyên cho test. Không bao giờ chỉnh threshold trên
test.

**C4. Tại sao không tune hyperparameter?**
Để so sánh công bằng: nếu tune riêng cho từng cấu hình thì không biết điểm tăng do feature hay do tune. Đổi lại, kết
quả chưa phải hiệu năng tối đa; một số lần chạy chạm giới hạn 1.000 cây. Đã ghi trong giới hạn.

**C5. Chỉ 3 seed thì có ý nghĩa thống kê không?**
Không đủ để khẳng định statistical significance; bài báo không tuyên bố điều đó. Ba seed cho thấy tính nhất quán: mọi
paired difference đều dương ở 3/3 seed.

**C6. Tại sao XGBoost?**
Mạnh trên dữ liệu dạng bảng, xử lý NaN tự nhiên, chạy GPU nhanh, có feature importance. Mục tiêu là giữ model cố định
để đo tác động của feature.

### D. Data augmentation và LLM

**D1. Tại sao LLM lại làm kết quả kém đi?**
Discriminator AUC khoảng 0,98–0,996 cho thấy dữ liệu LLM sinh ra khác hẳn dữ liệu thật. Mỗi giao dịch rửa tiền là
một phần của chuỗi có cấu trúc thời gian và quan hệ. LLM sinh từng dòng độc lập trong feature space nên không giữ được
các ràng buộc đó; model học "vùng giả" và bỏ sót ca thật (recall giảm).

**D2. Dùng model LLM nào, prompt thế nào, tốn bao nhiêu?**
LLM thương mại qua OpenAI API. Prompt gồm ví dụ giao dịch thật, thống kê cột và ràng buộc logic. Biến thể E3b thêm mô
tả các laundering typology. Code có budget ledger giới hạn tổng chi phí **2 USD**.
⚠️ *Trước buổi bảo vệ, người chạy phần LLM cần kiểm tra tên model chính xác trong file metadata của cache LLM
(`data/llm/.../llm_seed*_meta.json`) để trả lời cụ thể.*

**D3. Có kiểm tra LLM chép nguyên dữ liệu thật không?**
Có: output được validate, loại duplicate và loại các dòng trùng nguyên văn dữ liệu thật.

**D4. Discriminator AUC là gì?**
Train một classifier phân biệt "thật" và "giả". AUC 0,5 nghĩa là không phân biệt được (dữ liệu giả rất giống thật);
gần 1 nghĩa là dữ liệu giả lộ rõ.

**D5. SMOTE có bị rò rỉ không?**
Không. SMOTENC chỉ dùng positive sample của tập train; validation và test không bị đụng tới.

**D6. Thí nghiệm augmentation dùng protocol khác, vậy có so được không?**
Chỉ so các phương pháp augmentation **với nhau** (cùng protocol cũ), không so số tuyệt đối với F0–F4. Bài báo ghi rõ
điều này.

### E. Kết quả

**E1. Feature nào quan trọng nhất? Có ý nghĩa nghiệp vụ không?**
Số lượt nhận trước đó của sender, thời gian từ giao dịch gần nhất của receiver, độ lệch số tiền so với lịch sử cùng
currency, payment format, số lần cặp tài khoản đã giao dịch. Ý nghĩa: tài khoản vừa nhận tiền rồi chuyển đi với số tiền
bất thường là dấu hiệu money mule. Lưu ý: gain importance không phải quan hệ nhân quả.

**E2. Recall chỉ khoảng 51%, tức bỏ sót một nửa?**
Đúng, với ngân sách 1.000 alert. Muốn recall cao hơn thì phải chấp nhận nhiều alert hơn; đây là đánh đổi vận hành.
Precision 91,5% nghĩa là alert rất "sạch".

**E3. Tại sao sau 11/09 chỉ F0 đã đạt AP 0,97?**
Phần này chỉ có khoảng 1.100 giao dịch nhưng gần 60% là rửa tiền, nên phân bố khác hẳn; AP phụ thuộc tỷ lệ nhãn. Vì
vậy chỉ so các cấu hình trong cùng một phần.

### F. Góc nhìn An toàn bảo mật

**F1. Đồ án liên quan gì đến môn An toàn và bảo mật hệ thống thông tin?**
(1) Transaction monitoring là cơ chế phát hiện và phòng thủ, giống IDS. (2) **Data leakage là lỗ hổng phương pháp**:
làm điểm đánh giá sai, giống một hệ thống "trông an toàn" nhưng không phải vậy. (3) Bàn về attack surface và evasion.
(4) Quản lý bí mật: API key không đưa vào repo, dữ liệu thô và prediction không công bố.

**F2. Kẻ rửa tiền có né được model không?**
Có khả năng: chia nhỏ giao dịch (structuring), giãn thời gian để thoát time window, thêm money mule để làm nhiễu feature
quan hệ. Nhóm chưa đo được mức độ né; đây là hướng phát triển (evasion robustness).

**F3. Triển khai thực tế thế nào? Có chạy realtime được không?**
Feature là các phép quét tuyến tính, có thể cập nhật tăng dần theo từng giao dịch. Nhóm mới đo thời gian chạy theo lô
(35 giây cho 5 triệu giao dịch), chưa đo latency của một online service.

**F4. Làm sao giải thích một alert cho chuyên viên hoặc cơ quan quản lý?**
Feature có nghĩa nghiệp vụ rõ ("tài khoản nhận 12 lần trong 24 giờ rồi chuyển số tiền gấp 10 lần bình thường"). Có
thể bổ sung SHAP cho từng alert; hiện mới có gain importance toàn cục.

**F5. Dữ liệu khách hàng nhạy cảm thì xử lý sao?**
Model không dùng account ID làm feature; feature là thống kê tổng hợp. Triển khai thật cần access control, ẩn danh hóa,
và có thể dùng federated learning hoặc mã hóa đồng cấu khi nhiều ngân hàng hợp tác (như Effendi & Chattopadhyay 2024).

### G. Tái lập và phân công

**G1. Người khác chạy lại được không?**
Được: code, test, cấu hình và kết quả từng lần chạy có trên GitHub; manifest ghi hash dữ liệu và code; bảng trong bài
báo sinh tự động từ CSV bằng `paper/make_tables.py`. Cần tự tải dataset từ Kaggle.

**G2. 43 test kiểm tra những gì?**
Biên thời gian và window, self-transfer, tách currency, bất biến khi đổi thứ tự dòng cùng phút, không phụ thuộc nhãn và
tương lai, đối chiếu với cách tính trực tiếp trên mẫu nhỏ.

**G3. Mỗi thành viên làm gì?**
⚠️ *Nhóm tự điền trước buổi bảo vệ, và mỗi người nên trình bày đúng phần mình làm.*

| Thành viên | Phần phụ trách |
|---|---|
| Nguyễn Thế Vĩnh (26848266) | … |
| Hoàng Đức Chung (26848014) | … |
| Nguyễn Hải Đăng (26848027) | … |

### H. Nếu có thêm thời gian, nhóm sẽ làm gì?

1. So sánh với GFP **dưới cùng time policy** và phân tích recall theo từng laundering typology (file
   `HI-Small_Patterns.txt` có sẵn nhãn 8 kiểu).
2. Đánh giá **evasion robustness**: mô phỏng structuring, giãn thời gian, money mule trên test set và đo recall giảm bao
   nhiêu.
3. Kiểm chứng trên HI-Medium / LI-Small với một test set chưa từng xem.
4. Tune hyperparameter trên validation và thêm SHAP để giải thích từng alert.
