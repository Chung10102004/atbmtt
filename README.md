# Phát hiện rửa tiền bằng học máy – IBM AML (HI-Small)

Đề tài môn An toàn và bảo mật hệ thống thông tin. Câu hỏi nghiên cứu: **làm giàu dữ liệu train bằng LLM (ChatGPT API) có giúp mô hình phát hiện rửa tiền tốt hơn không, và có tốt hơn các cách tăng cường truyền thống (SMOTE, CTGAN) không?**

## 1. Đã làm gì

| Bước | Trạng thái | File |
|---|---|---|
| Tải dữ liệu IBM AML HI-Small | ✅ | `data/raw/HI-Small_Trans.csv` |
| Trích xuất 28 đặc trưng, chia train/val/test theo thời gian | ✅ | `src/data.py` |
| E0 – baseline XGBoost (đã chạy, xem mục 5) | ✅ | `src/model.py`, `run_experiments.py` |
| E1 – tăng cường bằng SMOTE (SMOTENC) | ✅ | `src/augment.py` |
| E2 – tăng cường bằng CTGAN | ✅ | `src/augment.py` |
| Đo chất lượng dữ liệu sinh (KS, bộ phân loại thật/giả) | ✅ | `src/augment.py` |
| Chạy trên Kaggle GPU qua Kaggle CLI | ✅ | `kaggle/build_kernel.py` |
| E3 – tăng cường bằng LLM (ChatGPT API) | 🟡 Code đã sẵn sàng, **chỉ cần điền API key** | `llm_config.py`, `src/llm_gen.py`, `generate_llm_data.py`, `run_llm.sh` |

## 2. Dữ liệu
[IBM Transactions for AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), file `HI-Small_Trans.csv`: 5.078.345 giao dịch tổng hợp (01–18/09/2022), 5.177 giao dịch rửa tiền (~0,1%).

Chia **theo thời gian** 60/20/20, không xáo trộn, để mô phỏng thực tế (train trên quá khứ, dự đoán tương lai) và tránh rò rỉ dữ liệu:

| Tập | Giai đoạn | Số giao dịch | Rửa tiền | Tỷ lệ |
|---|---|---|---|---|
| train | 01/09 → 06/09 13:36 | 3.047.007 | 2.298 | 0,075% |
| val | → 08/09 16:12 | 1.015.669 | 1.082 | 0,107% |
| test | → 18/09 | 1.015.669 | 1.797 | 0,177% |

## 3. Đặc trưng (28 cột, `src/data.py`)
- **Bản thân giao dịch (11):** log số tiền gửi/nhận, giờ, thứ, hình thức thanh toán, loại tiền gửi/nhận, cùng ngân hàng, tự chuyển cho chính mình, có đổi tiền tệ, số tiền tròn trăm.
- **Lịch sử tài khoản (16):** tính cho cả bên gửi (`src_`) và bên nhận (`dst_`) gồm số giao dịch vào/ra trước đó, số tiền trung bình vào/ra, số giây từ giao dịch trước, số giao dịch cùng ngày. Đây là đặc trưng **nhân quả**: chỉ dùng các giao dịch xảy ra *trước* giao dịch hiện tại, nên không nhìn trộm tương lai.
- **Cặp tài khoản (1):** số lần A→B đã giao dịch trước đó.

## 4. Thiết kế thí nghiệm
Giữ cố định:
- val/test là **dữ liệu thật, không bao giờ tăng cường**;
- cùng mô hình XGBoost với siêu tham số cố định: `lr=0.05, max_depth=8, subsample=0.8, colsample=0.8`, early stopping theo PR-AUC trên val;
- ngưỡng phân loại chọn trên val (max F1) rồi áp dụng nguyên cho test;
- 3 seed (0, 1, 2), báo cáo trung bình ± độ lệch chuẩn.

Chỉ thay đổi **tập train**:

| Mã | Tập train | Cách sinh dữ liệu |
|---|---|---|
| E0 | gốc | – |
| E1 | gốc + SMOTE | SMOTENC (k=5): nội suy giữa các giao dịch rửa tiền lân cận, cột phân loại lấy mode |
| E2 | gốc + CTGAN | GAN có điều kiện cho dữ liệu bảng, 300 epoch, chỉ học trên giao dịch rửa tiền |
| E3 | gốc + LLM | ChatGPT sinh giao dịch rửa tiền mới từ ví dụ thật, thống kê và mô tả các kiểu rửa tiền |

Số mẫu sinh thêm = `ratio × 2.298` (mặc định `ratio=1.0`, tức nhân đôi số mẫu rửa tiền). Cả 3 phương pháp đều sinh **trong cùng không gian đặc trưng** để so sánh công bằng.

**Metric** (lớp rửa tiền): PR-AUC (chính, phù hợp dữ liệu mất cân bằng), ROC-AUC, F1, Precision, Recall, Recall@FPR=1% (bắt được bao nhiêu % ca rửa tiền nếu chỉ chấp nhận 1% cảnh báo sai).

**Chất lượng dữ liệu sinh:**
- `ks_mean`: thống kê KS trung bình theo cột giữa dữ liệu sinh và dữ liệu rửa tiền thật, 0 nghĩa là phân phối giống hệt.
- `disc_auc`: AUC của một bộ XGBoost phân biệt thật và giả, 0,5 nghĩa là không phân biệt được (tốt), 1,0 nghĩa là dễ dàng phân biệt (kém).

## 5. Kết quả E0–E2
Chạy trên Kaggle GPU, 3 seed, `ratio=1.0` (+2.298 mẫu rửa tiền tổng hợp). Kết quả trên tập **test thật** (1.797 ca rửa tiền / 1.015.669 giao dịch). Dữ liệu gốc: `results/kaggle/results/results.csv`.

| Thí nghiệm | PR-AUC | ROC-AUC | F1 | Precision | Recall | Recall@FPR1% | ks_mean ↓ | disc_auc ↓ |
|---|---|---|---|---|---|---|---|---|
| **E0** gốc | **0,668 ± 0,005** | **0,988 ± 0,000** | **0,671 ± 0,009** | 0,776 ± 0,035 | 0,591 ± 0,012 | **0,838 ± 0,002** | – | – |
| E1 + SMOTE | 0,654 ± 0,016 | 0,987 ± 0,002 | 0,657 ± 0,015 | **0,845 ± 0,037** | 0,538 ± 0,026 | 0,827 ± 0,003 | 0,068 | 0,995 |
| E2 + CTGAN | 0,484 ± 0,130 | 0,983 ± 0,001 | 0,512 ± 0,093 | 0,455 ± 0,149 | **0,608 ± 0,003** | 0,803 ± 0,012 | 0,232 | 1,000 |

**Nhận xét:**
- **Không phương pháp tăng cường truyền thống nào cải thiện được baseline.** E0 có PR-AUC, F1 và Recall@FPR1% tốt nhất.
- **SMOTE (E1)** gần như không đổi về PR-AUC (−0,014, nằm trong khoảng dao động giữa các seed), nhưng làm mô hình "thận trọng" hơn: Precision tăng, Recall giảm. Dữ liệu SMOTE giống dữ liệu thật theo từng cột (KS 0,07), nhưng gần như phân biệt được hoàn toàn khi xét **kết hợp** các cột (disc_auc 0,995). Lý do là nội suy tuyến tính phá vỡ quan hệ giữa các đặc trưng, ví dụ đặc trưng đếm bị ra số lẻ.
- **CTGAN (E2)** làm mô hình kém đi rõ rệt và rất không ổn định (PR-AUC 0,41–0,63 tùy seed). Với chỉ 2.298 mẫu để học, GAN không mô phỏng được phân phối: KS 0,23, disc_auc 1,0. Precision giảm mạnh, tức mô hình học theo các mẫu giả không giống thật nên gắn cờ sai nhiều hơn. Ngoài ra `best_iter` ≈ 1000 (chạm trần), cho thấy mô hình mất nhiều vòng để "học thuộc" các mẫu giả.
- **Hàm ý cho E3:** để LLM có ích, dữ liệu nó sinh ra phải tốt hơn đáng kể về `disc_auc`/KS so với SMOTE và CTGAN. Đây là chỉ số cần theo dõi đầu tiên khi có dữ liệu LLM.

## 6. E3 – LLM (ChatGPT API)

### Cách hoạt động (`src/llm_gen.py`)
Mỗi lần gọi API:
1. Lấy ngẫu nhiên 20 giao dịch rửa tiền **thật** trong tập train làm ví dụ few-shot.
2. Prompt gồm mô tả 28 cột (kèm bảng mã hình thức thanh toán/tiền tệ), thống kê của lớp rửa tiền (p5/median/p95, tỷ lệ null, phân bố các cột phân loại), 7 kiểu rửa tiền của bộ IBM (fan-out, fan-in, cycle, scatter-gather, gather-scatter, bipartite/stack, random), và các ràng buộc logic (ví dụ `prev_n = prev_out + prev_in`).
3. Yêu cầu sinh **25 giao dịch mới** dạng JSON (`response_format=json_object`), mỗi dòng ghi kèm `typology` để phân tích sau.
4. Hậu xử lý:
   - bỏ dòng thiếu cột hoặc sai kiểu;
   - sửa các ràng buộc tất định (cross_currency, prev_n…);
   - bỏ dòng trùng lặp;
   - **bỏ dòng sao chép y nguyên dữ liệu thật** (kiểm tra mô hình có "học thuộc" không);
   - cắt giá trị về đúng miền của dữ liệu thật.

Dữ liệu được sinh **một lần** rồi lưu vào `data/llm/llm_seed{0,1,2}.csv`. Khi train, E3 chỉ đọc file này. Nhờ vậy có thể tái lập kết quả, không tốn API mỗi lần chạy, và API key không bao giờ bị đẩy lên Kaggle.

Với mỗi seed còn lưu:
- `llm_seed{s}_meta.json`: model, số lần gọi, số token, số dòng lỗi hoặc sao chép, phân bố typology;
- `sample_prompt_seed{s}.txt` và `sample_response_seed{s}.json`: prompt và phản hồi mẫu để đưa vào báo cáo.

Chi phí ước tính cho mỗi seed: khoảng 92 lần gọi × (~4,5k token vào + ~9k token ra). Với `gpt-4o-mini` chi phí rất nhỏ (cỡ vài chục cent cho cả 3 seed).

### Cách chạy
1. `cp llm_config.example.py llm_config.py` (nếu chưa có), rồi mở `llm_config.py`, điền key vào dòng `OPENAI_API_KEY = ""` (hoặc `export OPENAI_API_KEY=...`). Có thể đổi model qua `OPENAI_MODEL`.
2. Xem trước prompt (không gọi API, không tốn tiền):
   ```bash
   .venv/bin/python generate_llm_data.py --dry-run
   ```
3. Sinh dữ liệu và benchmark:
   ```bash
   ./run_llm.sh          # sinh dữ liệu local → chạy lại E0–E3 trên Kaggle GPU (khuyên dùng)
   ./run_llm.sh local    # sinh dữ liệu + chạy E3 ngay trên máy (cần ~16GB RAM)
   SEEDS="0" RATIO=0.5 ./run_llm.sh   # thử nhanh với 1 seed, 0,5× dữ liệu
   ```

⚠️ `llm_config.py` chứa API key: **không commit, không gửi cho người khác** (đã có trong `.gitignore`).

## 7. Chạy lại E0–E2
```bash
python3 -m venv .venv
.venv/bin/pip install pandas pyarrow scikit-learn xgboost imbalanced-learn ctgan tabulate kaggle openai

# Local (cần ~16GB RAM)
.venv/bin/python run_experiments.py --exps E0 E1 E2 --seeds 0 1 2 --ratio 1.0

# Kaggle (GPU, 30GB RAM) – dùng Kaggle CLI, cần ~/.kaggle token
.venv/bin/python kaggle/build_kernel.py --args "--exps E0 E1 E2 --seeds 0 1 2 --ratio 1.0"
.venv/bin/kaggle kernels push -p kaggle/kernel
.venv/bin/kaggle kernels status hoangducchung/aml-baseline-e0-e2
.venv/bin/kaggle kernels output hoangducchung/aml-baseline-e0-e2 -p results/kaggle
```
Kết quả: `results.csv` (từng lần chạy) và `summary.md` (trung bình ± độ lệch chuẩn).

## 8. Cấu trúc thư mục
```
llm_config.py          # cấu hình ChatGPT API (ĐIỀN KEY Ở ĐÂY)
generate_llm_data.py   # gọi API, sinh dữ liệu E3 → data/llm/
run_llm.sh             # chạy toàn bộ E3 (sinh dữ liệu + benchmark)
run_experiments.py     # chạy E0–E3, ghi results/
src/data.py            # đọc dữ liệu, đặc trưng, chia theo thời gian
src/augment.py         # SMOTE, CTGAN, LLM (đọc cache), đo chất lượng dữ liệu sinh
src/llm_gen.py         # prompt, gọi API, kiểm tra/làm sạch dữ liệu LLM sinh
src/model.py           # XGBoost + metric
kaggle/build_kernel.py # đóng gói code thành kernel Kaggle
```

## 9. Việc tiếp theo (gợi ý)
- **Ablation tỷ lệ sinh:** `--ratio 0.25 0.5 1 2`, vẽ PR-AUC theo tỷ lệ cho E1/E2/E3.
- **E4 – TSTR:** train chỉ trên dữ liệu sinh, test trên dữ liệu thật.
- **Phân tích theo typology:** dùng cột `typology` của dữ liệu LLM.
- **SHAP:** đặc trưng nào quan trọng, và có thay đổi sau khi tăng cường không.
