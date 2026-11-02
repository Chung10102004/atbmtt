"""Kịch bản thuyết trình: một nguồn duy nhất cho docs/kich-ban-thuyet-trinh.md và speaker notes của slide.

    .venv/bin/python slide/kich_ban.py

Sửa lời nói, thời lượng hoặc người nói trong SLIDES rồi chạy lại. Thứ tự SLIDES phải khớp thứ tự slide trong deck
(slide dự phòng cuối cùng không có trong SLIDES); script kiểm tra tiêu đề trước khi ghi.
"""
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "slide/AML_thuyet_trinh.pptx"
DOC = ROOT / "docs/kich-ban-thuyet-trinh.md"

SLIDES = [
 {
  "title": "Phát hiện rửa tiền bằng machine learning",
  "who": 1,
  "secs": 20,
  "goal": "Chào, giới thiệu nhóm và nêu câu hỏi trung tâm để người nghe biết cả bài đi về đâu.",
  "point": "Hình mạng tài khoản bên phải.",
  "stress": "Câu hỏi trung tâm: thêm dữ liệu hay thêm bối cảnh?",
  "say": "Kính chào thầy cô và các bạn. Nhóm em gồm Nguyễn Thế Vĩnh, Hoàng Đức Chung và Nguyễn Hải Đăng, xin trình bày đồ án Phát hiện rửa tiền bằng machine learning. Câu hỏi trung tâm của nhóm là: muốn phát hiện rửa tiền tốt hơn, nên thêm dữ liệu tổng hợp, hay cho model thấy bối cảnh hành vi của tài khoản? Như hình bên phải, tín hiệu nằm ở chuỗi giao dịch giữa các tài khoản, không ở một giao dịch đơn lẻ.",
  "nxt": "Bài trình bày gồm sáu phần."
 },
 {
  "title": "Nội dung trình bày",
  "who": 1,
  "secs": 20,
  "goal": "Cho người nghe bản đồ của bài nói và giới thiệu ai trình bày phần nào.",
  "point": "Lần lượt sáu thẻ, nhấn vào thẻ 3 và 4 (hai hướng A, B).",
  "stress": "Hai hướng A và B là trục chính của bài.",
  "say": "Đầu tiên là bài toán và dữ liệu. Phần chính là hai hướng cải thiện: hướng A thêm dữ liệu, hướng B thêm bối cảnh. Sau đó nhóm tự kiểm tra tính hợp lệ của đánh giá, liên hệ với an toàn bảo mật, và kết luận. Em trình bày phần một và hai, [tên người nói 2] trình bày hai hướng, và [tên người nói 3] trình bày phần còn lại.",
  "nxt": "Trước hết, vì sao đây là bài toán của môn an toàn thông tin?"
 },
 {
  "title": "Giám sát giao dịch hoạt động như một IDS cho dòng tiền",
  "who": 1,
  "secs": 50,
  "goal": "Kết nối đề tài với môn An toàn và bảo mật hệ thống thông tin: đây là một bài toán phát hiện.",
  "point": "Sơ đồ bốn khối hàng trên, rồi hàng IDS tương ứng bên dưới.",
  "stress": "Đây là bài toán phát hiện, giống phát hiện xâm nhập.",
  "say": "Rửa tiền là hợp thức hóa tiền bất hợp pháp qua chuỗi giao dịch để che giấu nguồn gốc. Ngân hàng có nghĩa vụ giám sát và báo cáo giao dịch đáng ngờ theo khuyến nghị FATF và Luật Phòng, chống rửa tiền 2022. Nhìn từ góc độ an toàn thông tin, hệ thống giám sát giao dịch hoạt động như một IDS: luồng giao dịch giống luồng gói tin, detector chấm điểm, chỉ một số alert đầu danh sách được xem, và chuyên viên điều tra đóng vai SOC analyst. Mỗi false positive tốn thời gian chuyên viên; mỗi false negative là một dòng tiền bẩn lọt qua.",
  "nxt": "Bài toán này có ba khó khăn khiến rất dễ đánh giá sai."
 },
 {
  "title": "Ba khó khăn khiến bài toán dễ bị đánh giá sai",
  "who": 1,
  "secs": 50,
  "goal": "Nêu ba bẫy của bài toán; mỗi bẫy dẫn tới một quyết định thiết kế của nhóm.",
  "point": "Ba con số lớn: 0,102%, AP 0,171, > 300.",
  "stress": "Accuracy 99,9% là vô nghĩa.",
  "say": "Thứ nhất, positive class cực hiếm: chỉ 0,102% giao dịch là rửa tiền. Đoán tất cả là bình thường đã có accuracy 99,9%, nên nhóm dùng Average Precision. Thứ hai, tín hiệu nằm ở bối cảnh: nếu chỉ nhìn thuộc tính của từng giao dịch, AP chỉ đạt 0,171. Thứ ba, dễ rò rỉ thời gian: timestamp chỉ đến phút, mỗi phút có hơn 300 giao dịch; feature vô tình dùng tương lai hoặc thứ tự dòng trong file sẽ cho điểm cao giả tạo. Ba khó khăn này dẫn tới ba quyết định của nhóm: dùng AP, xây feature hành vi, và time policy nghiêm ngặt.",
  "nxt": "Từ đó nhóm đặt câu hỏi nghiên cứu."
 },
 {
  "title": "Thêm dữ liệu hay thêm bối cảnh? Hai hướng, một protocol",
  "who": 1,
  "secs": 40,
  "goal": "Đưa ra khung của toàn bài: hai hướng cải thiện được so sánh trên cùng một protocol.",
  "point": "Hai cột Hướng A, Hướng B, rồi ô tối 'Protocol cố định'.",
  "stress": "So sánh có kiểm soát.",
  "say": "Hướng A bổ sung giao dịch rửa tiền tổng hợp vào train set: oversampling, SMOTENC, CTGAN và LLM. Hướng B mô tả hành vi theo thời gian của tài khoản bằng ba nhóm feature. Điểm quan trọng nhất là ô phía dưới: hai hướng được đánh giá trên cùng một protocol, cùng temporal split, cùng XGBoost cố định, cùng cách chọn threshold, ba seed. Nhờ vậy mọi khác biệt chỉ đến từ dữ liệu hoặc feature.",
  "nxt": "Nhóm xin nói rõ đóng góp của đồ án."
 },
 {
  "title": "Đóng góp: một phép so sánh có kiểm soát, không phải thuật toán mới",
  "who": 1,
  "secs": 50,
  "goal": "Định vị trung thực: không có thuật toán mới; đóng góp nằm ở thiết kế thí nghiệm và tính đúng của đánh giá.",
  "point": "Bốn đóng góp bên trái, rồi bảng công trình liên quan bên phải.",
  "stress": "Không phải thuật toán mới; là so sánh có kiểm soát.",
  "say": "Đồ án không đề xuất thuật toán mới. Đóng góp là bốn điểm bên trái: một bộ 63 feature tính nghiêm ngặt theo thời gian; ablation có kiểm soát với 5 cấu hình định trước; đánh giá 4 phương pháp augmentation, kể cả LLM, có chẩn đoán chất lượng dữ liệu; và phân tích tính hợp lệ của chính đánh giá. So với công trình liên quan: nhóm dùng lại dataset AMLworld của IBM; khác Graph Feature Preprocessor ở chỗ dùng feature cục bộ, dễ giải thích và time policy nghiêm ngặt hơn; khác Multi-GNN ở chỗ dùng XGBoost rẻ và giải thích được; và thử LLM sinh dữ liệu cho bài toán AML cực mất cân bằng.",
  "nxt": "Tiếp theo là dữ liệu."
 },
 {
  "title": "IBM AML HI-Small: 5,08 triệu giao dịch, chia theo thời gian",
  "who": 1,
  "secs": 50,
  "goal": "Giới thiệu dữ liệu và cách chia đúng theo thời gian, làm nền cho các con số phía sau.",
  "point": "Thanh Train / Val / Test và bảng chia tập.",
  "stress": "Chia theo thời gian, không chia ngẫu nhiên.",
  "say": "Dữ liệu là IBM AML HI-Small: hơn 5 triệu giao dịch mô phỏng từ ngày 1 đến 18 tháng 9 năm 2022, trong đó 5.177 giao dịch rửa tiền. Nhóm chia theo thời gian 60/20/20 như thanh màu phía trên, không chia ngẫu nhiên, vì thực tế model học từ quá khứ để dự đoán tương lai. Điểm cắt được dịch về đầu phút để một phút không bị tách sang hai tập. Nhãn của validation và test không tham gia tính feature, và ID tài khoản chỉ dùng để tra cứu lịch sử, không đưa vào model.",
  "nxt": "Tiếp theo, [tên người nói 2] sẽ trình bày thiết lập thực nghiệm và hai hướng cải thiện."
 },
 {
  "title": "Cùng model, cùng split, cùng cách chọn threshold",
  "who": 2,
  "secs": 45,
  "goal": "Chứng minh phép so sánh công bằng: mọi cấu hình dùng cùng model và cùng cách đánh giá.",
  "point": "Ba thẻ Model, Đánh giá, Augmentation.",
  "stress": "Không tuning riêng: điểm tăng đến từ feature hoặc dữ liệu.",
  "say": "Cảm ơn bạn. Mọi cấu hình dùng cùng một XGBoost, tối đa 1.000 cây, early stopping theo AUC-PR trên validation. Nhóm cố ý không tuning riêng từng cấu hình, để điểm tăng chắc chắn đến từ feature hoặc dữ liệu, không phải từ tuning. Chỉ số chính là AP; threshold chọn trên validation rồi giữ nguyên khi chạy test. Precision và Recall trong 1.000 alert cao nhất mô phỏng năng lực của chuyên viên. Với augmentation, dữ liệu tổng hợp chỉ thêm vào train, 1.148 mẫu mỗi lượt; validation và test luôn là dữ liệu thật.",
  "nxt": "Bắt đầu với hướng A: thêm dữ liệu. Trước hết là cách bốn phương pháp sinh dữ liệu."
 },
 {
  "title": "Bốn cách sinh thêm giao dịch rửa tiền cho train set",
  "who": 2,
  "secs": 50,
  "goal": "Giải thích cơ chế của bốn phương pháp augmentation trước khi xem kết quả, để người nghe hiểu vì sao kết quả khác nhau.",
  "point": "Lần lượt bốn sơ đồ nhỏ (xanh là dữ liệu thật, vàng là dữ liệu sinh ra), rồi dải ghi chú chung.",
  "stress": "Hai cách đầu bám sát dữ liệu thật; hai cách sau phải tự học hoặc tự đoán phân phối.",
  "say": "Cả bốn phương pháp đều sinh thêm giao dịch rửa tiền giả cho train set. Random oversampling chỉ nhân bản các ca thật, nên không tạo thông tin mới. SMOTENC chọn một ca thật và một ca gần nó, rồi tạo điểm nằm giữa hai ca. CTGAN là một mạng GAN: generator sinh dòng giả, discriminator phân biệt thật và giả, hai mạng học đối kháng; nhưng nó chỉ có 2.297 ca để học. Cuối cùng, LLM nhận prompt gồm ví dụ thật, thống kê và ràng buộc, rồi trả về dòng JSON; biến thể E3b thêm mô tả các kiểu rửa tiền. Mọi phương pháp chỉ thêm 1.148 mẫu vào train, còn validation và test luôn là dữ liệu thật.",
  "nxt": "Và đây là kết quả của hướng A."
 },
 {
  "title": "Dữ liệu tổng hợp không giúp; dữ liệu LLM làm AP giảm mạnh nhất",
  "who": 2,
  "secs": 80,
  "goal": "Trình bày kết quả âm của hướng A có giải thích nguyên nhân; đây là lý do chuyển sang hướng B.",
  "point": "Biểu đồ trái (ΔAP), rồi biểu đồ phải (discriminator AUC).",
  "stress": "Dữ liệu giả không giống thật nên làm model học sai.",
  "say": "Biểu đồ trái là AP thay đổi bao nhiêu so với không augmentation, mốc AP 0,670. SMOTENC tăng 0,003, nằm trong dao động giữa các seed, tức là không cải thiện thật. Oversampling và CTGAN giảm nhẹ. Hai biến thể LLM giảm mạnh nhất, khoảng 0,06 đến 0,07, kể cả khi đưa kiến thức về các kiểu rửa tiền vào prompt. Biểu đồ phải giải thích vì sao. Nhóm huấn luyện một classifier để phân biệt dữ liệu giả với dữ liệu thật: AUC 0,5 nghĩa là không phân biệt được. Với CTGAN và LLM, AUC từ 0,98 trở lên, tức dữ liệu giả lộ rõ là giả. Mẫu giả vẫn mang nhãn rửa tiền, nên model học những vùng không tồn tại trong thực tế và bỏ sót ca thật. Lưu ý thí nghiệm này dùng pipeline cũ, nên chỉ so các phương pháp với nhau.",
  "nxt": "Vì thêm dữ liệu không giúp, nhóm chuyển sang hướng B: thêm bối cảnh."
 },
 {
  "title": "Chỉ nhìn quá khứ: lịch sử nghiêm ngặt theo timestamp",
  "who": 2,
  "secs": 50,
  "goal": "Trình bày nền tảng tính đúng của hướng B: không nhìn trước tương lai, không phụ thuộc thứ tự dòng.",
  "point": "Công thức H(t), rồi hình timeline (cùng phút chung snapshot, tương lai không dùng).",
  "stress": "Không nhìn trước tương lai, không phụ thuộc thứ tự dòng.",
  "say": "Nền tảng của hướng B là time policy. Lịch sử của giao dịch tại thời điểm t chỉ gồm các giao dịch có timestamp nhỏ hơn t. Vì timestamp chỉ đến phút và không biết thứ tự thật trong cùng phút, mọi giao dịch cùng phút dùng chung một snapshot lịch sử: như hình, các giao dịch tại t không thấy nhau, và tương lai không bao giờ được dùng. Cửa sổ thời gian là từ t trừ W đến t, với W là 1 giờ và 24 giờ. Tổng tiền tách theo currency vì không có tỷ giá. Có 43 regression test kiểm tra đúng các quy tắc này.",
  "nxt": "Trên nền đó, nhóm xây 63 feature."
 },
 {
  "title": "63 feature hành vi chia thành bốn nhóm",
  "who": 2,
  "secs": 50,
  "goal": "Giải thích bốn nhóm feature bằng ví dụ dễ hiểu và cách thiết kế 5 cấu hình F0–F4.",
  "point": "Bốn thẻ màu, rồi bảng cấu hình F0–F4.",
  "stress": "Mỗi nhóm một ví dụ đời thường.",
  "say": "63 feature chia bốn nhóm. Transaction là thuộc tính của chính giao dịch. History là lịch sử tài khoản, ví dụ tài khoản đã nhận bao nhiêu lần, giao dịch gần nhất cách bao lâu. Activity là nhịp độ trong 1 giờ và 24 giờ, và số tiền hiện tại lệch bao nhiêu so với thói quen. Relationship là quan hệ đối tác, ví dụ cặp tài khoản đã giao dịch với nhau bao lâu, có bao nhiêu đối tác khác nhau. Bảng phía dưới là 5 cấu hình F0 đến F4 định trước khi chạy, để mỗi hiệu số đo đúng đóng góp của một nhóm.",
  "nxt": "Phần kết quả, [tên người nói 3] sẽ trình bày."
 },
 {
  "title": "Lịch sử tạo bước nhảy lớn nhất; kết hợp ba nhóm cho AP cao nhất",
  "who": 3,
  "secs": 80,
  "goal": "Kết quả chính của đồ án: thêm bối cảnh hành vi làm AP tăng mạnh và nhất quán.",
  "point": "Các cột F0 → F4 ở biểu đồ trái, rồi dòng ΔAP phía dưới.",
  "stress": "History tạo bước nhảy lớn nhất; F4 đạt AP 0,684; 3/3 seed đều tăng.",
  "say": "Cảm ơn bạn. Đây là kết quả chính. Chỉ dùng thuộc tính giao dịch, F0, AP là 0,171. Thêm History, AP nhảy lên 0,534, đây là bước nhảy lớn nhất. Khi đã có History, Activity đóng góp nhiều hơn Relationship: 0,662 so với 0,610. Kết hợp cả ba nhóm, F4 đạt 0,684, gấp bốn lần F0 với cùng một model. Recall@1000 bên phải có cùng xu hướng. Dòng dưới cùng là hiệu số theo cặp cùng seed: mọi bước bổ sung đều dương ở cả ba seed, kể cả F4 so với F2, nghĩa là Relationship vẫn mang thông tin bổ sung. Ba seed cho thấy tính nhất quán, nhưng chưa đủ để khẳng định ý nghĩa thống kê.",
  "nxt": "Vậy con số này có ý nghĩa gì với một ngân hàng?"
 },
 {
  "title": "Cùng 1.000 cảnh báo, F4 tìm thêm khoảng 200 ca rửa tiền",
  "who": 3,
  "secs": 50,
  "goal": "Dịch AP sang ngôn ngữ vận hành, nêu đánh đổi precision/recall và diễn giải feature quan trọng.",
  "point": "Cột F1 và F4 ở biểu đồ trái, rồi top feature bên phải.",
  "stress": "Cùng khối lượng công việc, bắt thêm khoảng 200 ca; nhưng vẫn sót gần một nửa.",
  "say": "Giả sử chuyên viên chỉ xử lý được 1.000 alert. Với F1, khoảng 712 alert là rửa tiền thật; với F4 là 915, precision 91,5%. Cùng khối lượng công việc, tìm thêm khoảng 200 ca. Nhưng recall chỉ 50,9%: vẫn còn 883 ca nằm ngoài danh sách, muốn bắt nhiều hơn phải chấp nhận nhiều alert hơn. Bên phải là các feature quan trọng nhất: số lượt nhận trước đó của sender, thời gian từ giao dịch gần nhất của receiver, độ lệch số tiền so với lịch sử. Đó là dấu hiệu money mule: tài khoản vừa nhận tiền rồi chuyển đi với số tiền bất thường. Gain importance không phải quan hệ nhân quả, nhưng giúp giải thích alert cho chuyên viên.",
  "nxt": "Một kết quả tốt chỉ có giá trị khi cách đánh giá đúng, nên nhóm tự kiểm tra lại."
 },
 {
  "title": "Kiểm tra lại tính hợp lệ của chính đánh giá",
  "who": 3,
  "secs": 70,
  "goal": "Thể hiện tư duy phản biện: chủ động nêu điểm yếu của đánh giá trước khi bị hỏi.",
  "point": "Hai biểu đồ theo ngày, rồi ba mục a, b, c.",
  "stress": "Nhóm tự tìm ra điểm yếu của chính đánh giá.",
  "say": "Có ba điểm. Một, distribution shift: từ ngày 11 tháng 9, lưu lượng giảm từ hơn 200 nghìn xuống vài trăm giao dịch mỗi ngày, còn tỷ lệ rửa tiền vọt lên gần 60%. Phát hiện này đến sau khi chạy, nên nhóm không xóa dữ liệu mà tách riêng để kiểm tra: trên phần trước 11 tháng 9, F4 vẫn hơn F1 rõ rệt, 0,528 so với 0,333. Hai, time policy ảnh hưởng mạnh: pipeline cũ xử lý cùng phút theo thứ tự dòng đạt AP 0,670, trong khi pipeline nghiêm ngặt với 28 feature tương ứng chỉ 0,534; đây không phải so sánh có kiểm soát, nhưng cho thấy cách xử lý thời gian có thể thổi phồng điểm số. Ba, test set đã được dùng trong thí nghiệm trước, nên kết quả mang tính khám phá.",
  "nxt": "Những điều này liên hệ trực tiếp với môn an toàn bảo mật."
 },
 {
  "title": "Detector là thành phần phòng thủ, và cũng có bề mặt tấn công",
  "who": 3,
  "secs": 50,
  "goal": "Trả lời trực tiếp câu hỏi 'đồ án liên quan gì đến môn An toàn và bảo mật hệ thống thông tin'.",
  "point": "Bốn thẻ theo thứ tự: phòng thủ, leakage, evasion, bảo vệ dữ liệu.",
  "stress": "Leakage là lỗ hổng phương pháp; evasion là bề mặt tấn công.",
  "say": "Có bốn ý. Một, detector là thành phần phòng thủ: hiệu quả đo bằng số ca tìm được trong alert budget, và feature hành vi giải thích được bằng ngôn ngữ nghiệp vụ. Hai, data leakage là một lỗ hổng phương pháp: hệ thống có điểm cao, trông an toàn, nhưng thực ra không. Ba, chính feature hành vi tạo ra bề mặt tấn công: kẻ rửa tiền có thể chia nhỏ giao dịch, giãn thời gian để thoát cửa sổ, hoặc dùng money mule; đây là evasion attack mà nhóm chưa đo được. Bốn, bảo vệ dữ liệu và bí mật: account ID không phải feature, repository không chứa API key hay dữ liệu thô.",
  "nxt": "Cuối cùng là giới hạn và hướng phát triển."
 },
 {
  "title": "Giới hạn và hướng phát triển",
  "who": 3,
  "secs": 40,
  "goal": "Nói rõ phạm vi của kết luận và chỉ ra bước tiếp theo.",
  "point": "Cột Giới hạn, rồi cột Hướng phát triển.",
  "stress": "Kết quả có giá trị so sánh tương đối.",
  "say": "Dữ liệu là mô phỏng; test set đã được xem trước; ba seed chưa đủ cho ý nghĩa thống kê; hyperparameter cố định; feature quan hệ mới là cục bộ. Vì vậy kết quả có giá trị so sánh tương đối giữa các phương pháp, chưa khẳng định hiệu năng ở ngân hàng thật. Hướng phát triển: so sánh với GFP dưới cùng time policy, phân tích theo tám kiểu rửa tiền, đánh giá khả năng chống né, và kiểm chứng trên dataset khác.",
  "nxt": "Xin tóm lại."
 },
 {
  "title": "Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn bổ sung dữ liệu giả.",
  "who": 3,
  "secs": 30,
  "goal": "Chốt thông điệp một câu và ba con số, mời hội đồng đặt câu hỏi.",
  "point": "Ba con số lớn.",
  "stress": "Thông điệp một câu.",
  "say": "Với dữ liệu giao dịch, bối cảnh hành vi và đánh giá đúng thời gian quan trọng hơn thêm dữ liệu giả. Ba con số: feature hành vi nâng AP thêm 0,513; precision 91,5% trong 1.000 alert; còn dữ liệu do CTGAN và LLM sinh ra bị phân biệt với dữ liệu thật ở mức AUC từ 0,98. Toàn bộ mã nguồn và số liệu có trên GitHub. Nhóm em xin cảm ơn thầy cô và các bạn đã lắng nghe, và sẵn sàng nhận câu hỏi.",
  "nxt": "(Hỏi đáp; mở slide dự phòng khi được hỏi số liệu chi tiết.)"
 }
]


def mm(s):
    return f"{s // 60}:{s % 60:02d}"


def dur(s):
    if s < 60:
        return f"~{s} giây"
    return f"~{s // 60} phút {s % 60} giây" if s % 60 else f"~{s // 60} phút"


def write_notes():
    prs = Presentation(DECK)
    slides = list(prs.slides)
    assert len(slides) == len(SLIDES) + 1, (len(slides), len(SLIDES))
    for i, d in enumerate(SLIDES):
        texts = [sh.text_frame.text.replace("\n", " ").replace("\v", " ") for sh in slides[i].shapes if sh.has_text_frame]
        assert any(t.startswith(d["title"][:30]) for t in texts), (i + 1, d["title"][:50])
        slides[i].notes_slide.notes_text_frame.text = "\n".join([
            f"[Người nói {d['who']} · {dur(d['secs'])}]",
            f"MỤC ĐÍCH: {d['goal']}",
            f"CHỈ VÀO: {d['point']}",
            f"NÓI: {d['say']}",
            f"NHẤN MẠNH: {d['stress']}",
            f"CHUYỂN: {d['nxt']}",
        ])
    prs.save(DECK)


def write_doc():
    pos = lambda prefix: next(i for i, d in enumerate(SLIDES, 1) if d["title"].startswith(prefix))
    end_at = lambda n: sum(d["secs"] for d in SLIDES[:n])
    s_data, s_abl = pos("IBM AML HI-Small"), pos("Lịch sử tạo bước nhảy")
    s_a, s_val, s_sec = pos("Dữ liệu tổng hợp không giúp"), pos("Kiểm tra lại tính hợp lệ"), pos("Detector là thành phần")
    total = end_at(len(SLIDES))
    who = sorted({d["who"] for d in SLIDES})
    part = {w: [i for i, d in enumerate(SLIDES, 1) if d["who"] == w] for w in who}
    rows, t = [], 0
    for i, d in enumerate(SLIDES, 1):
        t += d["secs"]
        rows.append(f"| {i} | {d['title']} | {d['who']} | {mm(d['secs'])} | {mm(t)} |")
    out = [f"""# Kịch bản thuyết trình: Phát hiện rửa tiền bằng machine learning

Kịch bản cho bộ slide [slide/AML_thuyet_trinh.pptx](../slide/AML_thuyet_trinh.pptx). Cùng nội dung đã được ghi vào
speaker notes của từng slide (bật Presenter View để xem khi nói). Câu hỏi có thể gặp và gợi ý trả lời nằm ở
[thuyet-trinh-va-cau-hoi.md](thuyet-trinh-va-cau-hoi.md). File này được sinh bởi `slide/kich_ban.py`; sửa nội dung ở đó
rồi chạy lại.

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

Tổng thời lượng **{mm(total)}**, trong giới hạn 15 phút.

| Người nói | Họ tên (nhóm tự điền) | Slide | Thời lượng |
|---|---|---|---|
""" + "\n".join(f"| {w} | … | {part[w][0]}–{part[w][-1]} | {mm(sum(SLIDES[i - 1]['secs'] for i in part[w]))} |" for w in who) + f"""

Thay `[tên người nói 2]`, `[tên người nói 3]` trong lời nói bằng tên thật. Nên phân người theo phần mình đã trực tiếp
làm để trả lời câu hỏi sâu được.

**Mốc kiểm tra thời gian** (cột "Kết thúc lúc"): nếu đến cuối slide {s_data} đã quá {mm(end_at(s_data) + 35)}, hoặc cuối slide
{s_abl} đã quá {mm(end_at(s_abl) + 45)}, hãy rút gọn các slide sau theo mục 4.

| # | Slide | Người nói | Thời lượng | Kết thúc lúc |
|---|---|---|---|---|
""" + "\n".join(rows) + f"""
| {len(SLIDES) + 1} | Bảng số liệu đầy đủ (dự phòng) | – | – | chỉ mở khi được hỏi |

## 3. Kịch bản từng slide
"""]
    for i, d in enumerate(SLIDES, 1):
        out.append(f"""
### Slide {i}. {d['title']}
*Người nói {d['who']} · {dur(d['secs'])}*

- **Mục đích:** {d['goal']}
- **Chỉ vào:** {d['point']}
- **Nhấn mạnh:** {d['stress']}

> {d['say']}

**Chuyển:** {d['nxt']}
""")
    out.append(f"""
## 4. Mẹo khi nói

- **Nói theo ý, không đọc thuộc lòng.** Mỗi slide chỉ cần nhớ mục đích, chỗ cần chỉ và câu nhấn mạnh; lời nói ở trên
  là gợi ý.
- **Chỉ vào slide khi nói con số** để hội đồng nhìn đúng chỗ.
- **Câu chuyển là chỗ hay bị quên.** Đặc biệt câu chuyển ở slide {s_a} ("vì thêm dữ liệu không giúp, nhóm chuyển sang
  hướng B") là lý do của cả phần sau.
- **Nếu bị lố giờ:** bỏ slide 2 (nội dung) và slide 6 (đóng góp); ở slide {s_val} chỉ nói ý (a); slide {s_sec} nói trong 30 giây.
- **Nếu bị hỏi giữa chừng:** trả lời ngắn một câu, nói "nhóm sẽ trình bày kỹ ở slide …" nếu câu hỏi thuộc phần sau,
  rồi quay lại mạch.
- **Khi bị hỏi số liệu chi tiết:** mở slide {len(SLIDES) + 1} (dự phòng).
- **Nếu bị hỏi "so với các paper khác thì sao?":** xem câu A4 trong [thuyet-trinh-va-cau-hoi.md](thuyet-trinh-va-cau-hoi.md).
""")
    DOC.write_text("".join(out), encoding="utf-8")
    return total, {w: mm(sum(SLIDES[i - 1]["secs"] for i in part[w])) for w in who}


if __name__ == "__main__":
    write_notes()
    print("total", *write_doc())
