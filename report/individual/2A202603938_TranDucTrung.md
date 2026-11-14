# Báo cáo cá nhân — Day 06: Data Ingestion Engineering

**Học viên:** Trần Đức Trung · **MSSV:** 2A202603938 · **Lớp:** L3B  
**Repository:** [K5-L3B-DAY06-TranDucTrung-2A202603938-DataIngestion](https://github.com/kientrungtd135/K5-L3B-DAY06-TranDucTrung-2A202603938-DataIngestion)  
**Ngày lập báo cáo:** 2026-11-14  
**Commit chốt bài:** Chưa có. Working tree còn các thay đổi chưa commit; không dùng commit hiện tại làm commit hoàn chỉnh của bài.

## 1. Dữ liệu và pipeline

### Corpus, nguồn và quyền sử dụng

Corpus là snapshot 30 bài báo Markdown độc lập, gồm 15 bài `business` và 15 bài `technology`, lấy từ [MultiHopRAG trên Hugging Face](https://huggingface.co/datasets/yixuantt/MultiHopRAG). Mỗi file Markdown có một body bài viết, một `source_id` riêng và một `unit_id` dạng `doc_xxxx:unit_1`. Manifest ghi 30 nguồn, 30 đơn vị, tổng 247.941 ký tự body đã chuẩn hóa; độ dài đơn vị từ 5.445 đến 21.897 ký tự. Vì vậy đây là **30 đơn vị bài báo**, không phải 30 trang PDF. `page=1` biểu diễn đơn vị bài viết để truy vết.

Manifest nằm ở `starter/data/manifest.json`; corpus được giữ tại `starter/data/corpus/`. Manifest ghi URL nguồn, publisher, ID, category, local path, ranh giới dòng, số ký tự và nguồn dataset. ODC-BY là license **do dataset MultiHopRAG công bố**; điều đó không xác lập license hay quyền tái phân phối riêng của nội dung từng bài báo. Repo không có minh chứng cấp quyền từ từng nhà xuất bản. Cần xác minh quyền áp dụng trước khi tiếp tục phân phối nội dung; việc lưu URL, publisher và attribution phù hợp không tự nó chứng minh đã có quyền. `accessed_at` trong manifest là thời điểm đọc snapshot cục bộ; thời điểm truy cập web ban đầu được ghi là không biết.

### Chuẩn hóa, truy vết và lỗi nguồn

`load_markdown()` đọc các bài được manifest liệt kê, kiểm tra header, category và body, rồi gắn `document_id`, URL `source`, `page=1`, `unit_id` và metadata. Không quét các dữ liệu ngoài danh sách manifest. Chuẩn hóa dùng Unicode NFC, chuẩn hóa line ending và khoảng trắng ngang, giữ ranh giới đoạn, gộp các dòng trống liên tiếp và bỏ khoảng trắng ngoài body. Kiểm tra snapshot hiện có 30 file Markdown, không thấy ký tự thay thế Unicode U+FFFD. `starter/results/errors.log` không ghi lỗi đọc/parse nguồn; các dòng hiện có là lưu ý về đo thời gian, usage và chi phí.

Không thể coi việc không có lỗi parse là bảo đảm nội dung nguồn hoàn hảo. Ví dụ `doc_0030` có đoạn/list ghép không thuận lợi cho chunking; vấn đề này thể hiện trong audit coherence. Các nguồn là tập nhỏ, chủ đề bài báo khác nhau và snapshot có thể không còn khớp nội dung web hiện tại.

### Embedding, vector store và metadata

- Provider: CKEY qua endpoint tương thích OpenAI `https://api.xah.io/v1`.
- Model được ghi trong artifact: `text-embedding-3-small`, vector 1536 chiều. Artifact không ghi phiên bản bất biến riêng của model.
- Vector store: PostgreSQL 16.16 + pgvector 0.8.7, bảng `public.day6_chunks`.
- Metric: cosine distance; truy vấn exact, không dùng approximate vector index.
- Metadata gồm `document_id`, URL `source`, `page`, `unit_id` và `category`. `category` lấy từ metadata bài báo/thư mục corpus, có hai giá trị business và technology; đây là trường filter.
- Môi trường và thư viện được ghi trong config: Python 3.13.3, OpenAI SDK 2.54.0, psycopg 3.3.8, pypdf 6.21.4, python-dotenv 1.2.4.

## 2. Cấu hình và benchmark

Ba cấu hình dùng chung corpus, model, vector store và cosine metric:

| Strategy | Chunking | Filter | Giả thuyết |
|---|---|---|---|
| A | Fixed-size, `size=500`, `overlap=50` | Không | Cửa sổ chồng lấp có thể giữ đủ ngữ cảnh cho evidence. |
| B | Theo đoạn, `max_size=500` | Không | Gom đoạn trong giới hạn có thể giữ evidence mạch lạc hơn. |
| C | Chính xác chunks/metadata/vectors của B | Chỉ category được câu hỏi yêu cầu | Filter category có thể thu hẹp phạm vi với câu hỏi có điều kiện metadata. |

B–C chỉ khác việc áp dụng filter; C không có chunking hoặc document embedding độc lập. C tái sử dụng vectors của B.

Benchmark `benchmark/questions.json` có 20 câu, cố định 15 dev (`q01`–`q15`) và 5 test (`q16`–`q20`) trước retrieval; ba câu có filter, gồm hai câu dev và một câu test. Các gold document tách giữa hai split nhưng chủ đề vẫn có thể giao nhau. Trước chạy, evidence được đối chiếu với nội dung nguồn; query chỉ dùng câu hỏi, không chứa expected answer, gold source hay evidence. Với Markdown, `gold_page=1` là đơn vị bài báo.

Tiêu chí chọn được chốt trong config trước khi chạy TEST: ưu tiên Evidence Hit@3 dev, rồi Hit@1, rồi ít chunks lưu hơn; nếu vẫn hòa thì thứ tự A, B, C. DEV chọn B: B và C hòa ở Hit@3, Hit@1 và số chunks; thứ tự phá hòa chọn B. TEST không làm thay đổi lựa chọn.

Mẫu coherence cũng được chốt trong config trước khi xem điểm: lấy 5 `document_id` đầu theo thứ tự từ điển của mỗi category, cùng body offset `[0, 500)` cho mọi strategy. Các mẫu là business `doc_0002`, `doc_0003`, `doc_0028`, `doc_0029`, `doc_0030` và technology `doc_0010`, `doc_0011`, `doc_0012`, `doc_0015`, `doc_0016`. Mỗi strategy có 10 dòng audit.

## 3. Kết quả

### Retrieval accuracy

Định nghĩa hit yêu cầu chunk đúng nguồn/trang và chứa đầy đủ evidence. Bảng dưới đây dùng tử số/mẫu số từ `metrics_dev.json` và `metrics_test.json`.

| Strategy | Chunks | DEV Hit@1 | DEV Evidence Hit@3 | TEST Hit@1 | TEST Evidence Hit@3 |
|---|---:|---:|---:|---:|---:|
| A | 562 | 9/15 (60,0%) | 11/15 (73,3%) | 4/5 (80,0%) | 5/5 (100%) |
| B | 701 | 14/15 (93,3%) | 15/15 (100%) | 4/5 (80,0%) | 5/5 (100%) |
| C | 701 | 14/15 (93,3%) | 15/15 (100%) | 4/5 (80,0%) | 5/5 (100%) |

### Ba khía cạnh audit thủ công

| Strategy | Coherence: điểm TB /2 (n=10) | Coherence: số mức 0/1/2 | Grounding: số mức 0/1/2 (n=5), TB /2 | Metadata đúng/đủ | Filter tuân thủ |
|---|---:|---|---|---:|---:|
| A | 1,00 | 0 / 10 / 0 | 0 / 0 / 5; 2,00 | 15/15 | 1/1 câu có filter |
| B | 1,40 | 0 / 6 / 4 | 0 / 0 / 5; 2,00 | 15/15 | 1/1 câu có filter |
| C | 1,40 | 0 / 6 / 4 | 0 / 0 / 5; 2,00 | 15/15 | 1/1 câu có filter |

Metadata được kiểm tra trên 5 câu test × top-3 = 15 dòng mỗi strategy. Bốn câu không có filter được ghi N/A cho đánh giá filter; chỉ q20 có filter. Với q20, A/B không áp dụng filter truy vấn nhưng cả ba kết quả đều có category=technology; C áp dụng filter technology. Vì vậy cả ba đạt tuân thủ 1/1 câu. Metadata đúng và kết quả tình cờ phù hợp filter không chứng minh rằng bật filter đã cải thiện chất lượng.

Grounding đo độ đầy đủ của bằng chứng truy xuất, không đo độ chính xác của câu trả lời do một LLM sinh ra. Điểm 2 ở cả năm câu test không loại bỏ các lỗi xếp hạng top-1 hoặc vấn đề coherence ở những vị trí nguồn khác.

### Ví dụ lỗi và phân tích

1. **`q02`, A, DEV — evidence bị cắt giữa chunk.** Câu hỏi hỏi vì sao Ripple Rug mất vị trí buy box; gold là `doc_0003`, bài CNBC “Amazon sellers sound off on the FTC's 'long-overdue' antitrust case” (`https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html`). Trace A có Hit@1=0 và Hit@3=0. Top-1 bắt đầu ở phần đuôi “price. His product…”; top-3 chứa phần đầu câu “Amazon’s software had detected…” nhưng kết thúc sau “would no longer be shown in Am…”. Không chunk nào trong top-3 chứa trọn evidence quote. Đây là lỗi truy xuất bằng chứng thực tế, phù hợp với rủi ro cửa sổ fixed-size cắt câu qua ranh giới chunk.
2. **`q16`, B, TEST — đúng evidence nhưng không ở hạng 1.** Câu hỏi hỏi số người dự lễ hội AfroFuture; nguồn `doc_0086`, Music Business Worldwide (`https://www.musicbusinessworldwide.com/afrofutures-vision-is-to-bring-african-art-cuisine-fashion-and-music-to-a-global-audience/`). Top-1 B có ngày và địa điểm nhưng không có số người, nên Hit@1 sai. Top-2 có câu đầy đủ “attracting 4,600 attendees”, nên Evidence Hit@3 đúng. Đây là lỗi xếp hạng top-1, không phải thiếu bằng chứng trong top-3.
3. **`business_doc_0030_0_500`, A/B/C — coherence mức 1.** Audit của `doc_0030` (bài The Age “ASX set to open higher as Wall Street rebounds; $A rises”, file `starter/data/corpus/business/doc_0030_ASX_set_to_open_higher_as_Wall_Street_rebounds_A_rises.md`) ghi A cắt từ “Santos” thành “San” và nối đột ngột “The laggards Meanwhile”; B/C kết thúc đoạn danh sách ở “Woodside and”. Cùng một bài và vị trí cho thấy giới hạn có thể đến từ ranh giới câu/đoạn nguồn hoặc giới hạn kích thước; theo audit, paragraph chunking không tự động bảo đảm chunk luôn trọn ý.

Các trace đầy đủ và lý do audit nằm trong các file được liệt kê ở phần minh chứng dưới đây.

## 4. Tác động chiến lược, tài nguyên và đánh đổi

### A–B

B tăng 139 chunks so với A (A ít hơn khoảng 19,8% so với B), nhưng DEV tăng từ 9/15 lên 14/15 ở Hit@1 và từ 11/15 lên 15/15 ở Evidence Hit@3. Coherence trung bình của audit B cao hơn A (1,40 so với 1,00), nhưng vẫn có sáu vị trí B chỉ đạt mức 1; bốn vị trí còn lại đạt mức 2. Trên TEST, hai chiến lược hòa ở cả hai chỉ số retrieval. Vì TEST chỉ có năm câu, kết quả không chứng minh B tốt hơn tổng quát.

A có mean chunk length 488,5 ký tự và tổng chunk text 274.541 ký tự; overlap làm tổng này lớn hơn 247.941 ký tự nội dung nguồn. B/C có mean 352,3, median 397, p95 499, min 5, max 500 ký tự; tổng chunk text 246.977 ký tự. Độ dài tối đa được giữ trong giới hạn cấu hình, nhưng chunk rất ngắn và các đoạn dài bị chia vẫn có thể gây lỗi coherence.

### B–C

B/C có cùng 701 chunks, metadata và document vectors; metrics DEV/TEST giống nhau. Filter chỉ có cơ hội tác động ở ba câu benchmark có category filter, trong đó TEST chỉ có q20. A/B cũng trả về kết quả category=technology cho q20 dù không áp dụng filter. Tập đo này do đó không cho thấy lợi ích định lượng riêng của filter, và cũng quá nhỏ để kết luận filter không hữu ích trong phạm vi khác.

### Thời gian, API usage và chi phí

Các thời gian dưới đây là số đo trong artifacts; C dùng lại chunks/vectors của B, nên thời gian tạo chunk và embedding của C là **chi phí tăng thêm bằng 0 trong lượt chạy này**, không phải phép đo một lần chạy C độc lập.

| Giai đoạn | Số đo đã lưu |
|---|---|
| DEV: chunk generation | A 0,007 giây; B 0,011 giây; C 0 giây tăng thêm do dùng lại B |
| DEV: document embedding | A 14,242 giây; B 38,950 giây; C 0 giây tăng thêm do dùng lại vectors B |
| DEV: query embedding dùng chung | 15 request, 343 tokens, 10,272 giây |
| DEV: upsert ban đầu / nạp lại idempotent | A 1,086 / 0,774 giây; B 1,175 / 1,008 giây; C 0,967 / 0,907 giây |
| DEV: retrieval | A 0,449 giây; B 0,454 giây; C 0,424 giây |
| DEV tổng đến retrieval và selection | 74,458 giây |
| TEST: query embedding dùng chung | 5 request, 118 tokens, 5,615 giây |
| TEST: retrieval | A 0,169 giây; B 0,374 giây; C 0,189 giây |
| TEST tổng | 7,574 giây, gồm preflight, kiểm tra read-only index, query embedding, retrieval và xuất audit |

DEV ghi tổng 27 lệnh gọi `embeddings.create`, 109.345 tokens và chi phí **ước tính** USD 0,0088043. Tổng gồm dimension probe, document embedding A/B và 15 query embeddings DEV; C không có document embedding riêng. TEST ghi 5 lệnh gọi, 118 tokens, chi phí **ước tính** USD 0,0012; không ingest hay embed corpus. Cộng riêng những gì hai metrics artifact ghi nhận: 32 lệnh gọi, 109.463 tokens và USD 0,0100043 ước tính. Phép cộng này chỉ bao gồm hai pha DEV/TEST đã lưu, không bao gồm các API call thăm dò/kiểm tra trước những pha đó.

Ước tính dùng bảng giá CKEY được ghi trong config: USD 0,0448/triệu input tokens, tối thiểu USD 0,00024 mỗi response; tính tổng theo từng response lấy giá token hoặc mức tối thiểu, tùy số nào lớn hơn. Đây **không phải billing thực tế**; không có hóa đơn provider trong artifacts. Số request là số lần gọi qua SDK, không bao gồm retries HTTP nội bộ mà SDK không phơi bày. Không cộng các thời gian trên thành “thời gian toàn bộ lab”; thời gian DEV và TEST có phạm vi đo khác nhau.

Lần nạp lại giống hệt không tăng số dòng: A giữ 562, B 701, C 701. Kiểm tra cũng ghi nhận không có dòng ngoài corpus cần prune. Đây là bằng chứng idempotence cho lần kiểm tra đã lưu, không đảm bảo mọi thay đổi corpus khác đều được xử lý như nhau.

## 5. Phương pháp audit và giới hạn

ChatGPT và GitHub Copilot trong VS Code hỗ trợ đọc chunk, source context, trace, manifest và tiêu chí GUIDE để đề xuất điểm/lý do. Người học xác nhận 12 nhãn coherence đầu theo từng sample; các nhãn coherence còn lại, grounding và metadata do AI đề xuất, sau đó người học duyệt tổng hợp. Vì vậy đây không phải đánh giá độc lập/blind, cũng không phải toàn bộ nhãn do người học tự chấm. CSV giữ `review_status=human_confirmed` sau khi người học phê duyệt; trạng thái này không có nghĩa mọi điểm ban đầu đều do người học tự tạo. Các trường `reason` trong ba CSV audit được biên tập để sửa lỗi ký tự; điểm, nhãn và kết quả thực nghiệm không thay đổi.

Không có bằng chứng đủ để xác định model judge, SDK hoặc phiên bản cụ thể nên không nêu các thông tin đó. Nguy cơ gồm automation/confirmation bias, AI có thể bỏ sót ngữ cảnh hoặc diễn giải thiên lệch, người học có thể dễ chấp nhận lý do trôi chảy, và không có chấm độc lập thứ hai để đo inter-rater agreement. Mẫu coherence chỉ lấy phần đầu 500 ký tự của 10 bài; grounding/metadata chỉ có năm câu test. Các điểm audit vì thế chỉ mô tả các mẫu đã duyệt.

Grounding cao vẫn có thể đi cùng coherence không cao: grounding chấm liệu top-3 của năm câu test có bằng chứng đủ và truy nguồn được hay không; coherence chấm chunk tại 10 vị trí cố định khác trong corpus. Evidence hoàn chỉnh có thể xuất hiện ở một chunk được truy xuất trong khi nhiều chunk khác vẫn thiếu ngữ cảnh. Hơn nữa, grounding score không phải đánh giá câu trả lời được sinh bởi một chatbot.

## 6. Minh chứng và tái lập

- Cấu hình, versions, hash và quy tắc chọn: `starter/results/config.json`.
- Kết quả tài nguyên, retrieval DEV và kiểm tra nạp lại: `starter/results/metrics_dev.json`.
- Kết quả TEST, hash config/corpus/benchmark và xác nhận không embed corpus: `starter/results/metrics_test.json`.
- Trace retrieval: `starter/results/traces_dev.jsonl`, `starter/results/traces_test.jsonl`.
- Audit: `starter/results/manual_coherence.csv`, `starter/results/manual_grounding.csv`, `starter/results/manual_metadata.csv`.
- Nhật ký cảnh báo/giới hạn đo: `starter/results/errors.log`.
- Nguồn và đơn vị: `starter/data/manifest.json`; benchmark/gold: `benchmark/questions.json`.
- Nhật ký cấu hình và kết quả: `artifacts/experiment_log.csv`.
- Lệnh PowerShell: `README.md` và `starter/README.md`. Chạy từ `starter/` bằng `& .\.venv\Scripts\python.exe -m src.run`; TEST dùng `& .\.venv\Scripts\python.exe -m src.run --split test`.

**Kết luận:** Trên dev, B đạt retrieval cao hơn A và được chọn theo tiêu chí đã chốt; test năm câu không phân biệt được A/B/C theo Hit@1 hoặc Evidence Hit@3. B và C bằng nhau trên tập này vì dùng cùng chunks/vectors và filter chưa tạo khác biệt quan sát được. Các kết quả và audit hữu ích để mô tả corpus thử nghiệm, nhưng cỡ test, số câu filter và quy trình audit không đủ để khẳng định B là chiến lược tốt nhất nói chung.
