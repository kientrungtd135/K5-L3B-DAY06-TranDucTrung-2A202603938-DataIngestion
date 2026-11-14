# Starter · pipeline Day 06 đã hoàn thiện

Pipeline trong `src/` đọc Markdown, chuẩn hóa và chia chunk, gọi embedding API tương thích OpenAI, lưu vector vào PostgreSQL + pgvector, rồi chạy retrieval/evaluation. Mặc định `python -m src.run` chạy DEV; `--split test` chạy TEST trên cấu hình đã cố định, không chọn lại chiến lược hoặc embed lại corpus.

## Thành phần và dữ liệu

- `data/corpus/`: 30 bài Markdown độc lập, 15 business và 15 technology.
- `data/manifest.json`: nguồn, license ODC-BY theo khai báo của dataset MultiHopRAG, metadata, ranh giới và số ký tự các đơn vị. License dataset không xác nhận quyền riêng của từng bài báo; repo không có minh chứng cấp quyền từ từng nhà xuất bản. Cần xác minh quyền áp dụng trước khi tiếp tục phân phối nội dung và giữ attribution nguồn phù hợp. Với corpus này, `page=1` là ID vị trí bài viết, không phải trang PDF.
- `../benchmark/questions.json`: 20 câu đã chia 15 dev / 5 test; gold và evidence được giữ trong benchmark, không đưa vào query.
- `results/`: config, metrics, traces, log lỗi/cảnh báo và ba bảng audit thủ công.
- `public.day6_chunks`: bảng PostgreSQL + pgvector; tìm kiếm cosine exact, vector 1536 chiều.

## Setup trên Windows PowerShell

Từ thư mục gốc repository:

```powershell
Set-Location .\starter
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose -p day6_student_starter up -d --wait
& .\.venv\Scripts\python.exe scripts/check_setup.py
```

Điền thông tin kết nối database và thông tin xác thực API embedding vào `starter/.env` theo `.env.example`. Không chia sẻ hoặc commit `.env`. Nếu môi trường đã được cài đặt, bỏ qua các bước tạo venv/cài package và chỉ bảo đảm PostgreSQL đang chạy.

## Chạy

Chạy từ `starter/`:

```powershell
& .\.venv\Scripts\python.exe -m src.run
```

Lệnh mặc định chạy DEV với A/B/C, tạo embedding tài liệu/query, cập nhật index và lưu artifacts; có thể phát sinh chi phí API. Chiến lược được chọn theo selection rule trong config sau khi chỉ xem DEV.

TEST chỉ chạy sau khi index A/B/C và config đã có:

```powershell
& .\.venv\Scripts\python.exe -m src.run --split test
```

Lệnh này kiểm tra hash config/corpus/benchmark và index hiện tại, sau đó embed mỗi query test một lần dùng chung cho A/B/C. Nó không ingest hoặc embed lại tài liệu, và không thay đổi lựa chọn đã chốt. Nếu kiểm tra không khớp, chương trình dừng thay vì tự tạo lại index.

Trước khi chạy lại TEST, hãy bảo quản các CSV audit đã duyệt bên ngoài `starter/results/`. TEST sẽ dừng trước API, database và thao tác ghi nếu ba bảng audit có điểm, nhãn, lý do hoặc `review_status`. Chỉ chuyển các CSV cũ ra khỏi `results` sau khi đã giữ bản cần thiết và chủ định tái tạo audit; bảng chưa chấm hoặc chưa tồn tại vẫn được phép tạo.

`results/metrics_dev.json`, `metrics_test.json`, `traces_dev.jsonl`, `traces_test.jsonl`, `config.json`, `errors.log` và `manual_*.csv` là kết quả đã lưu; không cần chạy lại để đọc. Để dừng PostgreSQL nhưng giữ volume:

```powershell
docker compose -p day6_student_starter stop
```

Không xóa volume nếu cần giữ index hiện tại.
