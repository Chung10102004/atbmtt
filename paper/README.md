# Bài báo (LaTeX, khuôn ACM `sigconf` giống arXiv 2402.08593)

Hai bản cùng nội dung:

| Bản | Mã nguồn | PDF |
|---|---|---|
| Tiếng Việt (thuật ngữ chuyên ngành giữ tiếng Anh) | `main.tex` | `main.pdf` |
| Tiếng Anh | `main_en.tex` | `main_en.pdf` |

| Tệp / thư mục | Nội dung |
|---|---|
| `references.bib` | Tài liệu tham khảo (dùng chung) |
| `generated/vi/`, `generated/en/` | Bảng và số liệu sinh tự động từ `reports/evidence/` (không sửa tay) |
| `figures/` | Hình (PDF), sinh bởi `make_tables.py`, dùng chung cho hai bản |
| `make_tables.py` | Sinh lại `generated/` và `figures/` |

## Biên dịch

Bản tiếng Việt phải biên dịch bằng **XeLaTeX** (hoặc LuaLaTeX); bản tiếng Anh dùng được cả pdfLaTeX.

- **Overleaf:** tải cả thư mục `paper/` lên, chọn *Menu → Compiler → XeLaTeX*, chọn tệp chính
  (`main.tex` hoặc `main_en.tex`), rồi bấm Recompile.
- **Máy cá nhân (TeX Live):** `latexmk -xelatex main.tex` và `latexmk -xelatex main_en.tex` trong thư mục `paper/`.
- **Tectonic** (cách đã dùng để tạo PDF trong kho): `tectonic -X compile main.tex` và
  `tectonic -X compile main_en.tex`.

## Sinh lại bảng, số liệu và hình

```bash
.venv/bin/python paper/make_tables.py   # đọc reports/evidence/, ghi paper/generated/{vi,en} và paper/figures/
```
