# Bài báo (LaTeX, khuôn ACM `sigconf` giống arXiv 2402.08593)

| Tệp | Nội dung |
|---|---|
| `main.tex` | Bài báo (tiếng Việt) |
| `references.bib` | Tài liệu tham khảo |
| `generated/` | Bảng và số liệu sinh tự động từ CSV kết quả (không sửa tay) |
| `figures/` | Hình (PDF) |
| `make_tables.py` | Sinh lại `generated/` và hình phân tích theo kiểu rửa tiền |
| `main.pdf` | Bản PDF đã biên dịch |

## Biên dịch

Bài dùng tiếng Việt nên phải biên dịch bằng **XeLaTeX** (hoặc LuaLaTeX), không dùng pdfLaTeX.

- **Overleaf:** tải cả thư mục `paper/` lên, chọn *Menu → Compiler → XeLaTeX*, rồi bấm Recompile.
- **Máy cá nhân (TeX Live):**
  ```bash
  cd paper
  latexmk -xelatex main.tex
  # hoặc: xelatex main && bibtex main && xelatex main && xelatex main
  ```

## Sinh lại bảng số liệu

```bash
.venv/bin/python paper/make_tables.py   # đọc reports/evidence/, ghi paper/generated/
```

Bản `main.pdf` trong kho được biên dịch bằng [Tectonic](https://tectonic-typesetting.github.io) 0.15
(engine XeTeX, tự tải gói LaTeX): `tectonic -X compile main.tex` trong thư mục `paper/`.
