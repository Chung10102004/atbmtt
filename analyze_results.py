"""Compare compatible Kaggle runs and export research tables and figures."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def collect(directories):
    frames, manifests = [], []
    for directory in directories:
        manifests.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
        frames.append(pd.read_csv(directory / "results.csv"))
    reference = manifests[0]
    for m in manifests[1:]:
        for key in ("feature_version", "raw_sha256", "columns", "splits", "ratio", "seeds", "xgboost_params", "packages"):
            if m[key] != reference[key]:
                raise ValueError(f"Cannot combine experiments with different {key}")
        if m["source_sha256"]["src/model.py"] != reference["source_sha256"]["src/model.py"]:
            raise ValueError("Detector/evaluation source differs between runs")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["exp", "seed"]).any():
        raise ValueError("Duplicate experiment/seed rows would bias the summary")
    order = ["E0", "ER", "E1", "E2", "E3-basic", "E3-AML"]
    data["exp"] = pd.Categorical(data.exp, categories=order, ordered=True)
    return data.sort_values(["exp", "seed"]), reference


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("results/rerun_analysis"))
    ap.add_argument("--budget-db", type=Path, default=Path("data/llm/rerun_20261029/budget.sqlite"))
    args = ap.parse_args()
    data, manifest = collect(args.result_dirs)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "all_results.csv", index=False)
    metrics = ["pr_auc", "f1", "precision", "recall", "precision_at_1000", "recall_at_1000", "ks_mean", "disc_auc"]
    stats = data.groupby("exp", observed=True)[metrics].agg(["mean", "std"])
    stats.to_csv(output / "metrics_mean_std.csv")
    table = ["| Phương pháp | AP | F1 | Precision@1000 | Recall@1000 |", "|---|---:|---:|---:|---:|"]
    for exp, row in stats.iterrows():
        values = [f"{row[(metric, 'mean')]:.4f} ± {row[(metric, 'std')]:.4f}"
                  for metric in ["pr_auc", "f1", "precision_at_1000", "recall_at_1000"]]
        table.append("| " + " | ".join([str(exp), *values]) + " |")
    (output / "table.md").write_text("\n".join(table), encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    positions = np.arange(len(stats))
    colors = ["#475569" if name == "E0" else "#7c3aed" if str(name).startswith("E3") else "#0284c7" for name in stats.index]
    ax.bar(positions, stats[("pr_auc", "mean")], yerr=stats[("pr_auc", "std")], capsize=5,
           color=colors, alpha=.85, width=.65)
    for i, exp in enumerate(stats.index):
        points = data.loc[data.exp == exp, "pr_auc"].to_numpy()
        ax.scatter(i + np.linspace(-.12, .12, len(points)), points, color="#0f172a", s=23, zorder=3)
    ax.set(xticks=positions, xticklabels=[str(x) for x in stats.index], ylabel="Average Precision (AP)",
           title="IBM AML HI-Small: mean +/- sample SD across three seeds", ylim=(0, .8))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(output / "average_precision.png", dpi=200)
    fig.savefig(output / "average_precision.pdf")
    plt.close(fig)

    baseline = data[data.exp == "E0"].set_index("seed").pr_auc
    differences = []
    for exp in stats.index:
        if exp == "E0":
            continue
        delta = data[data.exp == exp].set_index("seed").pr_auc - baseline
        differences.append({"exp": str(exp), "ap_delta_mean": float(delta.mean()), "ap_delta_std": float(delta.std())})
    pd.DataFrame(differences).to_csv(output / "paired_ap_differences.csv", index=False)

    budget = None
    if args.budget_db.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from llm_budget import BudgetLedger
        budget = BudgetLedger(args.budget_db, 2., .1, .5).summary()
        (output / "api_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    splits = manifest["splits"]
    report = ["# Kết quả chạy lại IBM AML trên Kaggle — 29/10/2026", "",
              f"Phiên bản đặc trưng: `{manifest['feature_version']}`. Tỷ lệ bổ sung: {manifest['ratio']}. Các seed: {manifest['seeds']}.", "",
              "## Dữ liệu và phương pháp", "",
              "IBM AML là dữ liệu mô phỏng. Train/validation/test không có timestamp trùng nhau; các giao dịch cùng timestamp trong mỗi tập dùng thứ tự ổn định của file gốc.", ""]
    for name, split in splits.items():
        report.append(f"- {name}: {split['rows']:,} giao dịch, {split['positives']:,} ca dương.")
    report += ["", "Cùng XGBoost và siêu tham số, early stopping trên validation. Ngưỡng tối đa F1 được chọn trên validation, áp dụng nguyên cho test. K=1000 cố định cho chỉ số cảnh báo. AP được lưu dưới tên `pr_auc` trong CSV.", "",
               "## Kết quả", "", *table, "", "Các số sau dấu ± là độ lệch chuẩn mẫu qua ba seed, không phải khoảng tin cậy.", "",
               "## Chênh lệch AP so với E0", ""]
    report += [f"- {row['exp']}: {row['ap_delta_mean']:+.4f} trung bình; SD của chênh lệch theo seed {row['ap_delta_std']:.4f}." for row in differences]
    if all(name in stats.index for name in ("E0", "E3-basic", "E3-AML")):
        base_ap = stats.loc["E0", ("pr_auc", "mean")]
        report += ["", "## Trả lời câu hỏi nghiên cứu", ""]
        for name in ("E3-basic", "E3-AML"):
            ap_value = stats.loc[name, ("pr_auc", "mean")]
            delta = ap_value - base_ap
            report.append(f"- {name}: AP {ap_value:.4f}, chênh lệch {delta:+.4f} so với E0.")
        if all(stats.loc[name, ("pr_auc", "mean")] < base_ap for name in ("E3-basic", "E3-AML")):
            report.append("Trong cấu hình đã khóa của đợt chạy này, tăng cường bằng cả hai prompt LLM đều không cải thiện AP trung bình so với dữ liệu gốc.")
        basic = data[data.exp == "E3-basic"].set_index("seed").pr_auc
        aml = data[data.exp == "E3-AML"].set_index("seed").pr_auc
        paired = aml - basic
        report.append(f"Thêm mô tả typology AML tăng AP ở {int((paired > 0).sum())}/{len(paired)} seed; chênh lệch trung bình AML trừ basic là {paired.mean():+.4f}. Hiệu ứng không ổn định qua ba seed; không kết luận mô tả nghiệp vụ luôn có lợi hoặc luôn có hại.")
        counts = {name: round(stats.loc[name, ("precision_at_1000", "mean")] * 1000)
                  for name in ("E0", "E3-basic", "E3-AML")}
        report.append(f"Với 1.000 cảnh báo, số ca dương tìm được trung bình: E0 {counts['E0']}, LLM-basic {counts['E3-basic']}, LLM-AML {counts['E3-AML']}.")
        report.append("Classifier phân biệt train dương và mẫu LLM có AUC cao trong phép thử đã chạy. Đây là dấu hiệu khác biệt phân phối còn lớn; chưa đủ chứng minh nguyên nhân làm AP giảm.")
    if budget:
        settled = budget["calls_by_status"].get("settled", {})
        report += ["", "## Ngân sách API", "",
                   f"Chi phí suy ra từ usage và đơn giá người dùng cung cấp: **{settled.get('usd', 0):.4f} USD**. Tổng đã dùng cộng dự phòng: **{budget['committed_usd']:.4f} / 2 USD**.",
                   "Đơn giá giả định: 0,1 USD/triệu token vào và 0,5 USD/triệu token ra. Đây không phải số dư hay hóa đơn được xác minh từ nhà cung cấp."]
    report += ["", "## Giới hạn và diễn giải", "",
               "- Ba seed trên cùng một cách chia thời gian không tương đương ba bộ dữ liệu độc lập. Không kết luận ý nghĩa thống kê chỉ từ mean ± SD.",
               "- Sinh trực tiếp 28 đặc trưng không tái tạo được đồ thị tài khoản hoặc chứng minh typology do LLM tự gán.",
               "- KS là độ giống phân phối từng cột; classifier thật/giả kiểm tra một phần quan hệ kết hợp. Cả hai không thay thế hiệu quả phát hiện trên test.",
               "- Các phương pháp sinh được sửa miền/đếm/null bằng cùng hàm; số dòng sửa nằm trong results.csv. Các kiểm tra này là điều kiện cần, không chứng minh toàn bộ lịch sử giao dịch có thể tồn tại.",
               "- Không trộn kết quả này với bảng lịch sử trong README vì phiên bản đặc trưng và ranh giới chia đã thay đổi.",
               "- Chỉ suy rộng kết luận đến cấu hình và dữ liệu mô phỏng này; chưa có đánh giá trên dữ liệu ngân hàng thực.", "",
               "Các file kèm theo: all_results.csv, metrics_mean_std.csv, paired_ap_differences.csv, average_precision.png/pdf và manifest của từng job."]
    (output / "report_vi.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(table))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
