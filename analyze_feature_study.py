"""Independently verify the feature ablation predictions and write course artifacts."""
import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             f1_score, precision_score, recall_score, roc_curve)

LABELS = {"F0":"Giao dịch", "F1":"+ Lịch sử", "F2":"+ Cửa sổ", "F3":"+ Quan hệ", "F4":"Kết hợp"}


def alert_columns(columns):
    precision = next(c for c in columns if c.startswith("precision_at_") and c.rsplit("_",1)[1].isdigit())
    k = int(precision.rsplit("_",1)[1])
    recall = f"recall_at_{k}"
    if recall not in columns:
        raise ValueError("Missing matching fixed-budget recall")
    return precision,recall,k


def checked_metrics(y,p,threshold,expected_y):
    y,p = np.asarray(y),np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or not np.array_equal(y,expected_y):
        raise ValueError("Prediction labels/order do not match the fixed split")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Invalid labels or scores")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    pred = p>=threshold
    fpr,tpr,_ = roc_curve(y,p)
    k = min(1000,len(y))
    indices = np.argsort(-p,kind="stable")[:k]
    tp = int(y[indices].sum())
    return {"pr_auc":float(average_precision_score(y,p)),"roc_auc":float(roc_auc_score(y,p)),
            "f1":float(f1_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred)),"recall_at_fpr1":float(np.interp(.01,fpr,tpr)),
            f"precision_at_{k}":tp/k,f"recall_at_{k}":tp/max(1,int(y.sum()))}


def verify(directory,submission=None):
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((directory/"complete.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(directory/"results.csv")
    expected = {(exp,seed) for exp in manifest["groups"] for seed in manifest["seeds"]}
    actual = set(zip(rows.exp,rows.seed))
    if not complete.get("complete") or len(rows) != len(expected) or complete.get("runs") != len(expected) or actual != expected:
        raise ValueError("Incomplete or duplicate experiment records")
    if manifest["smoke"]:
        print("WARNING: synthetic smoke artifacts, not research results")
    sources_verified = False
    if submission:
        tree = ast.parse(Path(submission).read_text(encoding="utf-8"))
        files = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id == "FILES" for t in node.targets))
        for name,digest in manifest["sources"].items():
            if hashlib.sha256(files[name].encode("utf-8")).hexdigest() != digest:
                raise ValueError("Submission/source hash mismatch: "+name)
        sources_verified = True
    with np.load(directory/"split_arrays.npz") as data:
        split = {name:data[name] for name in data.files}
    for name in ("val","test"):
        ids,times,y = (split[f"{name}_{key}"] for key in ("row_id","timestamp","y"))
        stats = manifest["splits"][name]
        if len(y) != stats["rows"] or y.sum() != stats["positives"] or not np.all(np.diff(ids)==1):
            raise ValueError("Split identity mismatch")
        if len(times) != len(y) or times[0] != stats["first_timestamp"] or times[-1] != stats["last_timestamp"] or (np.diff(times)<0).any():
            raise ValueError("Split timestamps mismatch")
    if not manifest["splits"]["train"]["last_timestamp"] < split["val_timestamp"][0] or not split["val_timestamp"][-1] < split["test_timestamp"][0]:
        raise ValueError("Overlapping temporal split")
    for row in rows.to_dict("records"):
        exp,seed = row["exp"],row["seed"]
        if row["n_features"] != len(manifest["groups"][exp]):
            raise ValueError("Feature count mismatch")
        with np.load(directory/f"{exp}_seed{seed}_val.npz") as data:
            yv,pv,threshold = data["y"],data["p"],float(data["threshold"])
        checked_metrics(yv,pv,threshold,split["val_y"])
        prec,rec,thr = precision_recall_curve(yv,pv)
        f1 = 2*prec*rec/np.maximum(prec+rec,1e-12)
        expected_threshold = float(thr[np.argmax(f1[:-1])])
        if not np.isclose(threshold,expected_threshold,atol=1e-9,rtol=1e-9) or not np.isclose(threshold,row["threshold"],atol=1e-9,rtol=1e-9):
            raise ValueError("Threshold was not selected from validation predictions")
        if not np.isclose(average_precision_score(yv,pv),row["val_ap"],atol=1e-9,rtol=1e-9):
            raise ValueError("Validation AP mismatch")
        with np.load(directory/f"{exp}_seed{seed}_test.npz") as data:
            metrics = checked_metrics(data["y"],data["p"],float(data["threshold"]),split["test_y"])
            if not np.isclose(float(data["threshold"]),threshold,atol=1e-9,rtol=1e-9):
                raise ValueError("Validation/test thresholds differ")
        for name,value in metrics.items():
            if not np.isclose(value,row[name],atol=1e-9,rtol=1e-9):
                raise ValueError(f"Metric mismatch {exp} seed{seed}: {name}")
    verification = {"verified_runs":len(rows),"verified_prediction_files":2*len(rows),
                    "identical_split_labels":True,"validation_thresholds_verified":True,
                    "submission_sources_verified":sources_verified,"raw_sha256":manifest["raw_sha256"]}
    return manifest,rows,verification


def temporal_diagnostics(directory,rows,output):
    """Post-hoc descriptive cohorts; never refit models or change primary metrics."""
    cutoff = int(np.datetime64("2022-09-11","s").astype(np.int64))
    with np.load(directory/"split_arrays.npz") as split:
        times,y = split["test_timestamp"],split["test_y"]
    masks = {"before_2022-09-11":times<cutoff,"from_2022-09-11":times>=cutoff}
    if any(not mask.any() for mask in masks.values()):
        return "Không có đủ hai giai đoạn để thực hiện chẩn đoán phần cuối HI-Small."
    records = []
    for row in rows.to_dict("records"):
        with np.load(directory/f"{row['exp']}_seed{row['seed']}_test.npz") as data:
            p = data["p"]
        for segment,mask in masks.items():
            yy,pp = y[mask],p[mask]
            records.append({"exp":row["exp"],"seed":row["seed"],"segment":segment,
                "rows":int(mask.sum()),"positives":int(yy.sum()),"positive_rate":float(yy.mean()),
                "ap":float(average_precision_score(yy,pp)),
                "f1":float(f1_score(yy,pp>=row["threshold"]))})
    frame = pd.DataFrame(records)
    frame.to_csv(output/"temporal_diagnostics.csv",index=False)
    summary = frame.groupby(["exp","segment"])[["ap","f1"]].agg(["mean","std"])
    summary.to_csv(output/"temporal_diagnostics_summary.csv")
    lines = ["Chẩn đoán **bổ sung sau khi quan sát phân bố dữ liệu**, không thay đổi cohort chính hoặc mô hình. "
             "Mốc 11/09 không phải ranh giới train/test; cả hai phần dưới đều thuộc test đã cố định.",
             "", "| Phần test | Giao dịch | Ca dương | Tỷ lệ dương |", "|---|---:|---:|---:|"]
    for name,mask in masks.items():
        lines.append(f"| {name} | {int(mask.sum()):,} | {int(y[mask].sum()):,} | {y[mask].mean():.2%} |")
    lines += ["", "| Cấu hình | AP trước 11/09 | AP từ 11/09 |", "|---|---:|---:|"]
    for exp in LABELS:
        cells = [f"{summary.loc[(exp,name),('ap','mean')]:.4f} ± {summary.loc[(exp,name),('ap','std')]:.4f}" for name in masks]
        lines.append("| "+" | ".join([exp,*cells])+" |")
    before,after = "before_2022-09-11","from_2022-09-11"
    lines += ["", f"Trong cùng phần test trước11/09, F1 có AP {summary.loc[('F1',before),('ap','mean')]:.4f} "
              f"và F4 đạt {summary.loc[('F4',before),('ap','mean')]:.4f}; lợi ích quan sát được không chỉ đến từ phần cuối. "
              f"Ở phần cuối, F0 đã đạt AP {summary.loc[('F0',after),('ap','mean')]:.4f}, "
              f"còn F4 đạt {summary.loc[('F4',after),('ap','mean')]:.4f}: thêm lịch sử/quan hệ không cải thiện mọi cohort."]
    lines += ["", "AP phụ thuộc tỷ lệ nhãn: không dùng AP giữa hai giai đoạn để kết luận giai đoạn nào dễ hơn. "
              "Chỉ so sánh các cấu hình trong cùng một giai đoạn. Phần cuối có tỷ lệ nhãn và mật độ hoạt động khác rõ rệt, "
              "nên kết quả toàn test không tự chứng minh khả năng triển khai trên luồng giao dịch ngân hàng ổn định. "
              "Nguyên nhân thay đổi này chưa được xác minh từ cơ chế sinh dữ liệu."]
    return "\n".join(lines)


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,default=Path("results/feature_study_analysis"))
    ap.add_argument("--submission",type=Path)
    args = ap.parse_args()
    manifest,rows,verification = verify(args.result_dir,args.submission)
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    (output/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    metrics = ["pr_auc","val_ap","f1","precision","recall","roc_auc","train_seconds"]
    metrics += [c for c in rows if c.startswith(("precision_at_","recall_at_"))]
    summary = rows.groupby("exp",sort=True)[metrics].agg(["mean","std"])
    summary.to_csv(output/"summary.csv")
    rows.to_csv(output/"all_results.csv",index=False)
    delta_rows = []
    for high,low in [("F1","F0"),("F2","F1"),("F3","F1"),("F4","F1"),("F4","F2"),("F4","F3")]:
        for seed in manifest["seeds"]:
            hi = rows[(rows.exp==high)&(rows.seed==seed)].iloc[0]
            lo = rows[(rows.exp==low)&(rows.seed==seed)].iloc[0]
            delta_rows.append({"comparison":high+" - "+low,"seed":seed,
                               "delta_ap":hi.pr_auc-lo.pr_auc,"delta_f1":hi.f1-lo.f1})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output/"paired_differences.csv",index=False)
    pcol,rcol,k = alert_columns(rows.columns)
    table = [f"| Cấu hình | Số feature | AP | F1 | Precision@{k} | Recall@{k} |",
             "|---|---:|---:|---:|---:|---:|"]
    for exp in LABELS:
        cells = [f"{summary.loc[exp,(m,'mean')]:.4f} ± {summary.loc[exp,(m,'std')]:.4f}" for m in ["pr_auc","f1",pcol,rcol]]
        table.append("| "+" | ".join([exp+" "+LABELS[exp],str(len(manifest["groups"][exp])),*cells])+" |")
    table_text = "\n".join(table)
    (output/"table.md").write_text(table_text+"\n",encoding="utf-8")
    importance = []
    for seed in manifest["seeds"]:
        imp = pd.read_csv(args.result_dir/f"F4_seed{seed}_importance.csv")
        imp["normalized_gain"] = imp.total_gain/max(float(imp.total_gain.sum()),1e-12)
        imp["seed"] = seed
        importance.append(imp)
    importance = pd.concat(importance).groupby("feature").normalized_gain.agg(["mean","std"]).sort_values("mean",ascending=False)
    importance.to_csv(output/"F4_feature_importance.csv")
    diagnostics = temporal_diagnostics(args.result_dir,rows,output)
    os.environ.setdefault("MPLCONFIGDIR",str((output/".mplcache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    order = list(LABELS)
    colors = ["#9ba8b7","#416c93","#2c9c95","#bc8c36","#8359a3"]
    for ax,metric,title in [(axes[0],"pr_auc","Average precision"),(axes[1],rcol,f"Recall at {k} alerts")]:
        ax.bar(order,[summary.loc[e,(metric,"mean")] for e in order],
               yerr=[summary.loc[e,(metric,"std")] for e in order],color=colors,capsize=4)
        ax.set_title(title)
        ax.set_ylim(0,1)
        ax.set_ylabel("Score")
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("IBM AML HI-Small · fixed XGBoost · strict historical timestamps")
    fig.supxlabel(f"F0: transaction · F1: history · F2: activity · F3: relationships · F4: combined\n"
                  f"Mean ± sample SD across {len(manifest['seeds'])} training seeds on the same test set",fontsize=9)
    fig.savefig(output/"feature_ablation.png",dpi=200)
    fig.savefig(output/"feature_ablation.pdf")
    plt.close(fig)
    pair_summary = deltas.groupby("comparison").delta_ap.agg(["mean","std"])
    best = summary[("pr_auc","mean")].idxmax()
    delta_full = float(pair_summary.loc["F4 - F1","mean"])
    conclusion = (f"F4 tăng AP trung bình {delta_full:.4f} so với F1." if delta_full>0 else
                  f"F4 không cải thiện AP trung bình so với F1 (chênh lệch {delta_full:.4f}).")
    split_lines = "\n".join(f"- {name}: {s['rows']:,} giao dịch; {s['positives']:,} ca dương." for name,s in manifest["splits"].items())
    delta_lines = "\n".join(f"- {name}: ΔAP {r['mean']:+.4f} ± {r['std']:.4f}." for name,r in pair_summary.iterrows())
    importance_lines = "\n".join(f"- `{name}`: {r['mean']:.3%} normalized total gain." for name,r in importance.head(12).iterrows())
    smoke_banner = "**CHỈ LÀ SMOKE TEST TRÊN DỮ LIỆU GIẢ LẬP NHỎ — không phải kết quả nghiên cứu.**\n\n" if manifest["smoke"] else ""
    report = f'''# Thử nghiệm đặc trưng phát hiện rửa tiền — đồ án ATBMHTTT

{smoke_banner}## Câu hỏi và kết luận
Lịch sử, hoạt động gần đây và quan hệ tài khoản có cải thiện phát hiện giao dịch rửa tiền khi giữ nguyên XGBoost và ngân sách cảnh báo không?

{conclusion} Cấu hình có AP trung bình cao nhất trong năm cấu hình đã định trước là {best} ({summary.loc[best,('pr_auc','mean')]:.4f}). Đây là kết quả thăm dò trên một bộ dữ liệu mô phỏng.

## Thiết kế
- IBM HI-Small; hash gốc `{manifest['raw_sha256']}`.
- Phiên bản đặc trưng `{manifest['feature_version']}`. Cùng phân chia theo thời gian 60/20/20, timestamp không xuất hiện ở hai tập.
- F0: 11 thuộc tính giao dịch; F1: thêm 17 đặc trưng lịch sử; F2: F1 + 20 đặc trưng hoạt động/số tiền; F3: F1 + 15 đặc trưng quan hệ cục bộ; F4: kết hợp 63 đặc trưng.
- Đặc trưng lịch sử chỉ dùng t < thời điểm dự đoán; cửa sổ [t-1h,t) và [t-24h,t). Mọi giao dịch cùng phút chia sẻ ảnh chụp lịch sử. Giao dịch hiện tại chỉ cung cấp thuộc tính của chính nó.
- Tổng tiền và độ lệch số tiền mới được tính riêng theo tài khoản/tiền tệ, dùng Amount Received cho bên nhận. Đặc trưng kiểm soát mean-log-amount lịch sử giữ cách gộp tiền tệ cũ; không coi nó là số dư tiền tệ.
- ID tài khoản chỉ để tra lịch sử; không đưa ID hoặc nhãn vào đầu vào mô hình. Những giao dịch test đã xảy ra trước đó được dùng làm lịch sử không nhãn, phù hợp giả định xử lý theo thời gian.
- Giữ XGB_PARAMS như đợt trước; không augmentation, không tìm siêu tham số hoặc thêm feature sau khi xem test. Seed {manifest['seeds']}; early stopping và ngưỡng tối đa F1 chọn bằng validation.

{split_lines}

## Kết quả
Mean ± sample SD theo seed; `pr_auc` trong CSV là average precision (AP), không phải tích phân hình thang PR.

{table_text}

## Phân rã tác động
Chênh lệch theo cặp cùng seed; đây là mô tả, không phải kiểm định ý nghĩa thống kê:

{delta_lines}

F2−F1 đo lợi ích nhóm hoạt động khi có lịch sử; F3−F1 đo nhóm quan hệ. F4−F2 và F4−F3 kiểm tra đóng góp bổ sung khi nhóm còn lại đã có. Với {k} cảnh báo trên toàn tập test, Precision@{k} × {k} là số ca dương trung bình được phát hiện; đây không phải ngân sách cảnh báo mỗi ngày.

## Đặc trưng được F4 sử dụng
Tỷ trọng total gain được chuẩn hóa từng seed rồi lấy trung bình; đây là cách mô hình sử dụng đặc trưng, không chứng minh nhân quả hay giá trị của từng feature khi đứng độc lập.

{importance_lines}

## Chẩn đoán phân bố theo thời gian
{diagnostics}

## Chi phí và khả năng tái lập
- Tạo toàn bộ đặc trưng, gồm đọc dữ liệu và JIT: {manifest['feature_build_seconds_including_io_jit']:.1f} giây. Bộ nhớ DataFrame đặc trưng: {manifest['feature_memory_bytes']/1e9:.2f} GB, không phải peak RAM.
- Tổng thời gian train {len(rows)} mô hình: {rows.train_seconds.sum():.1f} giây. Thời gian này không bao gồm toàn bộ đánh giá và ghi file; không suy ra độ trễ triển khai trực tuyến.
- Kiểm tra độc lập {verification['verified_runs']} lượt / {verification['verified_prediction_files']} file dự đoán: nhãn/split giống nhau, AP và chỉ số ngưỡng khớp, ngưỡng đúng từ validation. Snapshot nguồn đã xác minh: {verification['submission_sources_verified']}.
- Môi trường và toàn bộ tham số: xem manifest.json của lượt chạy. Không gọi API LLM trong thí nghiệm này.

## Giới hạn và cách trình bày trong đồ án
- Test HI-Small đã được xem trong nghiên cứu augmentation trước; đây là đánh giá thăm dò trên test dùng lại, không phải holdout xác nhận mới.
- {len(manifest['seeds'])} seed chỉ đo biến động huấn luyện trên cùng một mẫu test. Không tuyên bố ý nghĩa thống kê chỉ dựa vào SD.
- Dữ liệu IBM là mô phỏng, không có KYC/số dư hoặc xác nhận điều tra thực tế. Nhãn rửa tiền mô phỏng khác cảnh báo đáng ngờ tại ngân hàng thật.
- Baseline F1 dùng lịch sử nghiêm ngặt theo timestamp, khác E0 v3 cũ vốn cho phép thứ tự dòng trong cùng phút. Không ghép hai thí nghiệm như cùng một baseline.
- Quan hệ là thống kê cục bộ, chưa khai thác chuỗi nhiều bước hay vòng đồ thị. Mẫu giao dịch giống một kịch bản rửa tiền không tự nó chứng minh tội phạm.
- Kết quả chỉ áp dụng cấu hình đã khảo sát; nếu cần khẳng định khái quát, phải định trước đánh giá trên bộ dữ liệu hoặc giai đoạn mới.

## Tài liệu nền
Graph Feature Preprocessor (ICAIF 2024) là nghiên cứu trước về khai thác cấu trúc giao dịch cho boosting: https://research.ibm.com/publications/graph-feature-preprocessor-real-time-subgraph-based-feature-extraction-for-financial-crime-detection . Đồ án này là nghiên cứu thực nghiệm về đặc trưng theo thời gian và quan hệ, không tuyên bố phát minh hướng tiếp cận mới.
'''
    (output/"report_vi.md").write_text(report,encoding="utf-8")
    print(table_text)
    print(f"Verified {len(rows)} runs; report: {output/'report_vi.md'}")


if __name__ == "__main__":
    main()
