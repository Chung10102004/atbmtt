"""Generate LaTeX tables and number macros for paper/main.tex from the evidence CSVs in reports/evidence/.

    python paper/make_tables.py

Numbers in the paper text come from generated/numbers.tex, so the text cannot drift from the CSVs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "reports/evidence"
OUT = Path(__file__).resolve().parent / "generated"

FEATURE_VI = {
    "src_prev_in": "Số lượt nhận trước đó của bên gửi",
    "dst_secs_since_prev": "Thời gian từ giao dịch gần nhất của bên nhận",
    "src_log_amt_vs_out_mean": "Độ lệch số tiền so với lịch sử gửi (cùng tiền tệ)",
    "pay_format": "Hình thức thanh toán",
    "hour": "Giờ trong ngày",
    "pair_prev_n": "Số giao dịch trước đó của cặp gửi$\\to$nhận",
    "src_log_amt_vs_in_mean": "Độ lệch số tiền so với lịch sử nhận của bên gửi",
    "dst_log_amt_vs_in_mean": "Độ lệch số tiền so với lịch sử nhận của bên nhận",
    "src_log_in_sum_24h": "Log tổng tiền bên gửi đã nhận trong 24 giờ",
    "src_prev_in_amt_mean": "Trung bình log số tiền bên gửi đã nhận",
}
GROUP_VI = {"raw": "Giao dịch", "history": "Lịch sử", "activity": "Hoạt động", "relationships": "Quan hệ"}


def num(x, d=3):
    """Vietnamese decimal comma."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    text = f"{x:.{d}f}".replace(".", "{,}")
    return "$-$" + text[1:] if text.startswith("-") else text


def pm(m, s, d=3):
    return f"{num(m, d)} $\\pm$ {num(s, d)}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def bold_best(values, cells):
    best = np.nanmax(values)
    return [f"\\textbf{{{c}}}" if np.isclose(v, best) else c for v, c in zip(values, cells)]


def feature_study(macros):
    s = pd.read_csv(EVID / "feature_study/summary.csv", header=[0, 1], index_col=0)
    names = {"F0": "F0 Giao dịch", "F1": "F1 +Lịch sử", "F2": "F2 +Hoạt động",
             "F3": "F3 +Quan hệ", "F4": "F4 Kết hợp"}
    nfeat = {"F0": 11, "F1": 28, "F2": 48, "F3": 43, "F4": 63}
    word = {"F0": "Zero", "F1": "One", "F2": "Two", "F3": "Three", "F4": "Four"}
    cols = ["pr_auc", "val_ap", "f1", "precision_at_1000", "recall_at_1000"]
    cells = {c: bold_best(s[(c, "mean")].to_numpy(),
                          [pm(s.loc[e, (c, "mean")], s.loc[e, (c, "std")]) for e in s.index]) for c in cols}
    rows = [f"{names[e]} & {nfeat[e]} & " + " & ".join(cells[c][i] for c in cols) + r" \\"
            for i, e in enumerate(s.index)]
    (OUT / "tab_features.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    for e in s.index:
        for c, key in (("pr_auc", "AP"), ("val_ap", "Val"), ("f1", "Fone"), ("precision_at_1000", "Pk"),
                       ("recall_at_1000", "Rk")):
            macros.append(macro(f"F{word[e]}{key}", num(s.loc[e, (c, "mean")])))
        # number of true laundering transactions among the 1,000 alerts
        macros.append(macro(f"F{word[e]}Hits", f"{s.loc[e, ('precision_at_1000', 'mean')] * 1000:.0f}"))
    macros.append(macro("FFourPkPct", num(100 * s.loc["F4", ("precision_at_1000", "mean")], 2)))
    macros.append(macro("FFourRkPct", num(100 * s.loc["F4", ("recall_at_1000", "mean")], 2)))

    d = pd.read_csv(EVID / "feature_study/paired_differences.csv")
    g = d.groupby("comparison", sort=False)["delta_ap"]
    rows = []
    for c, m, sd in zip(g.mean().index, g.mean(), g.std()):
        wins = int((d[d.comparison == c].delta_ap > 0).sum())
        rows.append(f"{c.replace(' - ', ' $-$ ')} & +{num(m, 4)} & {num(sd, 4)} & {wins}/3 \\\\")
    (OUT / "tab_feature_deltas.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

    t = pd.read_csv(EVID / "feature_study/temporal_diagnostics_summary.csv", header=[0, 1], index_col=[0, 1])
    rows = []
    for e in ["F0", "F1", "F2", "F3", "F4"]:
        a, b = t.loc[(e, "before_2022-09-11")], t.loc[(e, "from_2022-09-11")]
        rows.append(f"{e} & {pm(a[('ap', 'mean')], a[('ap', 'std')])} & {pm(b[('ap', 'mean')], b[('ap', 'std')])} \\\\")
        macros.append(macro(f"F{word[e]}APpre", num(a[("ap", "mean")])))
        macros.append(macro(f"F{word[e]}APpost", num(b[("ap", "mean")])))
    (OUT / "tab_temporal.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def importance(macros):
    imp = pd.read_csv(EVID / "feature_study/F4_feature_importance.csv").head(10)
    cat = pd.read_csv(EVID / "feature_study/feature_catalog.csv").set_index("feature")["family"]
    rows = [f"{i + 1} & {FEATURE_VI[r.feature]} & {GROUP_VI[cat[r.feature]]} & {num(100 * r['mean'], 2)} \\\\"
            for i, (_, r) in enumerate(imp.iterrows())]
    (OUT / "tab_importance.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")


def augmentation(macros):
    m = pd.read_csv(EVID / "augmentation/metrics_mean_std.csv", header=[0, 1], index_col=0)
    names = {"E0": "E0 Không tăng cường", "ER": "ER Lặp mẫu dương", "E1": "E1 SMOTENC", "E2": "E2 CTGAN",
             "E3-basic": "E3a LLM, prompt cơ bản", "E3-AML": "E3b LLM, prompt có tri thức AML"}
    rows = []
    for e in m.index:
        r = m.loc[e]
        rows.append(f"{names[e]} & {pm(r[('pr_auc', 'mean')], r[('pr_auc', 'std')])} & "
                    f"{pm(r[('f1', 'mean')], r[('f1', 'std')])} & {num(r[('precision_at_1000', 'mean')])} & "
                    f"{num(r[('ks_mean', 'mean')])} & {num(r[('disc_auc', 'mean')])} \\\\")
    (OUT / "tab_aug.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    key = {"E0": "EZero", "ER": "ER", "E1": "EOne", "E2": "ETwo", "E3-basic": "ELLMa", "E3-AML": "ELLMb"}
    for e in m.index:
        macros.append(macro(f"{key[e]}AP", num(m.loc[e, ("pr_auc", "mean")])))
        macros.append(macro(f"{key[e]}Fone", num(m.loc[e, ("f1", "mean")])))
        if e != "E0":
            macros.append(macro(f"{key[e]}Disc", num(m.loc[e, ("disc_auc", "mean")])))
            macros.append(macro(f"{key[e]}KS", num(m.loc[e, ("ks_mean", "mean")])))
    d = pd.read_csv(EVID / "augmentation/paired_ap_differences.csv").set_index("exp")
    for e in d.index:
        macros.append(macro(f"{key[e]}Delta", num(d.loc[e, "ap_delta_mean"], 4)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    macros = ["% Auto-generated by make_tables.py - do not edit\n"]
    feature_study(macros)
    importance(macros)
    augmentation(macros)
    (OUT / "numbers.tex").write_text("".join(macros))
    print("Wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
