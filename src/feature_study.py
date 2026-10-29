"""Strictly past-only AML features; same-timestamp events share one snapshot.

Optional Numba accelerates linear event scans on Kaggle. The Python fallback
uses the identical algorithm for small local regression fixtures.
"""
import numpy as np
import pandas as pd

from data import CURRENCIES, PAY_FORMATS

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function

VERSION = "strict-time-feature-study-v1"
WINDOWS = ((3600, "1h"), (86400, "24h"))
RAW = ["log_amt_paid", "log_amt_recv", "hour", "weekday", "pay_format",
       "pay_currency", "recv_currency", "same_bank", "self_txn", "cross_currency", "round_amt"]
HISTORY_NAMES = ["prev_n", "prev_out", "prev_in", "prev_out_amt_mean",
                 "prev_in_amt_mean", "secs_since_prev", "same_day_n", "same_day_dir_n"]


@njit(cache=True)
def _history_scan(account, times, side, log_amount):
    result = np.empty((len(times), 8), np.float32)
    start = 0
    while start < len(times):
        end = start+1
        while end < len(times) and account[end] == account[start]:
            end += 1
        out_n, in_n, out_sum, in_sum = 0, 0, 0., 0.
        day_out, day_in, day, previous = 0, 0, -1, -1
        j = start
        while j < end:
            stop = j+1
            while stop < end and times[stop] == times[j]:
                stop += 1
            if times[j]//86400 != day:
                day_out, day_in, day = 0, 0, times[j]//86400
            for k in range(j, stop):
                result[k, 0] = out_n+in_n
                result[k, 1] = out_n
                result[k, 2] = in_n
                result[k, 3] = out_sum/out_n if out_n else np.nan
                result[k, 4] = in_sum/in_n if in_n else np.nan
                result[k, 5] = times[j]-previous if previous >= 0 else np.nan
                result[k, 6] = day_out+day_in
                result[k, 7] = day_out if side[k] else day_in
            for k in range(j, stop):
                if side[k]:
                    out_n += 1
                    out_sum += log_amount[k]
                    day_out += 1
                else:
                    in_n += 1
                    in_sum += log_amount[k]
                    day_in += 1
            previous = times[j]
            j = stop
        start = end
    return result


@njit(cache=True)
def _window_scan(account, times, side, partner, seconds, n_accounts):
    result = np.zeros((len(times), 4), np.float32)
    frequency = np.zeros((2, n_accounts), np.int32)
    start = 0
    while start < len(times):
        end = start+1
        while end < len(times) and account[end] == account[start]:
            end += 1
        left, right, j = start, start, start
        counts = np.zeros(2, np.int64)
        unique = np.zeros(2, np.int64)
        while j < end:
            stop = j+1
            while stop < end and times[stop] == times[j]:
                stop += 1
            while right < j:
                d, p = side[right], partner[right]
                if frequency[d, p] == 0:
                    unique[d] += 1
                frequency[d, p] += 1
                counts[d] += 1
                right += 1
            while left < right and times[left] < times[j]-seconds:
                d, p = side[left], partner[left]
                frequency[d, p] -= 1
                counts[d] -= 1
                if frequency[d, p] == 0:
                    unique[d] -= 1
                left += 1
            for k in range(j, stop):
                result[k, 0], result[k, 1] = counts[1], counts[0]
                result[k, 2], result[k, 3] = unique[1], unique[0]
            j = stop
        for k in range(left, right):
            frequency[side[k], partner[k]] = 0
        start = end
    return result


@njit(cache=True)
def _stable_add(sums, correction, w, d, value):
    # Neumaier accumulation preserves cents when a very large transfer expires.
    current = sums[w,d]
    total = current+value
    if abs(current) >= abs(value):
        correction[w,d] += (current-total)+value
    else:
        correction[w,d] += (value-total)+current
    sums[w,d] = total


@njit(cache=True)
def _money_scan(key, times, side, amount):
    result = np.empty((len(times), 6), np.float32)
    start = 0
    while start < len(times):
        end = start+1
        while end < len(times) and key[end] == key[start]:
            end += 1
        counts = np.zeros(2, np.int64)
        logs = np.zeros(2, np.float64)
        sums = np.zeros((2, 2), np.float64)
        correction = np.zeros((2, 2), np.float64)
        window_counts = np.zeros((2, 2), np.int64)
        left = np.full(2, start, np.int64)
        j = start
        while j < end:
            stop = j+1
            while stop < end and times[stop] == times[j]:
                stop += 1
            for w in range(2):
                seconds = 3600 if w == 0 else 86400
                while left[w] < j and times[left[w]] < times[j]-seconds:
                    k = left[w]
                    d = side[k]
                    _stable_add(sums, correction, w, d, -amount[k])
                    window_counts[w,d] -= 1
                    if window_counts[w,d] == 0:
                        sums[w,d], correction[w,d] = 0., 0.
                    left[w] += 1
            for k in range(j, stop):
                for w in range(2):
                    result[k, w*2] = np.log1p(max(0., sums[w, 1]+correction[w,1]))
                    result[k, w*2+1] = np.log1p(max(0., sums[w, 0]+correction[w,0]))
                result[k, 4] = np.log1p(amount[k])-logs[1]/counts[1] if counts[1] else np.nan
                result[k, 5] = np.log1p(amount[k])-logs[0]/counts[0] if counts[0] else np.nan
            for k in range(j, stop):
                d = side[k]
                counts[d] += 1
                logs[d] += np.log1p(amount[k])
                for w in range(2):
                    _stable_add(sums, correction, w, d, amount[k])
                    window_counts[w,d] += 1
            j = stop
        start = end
    return result


@njit(cache=True)
def _pair_scan(key, times, side):
    result = np.empty((len(times), 6), np.float32)
    start = 0
    while start < len(times):
        end = start+1
        while end < len(times) and key[end] == key[start]:
            end += 1
        out_n, reverse_n, first, last = 0, 0, -1, -1
        left = np.full(2, start, np.int64)
        window_n = np.zeros(2, np.int64)
        j = start
        while j < end:
            stop = j+1
            while stop < end and times[stop] == times[j]:
                stop += 1
            for w in range(2):
                seconds = 3600 if w == 0 else 86400
                while left[w] < j and times[left[w]] < times[j]-seconds:
                    window_n[w] -= side[left[w]]
                    left[w] += 1
            for k in range(j, stop):
                result[k, 0], result[k, 1] = out_n, reverse_n
                result[k, 2] = times[j]-first if first >= 0 else np.nan
                result[k, 3] = times[j]-last if last >= 0 else np.nan
                result[k, 4], result[k, 5] = window_n[0], window_n[1]
            for k in range(j, stop):
                if side[k]:
                    if first < 0:
                        first = times[j]
                    last = times[j]
                    out_n += 1
                    window_n[0] += 1
                    window_n[1] += 1
                else:
                    reverse_n += 1
            j = stop
        start = end
    return result


def _scatter(values, order):
    result = np.empty_like(values)
    result[order] = values
    return result


def build_study_features(df):
    """Return fixed feature families plus timestamp and label, without fitting on labels."""
    df = df.reset_index(drop=True)
    n = len(df)
    t = df.Timestamp.to_numpy(dtype="datetime64[s]").astype(np.int64)
    if not n or np.any(t[1:] < t[:-1]):
        raise ValueError("Expected nonempty transactions sorted by timestamp")
    accounts = pd.concat([df["From Bank"].astype(str)+"_"+df["Account"],
                          df["To Bank"].astype(str)+"_"+df["Account.1"]], ignore_index=True)
    account, uniques = pd.factorize(accounts)
    del accounts
    src, dst, n_accounts = account[:n], account[n:], len(uniques)
    paid, received = df["Amount Paid"].to_numpy(float), df["Amount Received"].to_numpy(float)
    if not np.isfinite(paid).all() or not np.isfinite(received).all() or (paid < 0).any() or (received < 0).any():
        raise ValueError("Amounts must be finite and nonnegative")
    def codes(column, categories):
        result = pd.Categorical(df[column], categories=categories).codes
        if (result < 0).any():
            raise ValueError("Unknown category: "+column)
        return result
    pay_ccy, recv_ccy = codes("Payment Currency", CURRENCIES), codes("Receiving Currency", CURRENCIES)
    f = {"log_amt_paid": np.log1p(paid), "log_amt_recv": np.log1p(received),
         "hour": df.Timestamp.dt.hour.to_numpy(), "weekday": df.Timestamp.dt.day_of_week.to_numpy(),
         "pay_format": codes("Payment Format", PAY_FORMATS), "pay_currency": pay_ccy,
         "recv_currency": recv_ccy, "same_bank": (df["From Bank"] == df["To Bank"]).to_numpy(),
         "self_txn": src == dst, "cross_currency": pay_ccy != recv_ccy,
         "round_amt": np.isclose(paid % 100, 0)}
    times = np.concatenate([t, t])
    side = np.concatenate([np.ones(n, np.int8), np.zeros(n, np.int8)])
    partner = np.concatenate([dst, src])
    amount = np.concatenate([paid, received])
    order = np.lexsort((times, account))
    a, ts, d = account[order], times[order], side[order]
    history = _scatter(_history_scan(a, ts, d, np.log1p(amount[order])), order)
    for role, sl in (("src", slice(0,n)), ("dst", slice(n,None))):
        for j, name in enumerate(HISTORY_NAMES):
            f[f"{role}_{name}"] = history[sl, j]
    del history
    for seconds, label in WINDOWS:
        stats = _scatter(_window_scan(a, ts, d, partner[order], seconds, n_accounts), order)
        for role, sl in (("src", slice(0,n)), ("dst", slice(n,None))):
            for j, name in enumerate(("out_n", "in_n", "unique_out", "unique_in")):
                f[f"{role}_{name}_{label}"] = stats[sl, j]
    del a, ts, d, stats
    ccy = np.concatenate([pay_ccy, recv_ccy]).astype(np.int64)
    money_key = account*len(CURRENCIES)+ccy
    order = np.lexsort((times, money_key))
    money = _scatter(_money_scan(money_key[order], times[order], side[order], amount[order]), order)
    for role, sl in (("src", slice(0,n)), ("dst", slice(n,None))):
        for j, name in enumerate(("log_out_sum_1h", "log_in_sum_1h", "log_out_sum_24h",
                                  "log_in_sum_24h", "log_amt_vs_out_mean", "log_amt_vs_in_mean")):
            f[f"{role}_{name}"] = money[sl, j]
    del money, money_key
    pair = np.concatenate([src*n_accounts+dst, dst*n_accounts+src])
    order = np.lexsort((times, pair))
    pair_stats = _scatter(_pair_scan(pair[order], times[order], side[order]), order)[:n]
    for j, name in enumerate(("pair_prev_n", "reverse_pair_prev_n", "pair_age_seconds",
                              "pair_seconds_since_previous", "pair_n_1h", "pair_n_24h")):
        f[name] = pair_stats[:, j]
    for role, denominator in (("src", f["src_out_n_24h"]), ("dst", f["dst_in_n_24h"])):
        f[f"{role}_partner_share_24h"] = np.divide(f["pair_n_24h"], denominator,
                                                     out=np.zeros(n,np.float32), where=denominator>0)
    result = pd.DataFrame({name: np.asarray(values, dtype=np.float32) for name, values in f.items()})
    result["timestamp"] = t
    result["is_laundering"] = df["Is Laundering"].to_numpy(np.int8)
    return result


def feature_groups(f):
    history = ["pair_prev_n"]+[f"{role}_{name}" for role in ("src", "dst") for name in HISTORY_NAMES]
    all_columns = [c for c in f if c not in ("timestamp", "is_laundering")]
    relation = [c for c in all_columns if c not in RAW+history and
                ("unique_" in c or "partner_share" in c or c.startswith("pair_") or c.startswith("reverse_pair_"))]
    activity = [c for c in all_columns if c not in RAW+history+relation]
    return {"F0": RAW.copy(), "F1": RAW+history, "F2": RAW+history+activity,
            "F3": RAW+history+relation, "F4": RAW+history+activity+relation}
