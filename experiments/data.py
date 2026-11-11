"""Shared deterministic window construction and native NetFlow features."""
import numpy as np
import pandas as pd
def windowize(log, flow_y, attack, t_end, L, group=None):
    """Disjoint consecutive windows. If group is given, windows never mix groups."""
    if group is None:
        n = len(log) // L
        sel = np.arange(n * L)
    else:
        # rows are already sorted by (group, time); cut each group into chunks of L
        starts = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1]
        ends = np.r_[starts[1:], len(group)]
        keep = []
        for s, e in zip(starts, ends):
            m = (e - s) // L
            if m:
                keep.append(np.arange(s, s + m * L))
        sel = np.concatenate(keep) if keep else np.zeros(0, int)
        n = len(sel) // L
    x = log[sel].reshape(n, L, -1).transpose(0, 2, 1).astype(np.float32)
    fy = flow_y[sel].reshape(n, L).astype(np.int8)
    att = attack[sel].reshape(n, L)
    te = t_end[sel].reshape(n, L)
    tags = np.array(["+".join(sorted(set(r) - {"Benign"})) or "Benign" for r in att])
    return dict(x=x, y=fy.max(1).astype(np.int64), flow_y=fy, tags=tags, t_end=te)


NF_COLS = ["FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
           "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
           "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV", "Label", "Attack"]


def nf_features(df):
    dur = df["FLOW_DURATION_MILLISECONDS"].to_numpy(np.float64)
    ip, op = df["IN_PKTS"].to_numpy(np.float64), df["OUT_PKTS"].to_numpy(np.float64)
    ib, ob = df["IN_BYTES"].to_numpy(np.float64), df["OUT_BYTES"].to_numpy(np.float64)
    fl = df["TCP_FLAGS"].to_numpy(np.int64)
    f = np.stack([dur, ip, op, ib, ob,
                  df["SRC_TO_DST_IAT_AVG"].to_numpy(np.float64),
                  df["SRC_TO_DST_IAT_STDDEV"].to_numpy(np.float64),
                  (ib + ob) / np.maximum(ip + op, 1),
                  np.maximum(df["MAX_IP_PKT_LEN"].to_numpy(np.float64) - df["MIN_IP_PKT_LEN"].to_numpy(np.float64), 0),
                  ((fl & 2) > 0).astype(np.float64), ((fl & 16) > 0).astype(np.float64),
                  ip / np.maximum(dur / 1000.0, 1e-3)], 1)
    return np.log1p(np.clip(f, 0, None)).astype(np.float32)


def nf_splits(name, t):
    """Chronological split boundaries (ms)."""
    if name == "NF-UNSW-NB15-v3":
        day2 = pd.Timestamp("2015-02-01").value // 10**6
        d2 = np.sort(t[t >= day2])
        cut = d2[int(0.2 * len(d2))]
        return {"train": (t < day2), "validation": (t >= day2) & (t < cut), "test": (t >= cut)}
    if name == "NF-ToN-IoT-v3":
        a = pd.Timestamp("2019-04-25").value // 10**6
        b = pd.Timestamp("2019-04-26").value // 10**6
        return {"train": t < a, "validation": (t >= a) & (t < b), "test": t >= b}
    raise ValueError(name)

