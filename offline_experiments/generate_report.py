import os, datetime, hashlib, subprocess, warnings
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scipy.special import expit

# ——————————————————————————— config ————————————————————————————
DB_URL = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
EXCLUDE = {"PI0", "PI0_FAST"}
T_BUCKET = 8 # discrete task buckets
TEST_FR = 0.10
EM_ITERS = 60
VERBOSE = False
rng = np.random.default_rng(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ————————————————————————— helpers ————————————————————————————
def canonical(x): return (x or "").strip()
def sha8(x): return hashlib.sha1(x.encode()).hexdigest()[:8]
def dbg(*a): print(*a) if VERBOSE else None
def safe_newton(x, g, h): return x - g / (h if abs(h) > 1e-8 else -1e-8)

def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 2: return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def mmrv(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 2: return np.nan
    a, b = a[m], b[m]
    return np.mean([
        max([abs(b[i] - b[j]) for j in range(len(a))
             if (a[i] > a[j]) != (b[i] > b[j])] or [0])
        for i in range(len(a))
    ])

# ——————————————————— load Postgres tables ————————————————————
eng = create_engine(DB_URL, pool_pre_ping=True)
with eng.begin() as con:
    ses = pd.read_sql(text(
        "SELECT * FROM sessions WHERE evaluation_notes ILIKE 'VALID_SESSION:%'"), con)
    eps = pd.read_sql(text("""
        SELECT e.* FROM episodes e
        JOIN sessions s ON s.id=e.session_id
        WHERE s.evaluation_notes ILIKE 'VALID_SESSION:%'"""), con)

# ——————————— preference dataframe (A/B tuples) ————————————
pairs = []
for _, r in ses.iterrows():
    A, B = map(canonical, (r.policyA_name, r.policyB_name))
    if A.upper() in EXCLUDE or B.upper() in EXCLUDE: continue
    pref = None
    for line in (r.evaluation_notes or "").splitlines():
        line = line.strip().upper()
        if line.startswith("PREFERENCE="):
            pref = {"A": 2, "B": 0, "TIE": 1}[line.split("=")[1]]
            break
    if pref is not None:
        pairs.append((r.id, A, B, pref))
pref_df = pd.DataFrame(pairs, columns=["sid", "i", "j", "y"])

# ————————— single-policy episodes for Policy-Task EM ————————
rows = []
for _, r in eps.iterrows():
    pol = canonical(r.policy_name)
    if pol.upper() in EXCLUDE: continue
    cmd = (r.command or "").strip()
    key = f"{cmd}|{r.third_person_camera_type or ''}|{r.third_person_camera_id or ''}"
    t = int(sha8(key), 16) % T_BUCKET
    win = int(float(r.partial_success or 0) >= 0.80) if pd.isna(r.binary_success) else int(r.binary_success)
    rows.append((pol, t, win, cmd, r.gcs_left_cam_path))
task_df = pd.DataFrame(rows, columns=["policy", "task", "win", "instr", "img"])

# ——————————— train / test split on preference sessions ———————
sid = np.unique(pref_df.sid)
rng.shuffle(sid)
train = pref_df[pref_df.sid.isin(sid[:int(len(sid) * (1 - TEST_FR))])]
test  = pref_df[~pref_df.sid.isin(train.sid)]

# ══════════════════════════════════════════════════════════════
# 1)  Davidson  (ties, no task)
# ══════════════════════════════════════════════════════════════
def davidson_mm(df, iters=200):
    # policy -> index
    pols = pd.unique(pd.concat([df.i, df.j]))
    idmap = {p: k for k, p in enumerate(pols)}
    m = len(pols)
    # encoded arrays
    i = df.i.map(idmap).to_numpy()
    j = df.j.map(idmap).to_numpy()
    y = df.y.to_numpy()
    win, loss, tie = (y == 2), (y == 0), (y == 1)
    θ = np.zeros(m)
    ν = 0.5
    for _ in range(iters):
        # MM update for θ
        num = np.bincount(i, weights=win + 0.5 * tie, minlength=m)
        den = np.zeros(m)
        for a, b in zip(i, j):
            den[a] += 1 / (1 + np.exp(θ[b] - θ[a]) + 2 * ν * np.exp(0.5 * (θ[a] + θ[b])))
            den[b] += 1 / (1 + np.exp(θ[a] - θ[b]) + 2 * ν * np.exp(0.5 * (θ[a] + θ[b])))
        θ = np.log((num + 1e-9) / den)
        θ -= θ.mean()
        # MM update for ν
        top = tie.sum()
        bot = 0.0
        for a, b in zip(i, j):
            bot += (
                2 * ν * np.exp(0.5 * (θ[a] + θ[b]))
                / (np.exp(θ[a]) + np.exp(θ[b]) + 2 * ν * np.exp(0.5 * (θ[a] + θ[b])))
            )
        ν = top / max(bot, 1e-8)
    return (
        pd.DataFrame({"policy": pols, "score": θ})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

tbl_dav = davidson_mm(train)

# ══════════════════════════════════════════════════════════════
# 2)  Policy-Task EM  (single-policy episodes only)
# ══════════════════════════════════════════════════════════════
def em_policy_task(df, iters=EM_ITERS):
    P, pols = pd.factorize(df.policy)
    p = P
    t = df.task.to_numpy()
    w = df.win.to_numpy()
    θ = np.zeros(len(pols))
    τ = np.zeros(T_BUCKET)
    π = np.full(T_BUCKET, 1 / T_BUCKET)

    for _ in range(iters):
        s = expit(θ[p] - τ[t])  # success probs
        # θ update
        for idx in range(len(pols)):
            mask = p == idx
            s_i = expit(θ[idx] - τ[t[mask]])
            θ[idx] = safe_newton(θ[idx], (w[mask] - s_i).sum(), -(s_i * (1 - s_i)).sum())
        θ -= θ.mean()
        # τ update
        for tt in range(T_BUCKET):
            mask = t == tt
            s_t = expit(θ[p[mask]] - τ[tt])
            τ[tt] = safe_newton(τ[tt], (s_t - w[mask]).sum(), -(s_t * (1 - s_t)).sum())
        τ -= τ.mean()
        π = np.bincount(t, minlength=T_BUCKET) + 1e-3
        π /= π.sum()

    board = pd.DataFrame({"policy": pols, "score": θ}).sort_values("score", ascending=False)
    return board.reset_index(drop=True), τ, π

tbl_pt, τ_pt, π_pt = em_policy_task(task_df)

# ══════════════════════════════════════════════════════════════
# 3)  Hybrid EM (A/B only, task-aware win model)
# ══════════════════════════════════════════════════════════════
def em_hybrid(df, iters=EM_ITERS):
    pols = pd.unique(pd.concat([df.i, df.j]))
    idmap = {p: k for k, p in enumerate(pols)}
    P = len(pols)
    i = df.i.map(idmap).to_numpy()
    j = df.j.map(idmap).to_numpy()
    y = df.y.to_numpy()
    win, loss, tie = (y == 2), (y == 0), (y == 1)

    θ = np.zeros(P)
    τ = rng.normal(scale=0.1, size=T_BUCKET)
    π = np.full(T_BUCKET, 1 / T_BUCKET)

    for _ in range(iters):
        s_i = expit(θ[i][:, None] - τ[None, :])
        s_j = expit(θ[j][:, None] - τ[None, :])
        p_win  = s_i * (1 - s_j)
        p_loss = (1 - s_i) * s_j
        p_tie  = 1 - p_win - p_loss
        like = p_win * win[:, None] + p_loss * loss[:, None] + p_tie * tie[:, None]
        like = np.maximum(like, 1e-20)
        γ = π[None, :] * like
        γ /= γ.sum(1, keepdims=True)

        # θ update
        for pidx in range(P):
            mi, mj = (i == pidx), (j == pidx)
            g = h = 0.0
            for tt in range(T_BUCKET):
                if mi.any():
                    si, sj = s_i[mi, tt], s_j[mi, tt]
                    di = si * (1 - si)
                    g += ((win[mi] * (1 - sj) - loss[mi] * sj + tie[mi] * (sj - si))
                          * γ[mi, tt]).sum()
                    h -= (di * γ[mi, tt]).sum()
                if mj.any():
                    si, sj = s_i[mj, tt], s_j[mj, tt]
                    dj = sj * (1 - sj)
                    g += ((loss[mj] * (1 - si) - win[mj] * si + tie[mj] * (si - sj))
                          * γ[mj, tt]).sum()
                    h -= (dj * γ[mj, tt]).sum()
            θ[pidx] = safe_newton(θ[pidx], g, h)
        θ -= θ.mean()

        # τ update
        for tt in range(T_BUCKET):
            si, sj = s_i[:, tt], s_j[:, tt]
            g = ((γ[:, tt] *
                  (win * (-si * (1 - sj)) +
                   loss * (sj * (1 - si)) +
                   tie * (si - sj) * 0.5)).sum())
            h = -((γ[:, tt] * (si * (1 - si) + sj * (1 - sj)) * 0.5).sum())
            τ[tt] = safe_newton(τ[tt], g, h)
        τ -= τ.mean()

        π = γ.mean(0)
        π /= π.sum()

    θ[~np.isfinite(θ)] = 0
    board = pd.DataFrame({"policy": pols, "score": θ}).sort_values("score", ascending=False)
    return board.reset_index(drop=True), τ, π

tbl_hy, τ_hy, π_hy = em_hybrid(train)

# ══════════════════════════════════════════════════════════════
# 4)  Online Elo  (K-factor 32, ties = 0.5)
# ══════════════════════════════════════════════════════════════
def elo_online(df, K=32, base=1200):
    rating = {}
    for _, r in df.sort_values("sid").iterrows():
        a, b, y = r.i, r.j, r.y
        ra = rating.get(a, base)
        rb = rating.get(b, base)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        sa = 1.0 if y == 2 else 0.0 if y == 0 else 0.5
        sb = 1 - sa
        rating[a] = ra + K * (sa - ea)
        rating[b] = rb + K * (sb - eb)
    board = pd.DataFrame({"policy": rating.keys(), "score": rating.values()})
    return board.sort_values("score", ascending=False).reset_index(drop=True)

tbl_elo = elo_online(train)

# ══════════════════════════════════════════════════════════════
# 5)  Classical BT (no ties)  – ties counted as 1/2 win each
# ══════════════════════════════════════════════════════════════
def bt_mm(df, iters=200):
    # convert tie rows to two half-rows
    half = df[df.y == 1].copy()
    half["y"] = 2 # treat as win for i
    df_bt = pd.concat([df[df.y != 1], half], ignore_index=True)

    pols = pd.unique(pd.concat([df_bt.i, df_bt.j]))
    idx = {p: k for k, p in enumerate(pols)}
    m = len(pols)
    i = df_bt.i.map(idx).to_numpy()
    j = df_bt.j.map(idx).to_numpy()
    win = (df_bt.y == 2).to_numpy()

    θ = np.zeros(m)
    for _ in range(iters):
        num = np.zeros(m)
        den = np.zeros(m)
        p_ij = 1 / (1 + np.exp(θ[j] - θ[i]))
        num += np.bincount(i, weights=win, minlength=m)
        num += np.bincount(j, weights=1 - win, minlength=m)
        den += np.bincount(i, weights=p_ij, minlength=m)
        den += np.bincount(j, weights=1 - p_ij, minlength=m)
        θ = np.log((num + 1e-9) / den)
        θ -= θ.mean()
    return (
        pd.DataFrame({"policy": pols, "score": θ})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

tbl_bt = bt_mm(train)

# ——————————— ground-truth mean partial success ———————————
eps["policy_canon"] = eps.policy_name.apply(canonical)
gt = (
    eps[~eps.policy_canon.str.upper().isin(EXCLUDE)]
    .groupby("policy_canon")["partial_success"]
    .mean()
    .dropna()
    .reset_index()
    .rename(columns={"policy_canon": "policy", "partial_success": "gt"})
)

# ——————————————— metrics functions ————————————————
def accuracy(df, board):
    sc = board.set_index("policy")["score"]
    ok = 0
    for _, r in df.iterrows():
        if r.i in sc and r.j in sc:
            pred = 2 if sc[r.i] > sc[r.j] else 0 if sc[r.j] > sc[r.i] else 1
            ok += int(pred == r.y)
    return ok / len(df)

def metric(board, name):
    merged = board.merge(gt, on="policy", how="inner")

    sc_vals = pd.to_numeric(merged["score"], errors="coerce").to_numpy(dtype=float)
    gt_vals = pd.to_numeric(merged["gt"],    errors="coerce").to_numpy(dtype=float)

    return (
        name,
        accuracy(test, board),
        pearson(sc_vals, gt_vals),
        1 - mmrv(sc_vals, gt_vals),
    )

rows = [
    metric(tbl_dav, "Davidson"),
    metric(tbl_bt,  "BT"),
    metric(tbl_elo, "Elo"),
    metric(tbl_pt,  "Policy-task"),
    metric(tbl_hy,  "Hybrid"),
]

# ——————————— rankings as text blocks ———————————————
def rank_txt(df): return "\n".join(f"{k+1:2d}. {p} ({s:.3f})"
                                   for k, (p, s) in enumerate(zip(df.policy, df.score)))

rank_d, rank_bt, rank_elo, rank_p, rank_h = map(rank_txt,
    (tbl_dav, tbl_bt, tbl_elo, tbl_pt, tbl_hy))

# ——————————— representative task lines —————————————
task_lines = []
for tt in range(T_BUCKET):
    ex = task_df[task_df.task == tt].head(1)
    if ex.empty: continue
    task_lines.append(
        f"* **Bucket {tt} (τ={τ_pt[tt]:.2f})** "
        f"“{ex.instr.iat[0][:60]}…”  `{ex.img.iat[0]}`")

# ——————————— Markdown template ————————————————
FMT_REPORT = """# Real-Eval snapshot ({timestamp:%Y-%m-%d %H:%M})

| model | accuracy | Pearson | 1 − MMRV |
|-------|:-------:|:-------:|:--------:|
{metrics_md}

## Leaderboards (BT coefficients)

### Davidson
{rank_d}

### BT (no ties)
{rank_bt}

### Elo (online)
{rank_elo}

### Policy-task EM
{rank_p}

### Hybrid EM
{rank_h}

## Task buckets τ (Policy-task EM)

{task_lines}

Mixing weights
*Policy-task* π̂ = {pi_pt}
*Hybrid* π̂      = {pi_hy}
"""

# ——————————— write Markdown ————————————————
metrics_md = "\n".join(
    f"| {n} | {a:.3f} | {p:.3f} | {m:.3f} |"
    for n, a, p, m in rows
)

report_md = FMT_REPORT.format(
    timestamp=datetime.datetime.now(),
    metrics_md=metrics_md,
    rank_d=rank_d,
    rank_bt=rank_bt,
    rank_elo=rank_elo,
    rank_p=rank_p,
    rank_h=rank_h,
    task_lines="\n".join(task_lines),
    pi_pt=", ".join(f"{x:.3f}" for x in π_pt),
    pi_hy=", ".join(f"{x:.3f}" for x in π_hy),
)

with open("real_eval_report.md", "w") as f_out:
    f_out.write(report_md)
