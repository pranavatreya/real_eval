from __future__ import annotations
import argparse, os, warnings, hashlib
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning)

EXCL = {"PI0", "PI0_FAST"}
T_BUCKET = 100
EM_ITERS = 50

def canonical(x: str | None) -> str:
    return (x or "").strip()

def sha8(x: str) -> str:
    return hashlib.sha1(x.encode()).hexdigest()[:8]

def read_tbl(root, name, pq):
    fn  = f"{name}.parquet" if pq else f"{name}.csv.gz"
    rdr = pd.read_parquet if pq else (lambda p: pd.read_csv(p, compression="gzip"))
    return rdr(os.path.join(root, fn))

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 2:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# 1 - MMRV (mean max rank violation) as before, after min-max normalize
def mmrv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return np.nan
    return np.mean([
        max([
            abs(b[i] - b[j])
            for j in range(len(a))
            if (a[i] > a[j]) != (b[i] > b[j])
        ] or [0])
        for i in range(len(a))
    ])

# Ranking-only metric: pairwise accuracy of predicted ranking vs ground truth
def pairwise_accuracy(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return np.nan
    correct = total = 0
    for i in range(n):
        for j in range(i+1, n):
            if b[i] == b[j]:
                continue
            total += 1
            if (a[i] - a[j]) * (b[i] - b[j]) > 0:
                correct += 1
    return correct / total if total else np.nan

def safe_newton(x, g, h):
    if abs(h) < 1e-8: return x
    return x - np.clip(g / h, -1.0, 1.0)

def make_frames(root: str, pq: bool):
    ses = read_tbl(root, "sessions", pq)
    eps0 = read_tbl(root, "episodes", pq)

    ses = ses[ses.evaluation_notes.fillna("").str.contains("VALID_SESSION:", case=False)]

    pref_rows = []
    for _, r in ses.iterrows():
        A, B = map(canonical, (r.policyA_name, r.policyB_name))
        if {A.upper(), B.upper()} & EXCL: continue
        pref = None
        for line in (r.evaluation_notes or "").splitlines():
            if line.upper().startswith("PREFERENCE="):
                pref = {"A": 2, "B": 0, "TIE": 1}[line.split("=", 1)[1].upper()]
                break
        if pref is not None:
            pref_rows.append((r.id, A, B, pref))
    pref_df = pd.DataFrame(pref_rows, columns=["sid", "i", "j", "y"])

    eps = eps0.merge(ses[["id"]].rename(columns={"id": "session_id"}), on="session_id")
    eps["letter"] = (
        eps.feedback.fillna("").str.split(";", n=1, expand=True)
        .iloc[:, 0].str.upper()
    )
    eps = eps[eps.letter.isin(["A", "B"])]

    task_rows = []
    for _, r in eps.iterrows():
        pol = canonical(r.policy_name)
        if pol.upper() in EXCL: continue
        key = f"{(r.command or '').strip()}|{r.third_person_camera_type}|{r.third_person_camera_id}"
        t = int(sha8(key), 16) % T_BUCKET
        win = int(float(r.partial_success or 0) >= .8) \
              if pd.isna(r.binary_success) else int(r.binary_success)
        task_rows.append((pol, t, win))
    task_df = pd.DataFrame(task_rows, columns=["policy", "task", "win"])
    return pref_df, task_df, eps0

def fit_davidson(df, rng, iters=EM_ITERS):
    pols = pd.unique(df[["i", "j"]].values.ravel())
    m, idx = len(pols), {p: k for k, p in enumerate(pols)}
    i, j, y = df.i.map(idx), df.j.map(idx), df.y.to_numpy()
    win, loss, tie = (y==2), (y==0), (y==1)

    θ = rng.normal(0., .1, m) # random init
    ν = 0.5
    for _ in tqdm(range(iters), desc="Davidson", leave=False):
        num = np.bincount(i, win+.5*tie, m)
        den = np.zeros(m)
        for a, b in zip(i, j):
            den[a] += 1/(1+np.exp(θ[b]-θ[a])+2*ν*np.exp(.5*(θ[a]+θ[b])))
            den[b] += 1/(1+np.exp(θ[a]-θ[b])+2*ν*np.exp(.5*(θ[a]+θ[b])))
        θ = np.log((num+1e-9)/den); θ -= θ.mean()
        ν = tie.sum() / max(sum(np.exp(.5*(θ[a]+θ[b])) for a, b in zip(i, j)), 1e-8)
    return pd.DataFrame({"policy": pols, "score": θ}).sort_values("score", ascending=False)

def fit_bt(df, rng, iters=EM_ITERS):
    half = df[df.y==1].copy(); half["y"]=2
    d = pd.concat([df[df.y!=1], half])
    pols = pd.unique(d[["i", "j"]].values.ravel())
    m, idx = len(pols), {p:k for k,p in enumerate(pols)}
    i, j, win = d.i.map(idx), d.j.map(idx), (d.y==2).to_numpy()

    θ = rng.normal(0., .1, m) # random init
    for _ in tqdm(range(iters), desc="BT", leave=False):
        p = expit(θ[i]-θ[j])
        num = np.bincount(i, win, m)+np.bincount(j, 1-win, m)
        den = np.bincount(i, p, m)  +np.bincount(j, 1-p, m)
        θ = np.log((num+1e-9)/den)
        θ -= θ.mean()
    return pd.DataFrame({"policy": pols, "score": θ}).sort_values("score", ascending=False)

def fit_elo(df, rng, K=32, base=1200):
    rating = defaultdict(lambda: base)
    for _, r in df.sort_values("sid").iterrows():
        a, b, y = r.i, r.j, r.y
        ra, rb = rating[a], rating[b]
        ea = 1/(1+10**((rb-ra)/400))
        sa = 1. if y==2 else 0. if y==0 else .5
        rating[a], rating[b] = ra+K*(sa-ea), rb+K*((1-sa)-(1-ea))
    bd = (pd.DataFrame({"policy":rating.keys(),"score":rating.values()}).sort_values("score",ascending=False).reset_index(drop=True))
    bd.score = (bd.score-bd.score.mean())/400
    return bd

def fit_policy_task(df, rng, iters=EM_ITERS):
    P, pols = pd.factorize(df.policy)
    p, t, w  = P, df.task.to_numpy(), df.win.to_numpy()

    θ = rng.normal(0., .1, len(pols)) # random init
    τ = rng.normal(0., .1, T_BUCKET) # random init
    for _ in tqdm(range(iters), desc="Policy-Task", leave=False):
        s = expit(θ[p]-τ[t])
        for k in range(len(pols)):
            msk = p==k
            s_i = expit(θ[k]-τ[t[msk]])
            θ[k] = safe_newton(θ[k], (w[msk]-s_i).sum(), -(s_i*(1-s_i)).sum())
        θ -= θ.mean()
        for tt in range(T_BUCKET):
            msk = t==tt
            s_t = expit(θ[p[msk]]-τ[tt])
            τ[tt] = safe_newton(τ[tt], (s_t-w[msk]).sum(), -(s_t*(1-s_t)).sum())
        τ -= τ.mean()
    return pd.DataFrame({"policy":pols,"score":θ}).sort_values("score",ascending=False)

def fit_hybrid(df,
               rng,
               iters: int = EM_ITERS,
               T: int = T_BUCKET,
               step_clip: float = 1.0,
               l2_psi: float = 1e-2):

    pols = pd.unique(df[["i", "j"]].values.ravel())
    P, idx = len(pols), {p: k for k, p in enumerate(pols)}
    i, j = df.i.map(idx).to_numpy(), df.j.map(idx).to_numpy()
    y = df.y.to_numpy()
    win, loss, tie = (y == 2), (y == 0), (y == 1)

    θ = rng.normal(0., .1, P) # random init
    τ = rng.normal(0., .1, T)
    ψ = np.zeros((P, T)) # zero init (could also randomize though)
    π = np.full(T, 1 / T)
    ν = 0.5

    def clip_step(x, g, h):
        if abs(h) < 1e-8: return x
        return x - np.clip(g / h, -step_clip, step_clip)

    for _ in tqdm(range(iters), desc="HybridEM-ψ", leave=False):
        δ_i = θ[i][:, None] + ψ[i]
        δ_j = θ[j][:, None] + ψ[j]
        logit = δ_i - δ_j - τ
        s_i = expit(logit)
        s_j = 1.0 - s_i

        pW = s_i * (1 - s_j)
        pL = (1 - s_i) * s_j
        pT = 2 * ν * np.sqrt(pW * pL)
        like = pW * win[:, None] + pL * loss[:, None] + pT * tie[:, None]

        γ = π * np.clip(like, 1e-12, None)
        γ /= γ.sum(1, keepdims=True)

        # θ update
        for pidx in range(P):
            mi, mj = (i == pidx), (j == pidx)
            g = h = 0.0
            for t in range(T):
                si, sj = s_i[:, t], s_j[:, t]
                if mi.any():
                    g += (( win[mi]*(1 - sj[mi]) - loss[mi]*sj[mi]
                          + tie[mi]*(sj[mi] - si[mi])) * γ[mi, t]).sum()
                    h -= ((si[mi]*(1 - si[mi]) + sj[mi]*(1 - sj[mi]))
                          * γ[mi, t]).sum()
                if mj.any():
                    g += ((loss[mj]*(1 - si[mj]) - win[mj]*si[mj]
                          + tie[mj]*(si[mj] - sj[mj])) * γ[mj, t]).sum()
                    h -= ((si[mj]*(1 - si[mj]) + sj[mj]*(1 - sj[mj]))
                          * γ[mj, t]).sum()
            θ[pidx] = clip_step(θ[pidx], g, h)
        θ -= θ.mean()

        # ψ update
        for pidx in range(P):
            for t in range(T):
                mi, mj = (i == pidx), (j == pidx)
                si, sj = s_i[:, t], s_j[:, t]
                g = h = 0.0
                if mi.any():
                    g += (( win[mi]*(1 - sj[mi]) - loss[mi]*sj[mi]
                          + tie[mi]*(sj[mi] - si[mi])) * γ[mi, t]).sum()
                    h -= ((si[mi]*(1 - si[mi]) + sj[mi]*(1 - sj[mi]))
                          * γ[mi, t]).sum()
                if mj.any():
                    g += (( loss[mj]*(1 - si[mj]) - win[mj]*si[mj]
                          + tie[mj]*(si[mj] - sj[mj])) * γ[mj, t]).sum()
                    h -= ((si[mj]*(1 - si[mj]) + sj[mj]*(1 - sj[mj]))
                          * γ[mj, t]).sum()
                g += l2_psi * ψ[pidx, t]
                h -= l2_psi
                ψ[pidx, t] = clip_step(ψ[pidx, t], g, h)
        ψ -= ψ.mean(axis=1, keepdims=True)

        # τ update
        for t in range(T):
            si, sj = s_i[:, t], s_j[:, t]
            g = ((win * (-si * (1 - sj))
                 + loss * (sj * (1 - si))
                 + tie  * 0.5 * (si - sj)) * γ[:, t]).sum()
            h = -((si * (1 - si) + sj * (1 - sj)) * γ[:, t]).sum()
            τ[t] = clip_step(τ[t], g, h)
        τ -= τ.mean()

        π = γ.mean(0); π /= π.sum()
        ν = 0.5 * ( (pT * γ).sum() / max((pW * γ).sum(), 1e-9))

    #θ = (θ - θ.min()) / (θ.max() - θ.min() + 1e-9)
    #θ = expit(θ)

    return (pd.DataFrame({"policy": pols, "score": θ})
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

def fit_mean_success(eps_full, valid_sid):
    eps = eps_full[eps_full.session_id.isin(valid_sid)].copy()
    eps["letter"] = eps.feedback.fillna("").str.split(";", n=1, expand=True).iloc[:,0].str.upper()
    eps = eps[eps.letter=="A"]
    eps["policy_c"] = eps.policy_name.apply(canonical)
    eps = eps[~eps.policy_c.str.upper().isin(EXCL)]
    bd = (eps.groupby("policy_c")["partial_success"].mean()
           .reset_index().rename(columns={"policy_c":"policy","partial_success":"score"})
           .sort_values("score",ascending=False).reset_index(drop=True))
    return bd

def fit_bt_taskvar_ties(df: pd.DataFrame,
                        rng,
                        max_outer: int = 50,
                        inner_bt_iter: int = 50,
                        min_sigma: float = 0.1):

    df_work = df.copy()
    ties = df_work[df_work.y == 1]
    if not ties.empty:
        half = ties.copy()
        half["y"] = 2
        df_work = pd.concat([df_work[df_work.y != 1], half], ignore_index=True)

    pols = pd.unique(df_work[["i", "j"]].values.ravel())
    m, idx = len(pols), {p:k for k,p in enumerate(pols)}
    i = df_work.i.map(idx).to_numpy()
    j = df_work.j.map(idx).to_numpy()
    y = df_work.y.to_numpy()
    win = (y == 2).astype(float)
    sid = df_work.sid.to_numpy()

    θ = rng.normal(0., .1, m) # random init
    σ_t = defaultdict(lambda: rng.uniform(0.8, 1.2)) # random init

    for _ in tqdm(range(max_outer), desc="BT-TaskVar-Ties", leave=False):
        scale = np.array([σ_t[s] for s in sid])
        for _ in range(inner_bt_iter):
            z = (θ[i]-θ[j]) / scale
            p = expit(z)
            num = np.zeros(m); den = np.zeros(m)
            np.add.at(num, i, win/scale)
            np.add.at(num, j, (1-win)/scale)
            np.add.at(den, i, p/scale)
            np.add.at(den, j, (1-p)/scale)
            θ = np.log((num+1e-9)/(den+1e-9)); θ -= θ.mean()

        for s in np.unique(sid):
            mask = sid == s
            dif = np.abs(θ[i[mask]] - θ[j[mask]])
            σ_t[s] = max(dif.std(ddof=0), min_sigma)

    return (pd.DataFrame({"policy": pols, "score": θ})
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

def board_txt(df):
    return "\n".join(
        f"{k+1:2d}. {p} ({s:.3f})"
        for k, (p, s) in enumerate(zip(df.policy, df.score))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--parquet", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pref_df, task_df, eps_all = make_frames(args.dump, args.parquet)

    sids = rng.permutation(pref_df.sid.unique())
    cut = int(len(sids) * .9)
    train_pref = pref_df[pref_df.sid.isin(sids[:cut])]
    test_pref  = pref_df[~pref_df.sid.isin(sids[:cut]) & (pref_df.y != 1)]

    boards = {
        "Davidson"     : fit_davidson       (train_pref, rng),
        "BT"           : fit_bt             (train_pref, rng),
        "Elo"          : fit_elo            (train_pref, rng),
        "PolicyTask"   : fit_policy_task    (task_df,    rng),
        "Hybrid"       : fit_hybrid         (train_pref, rng),
        "BT-TaskVar"   : fit_bt_taskvar_ties(train_pref, rng),
        "MeanSucc"     : fit_mean_success   (eps_all, pref_df.sid.unique()),
    }

    eps_all["policy_canon"] = eps_all.policy_name.apply(canonical)
    gt = (
        eps_all[~eps_all.policy_canon.str.upper().isin(EXCL)]
        .groupby("policy_canon")["partial_success"].mean()
        .dropna().reset_index()
        .rename(columns={"policy_canon": "policy", "partial_success": "gt"})
    )

    rows = []
    for n, b in boards.items():
        pm = b.set_index("policy")["score"]
        # test-set accuracy
        acc = np.mean([
            int((pm[i] > pm[j]) == (y == 2))
            for _, (_, i, j, y) in test_pref.iterrows()
            if i in pm and j in pm
        ])
        # merge for continuous comparisons
        merged = b.merge(gt, on="policy")
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
        merged["gt"]    = pd.to_numeric(merged["gt"],    errors="coerce")
        # Pearson and 1 - MMRV
        rho = pearson(merged["score"], merged["gt"])
        mm  = 1 - mmrv(merged["score"], merged["gt"])
        # Ranking-only metrics
        sr, _ = spearmanr(merged["score"], merged["gt"], nan_policy='omit')
        kt, _ = kendalltau(merged["score"], merged["gt"], nan_policy='omit')
        pw = pairwise_accuracy(merged["score"], merged["gt"])
        rows.append((n, acc, rho, mm, sr, kt, pw))

    # determine maxima for bolding
    max_acc      = max(r[1] for r in rows)
    max_pearson  = max(r[2] for r in rows)
    max_mm       = max(r[3] for r in rows)
    max_spearman = max(r[4] for r in rows)
    max_kendall  = max(r[5] for r in rows)
    max_pairwise = max(r[6] for r in rows)

    # build markdown
    md = [
        "# Real-Eval snapshot",
        f"*seed {args.seed}* — train {len(train_pref.sid.unique())}  | "
        f"test {len(test_pref.sid.unique())}\n",
        "| model | acc | pearson ρ | 1−MMRV | spearman ρ | kendall τ | pairwise_acc |",
        "|-------|:---:|:---------:|:------:|:---------:|:--------:|:-------------:|"
    ]
    for n, a, p, m, sr_val, kt_val, pw_val in rows:
        a_str  = f"**{a:.3f}**"  if a  == max_acc      else f"{a:.3f}"
        p_str  = f"**{p:.3f}**"  if p  == max_pearson  else f"{p:.3f}"
        m_str  = f"**{m:.3f}**"  if m  == max_mm       else f"{m:.3f}"
        sr_str = f"**{sr_val:.3f}**" if sr_val == max_spearman else f"{sr_val:.3f}"
        kt_str = f"**{kt_val:.3f}**" if kt_val == max_kendall  else f"{kt_val:.3f}"
        pw_str = f"**{pw_val:.3f}**" if pw_val == max_pairwise else f"{pw_val:.3f}"
        md.append(f"| {n} | {a_str} | {p_str} | {m_str} | {sr_str} | {kt_str} | {pw_str} |")

    md.append("\n## Leaderboards\n")
    for n, b in boards.items():
        md.append(f"### {n}\n{board_txt(b)}\n")

    with open("real_eval_report.md", "w") as f:
        f.write("\n".join(md))
    print("✓ real_eval_report.md  (seed=", args.seed, ")")

if __name__ == "__main__":
    main()
