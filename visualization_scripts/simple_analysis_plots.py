import base64, io, threading, time, datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, redirect, url_for
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo

# --------------------------------------------------------------
#  Database connection
# --------------------------------------------------------------
DB_URL = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
engine = create_engine(DB_URL, pool_pre_ping=True)

# --------------------------------------------------------------
#  Global cache for rendered PNGs / html snippets
# --------------------------------------------------------------
CACHE = {
    "policy_bar": "",
    "elo_table": "",
    "uni_bar": "",
    "preference_pie": "",
    "progress_hist": "",
    "last_update": "never"
}
REFRESH_INTERVAL_H = 4  # hours

# --------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------
EXCLUDE_POLICIES = {"PI0", "PI0_FAST"}  # exclude everywhere

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}'/>"

# --------------------------------------------------------------
#  (1) Average partial‑success per policy
# --------------------------------------------------------------
def build_policy_bar(df_episodes):
    df = df_episodes[~df_episodes["policy_name"].str.upper().isin(EXCLUDE_POLICIES)]
    grp = (
        df.groupby("policy_name")["partial_success"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    grp.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Average partial‑success")
    ax.set_title("Average partial‑success by policy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", ls=":", alpha=0.5)
    return fig_to_base64(fig)

# --------------------------------------------------------------
#  (2) Elo leaderboard from A/B preference
# --------------------------------------------------------------
def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def build_elo_table(df_sessions):
    prefs = []
    for _, row in df_sessions.iterrows():
        # skip if either A or B is in the exclude list
        if row["policyA_name"].upper() in EXCLUDE_POLICIES or \
           row["policyB_name"].upper() in EXCLUDE_POLICIES:
            continue
        notes = row["evaluation_notes"] or ""
        for l in notes.split("\n"):
            l = l.strip().upper()
            if l.startswith("PREFERENCE="):
                prefs.append((row["policyA_name"], row["policyB_name"], l.split("=")[1]))
                break

    elo = defaultdict(lambda: 1200.0)
    games = defaultdict(int)

    for pA, pB, outcome in prefs[::-1]:  # oldest first
        rA, rB = elo[pA], elo[pB]
        eA, eB = expected_score(rA, rB), expected_score(rB, rA)
        if outcome == "A":   sA, sB = 1.0, 0.0
        elif outcome == "B": sA, sB = 0.0, 1.0
        else:                sA = sB = 0.5
        K = 32
        elo[pA] = rA + K * (sA - eA)
        elo[pB] = rB + K * (sB - eB)
        games[pA] += 1
        games[pB] += 1

    leaderboard = (
        pd.DataFrame({
            "policy": elo.keys(),
            "elo": elo.values(),
            "n_eval": [games[p] for p in elo.keys()]
        })
        .sort_values("elo", ascending=False)
    )
    return leaderboard.to_html(index=False,
                               classes="table table-striped text-center")

# --------------------------------------------------------------
#  (3) Evaluations per university
# --------------------------------------------------------------
UNI_CANONICAL = {
    "UNIVERSITY OF CALIFORNIA BERKELEY": "Berkeley",
    "UCB": "Berkeley",
    "BERKELEY": "Berkeley",
    "STANFORD": "Stanford",
    "U PENN": "UPenn",
    "UNIVERSITY OF PENNSYLVANIA": "UPenn",
    "UPENN": "UPenn",
    "UNIVERSITY OF WASHINGTON": "UW",
    "UNVERSITY OF WASHINGTON": "UW",
    "UNIVERSITY OF WASHGINTON": "UW",
    "UW": "UW",
    "MILA": "UMontreal",
    "UNIVERSITY MONTREAL": "UMontreal",
    "UOF MONTREAL": "UMontreal",
    "YONSEI": "Yonsei",
    "UT AUSTIN": "UT Austin",
    "UNIVERSITY OF TEXAS AT AUSTIN": "UT Austin"
}
TARGET_UNIS = ["Berkeley", "Stanford", "UW", "UPenn",
               "UMontreal", "Yonsei", "UT Austin"]

def canonicalize_uni(raw):
    if not raw:
        return "Other"
    upper = raw.upper().strip()
    for key, canon in UNI_CANONICAL.items():
        if key in upper:
            return canon
    print(f"Error: a university was not able to be matched: {raw}")
    exit()
    return "Other"

def build_uni_bar(df_sessions):
    df_sessions["uni"] = df_sessions["evaluation_location"].apply(canonicalize_uni)
    counts = df_sessions["uni"].value_counts()
    # ensure all target unis are present + "Other"
    counts = counts.reindex(TARGET_UNIS + ["Other"]).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.sort_values(ascending=False).plot(kind="bar", ax=ax, color="seagreen")
    ax.axhline(100, ls="--", color="red", label="target = 100")
    ax.set_ylabel("# evaluations")
    ax.set_title("Evaluations per university (VALID sessions)")
    ax.legend()
    return fig_to_base64(fig)

# --------------------------------------------------------------
#  (4) Pie chart of A/B preference outcomes
# --------------------------------------------------------------
def build_preference_pie(df_sessions):
    counts = {"A": 0, "B": 0, "TIE": 0}
    for _, row in df_sessions.iterrows():
        if row["policyA_name"].upper() in EXCLUDE_POLICIES or \
           row["policyB_name"].upper() in EXCLUDE_POLICIES:
            continue
        notes = row["evaluation_notes"] or ""
        for line in notes.split("\n"):
            line = line.strip().upper()
            if line.startswith("PREFERENCE="):
                counts[line.split("=")[1]] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["A preferred", "B preferred", "Tie"]
    ax.pie([counts["A"], counts["B"], counts["TIE"]],
           labels=labels, autopct="%1.0f%%", startangle=140)
    ax.set_title("A/B preference outcomes")
    return fig_to_base64(fig)

# --------------------------------------------------------------
#  (5) Progress‑score histogram (10‑pt bins)
# --------------------------------------------------------------
def build_progress_hist(df_episodes):
    df = df_episodes[~df_episodes["policy_name"].str.upper().isin(EXCLUDE_POLICIES)]
    scores = (df["partial_success"].dropna() * 100).clip(0, 100)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(scores, bins=np.arange(0, 110, 10), color="purple", edgecolor="black",
            rwidth=0.85)
    ax.set_xlabel("Progress score (%)")
    ax.set_ylabel("# episodes")
    ax.set_title("Distribution of progress scores (all VALID episodes)")
    ax.set_xticks(np.arange(0, 110, 10))
    ax.grid(axis="y", ls=":", alpha=0.5)
    return fig_to_base64(fig)

# --------------------------------------------------------------
#  Main refresh function
# --------------------------------------------------------------
def refresh_cache():
    with engine.begin() as conn:
        ses = pd.read_sql(text("""
            SELECT * FROM sessions
            WHERE evaluation_notes ILIKE 'VALID_SESSION:%'
        """), conn)

        eps = pd.read_sql(text("""
            SELECT e.* FROM episodes e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.evaluation_notes ILIKE 'VALID_SESSION:%'
        """), conn)

    CACHE["policy_bar"]    = build_policy_bar(eps)
    CACHE["elo_table"]     = build_elo_table(ses)
    CACHE["uni_bar"]       = build_uni_bar(ses)
    CACHE["preference_pie"]= build_preference_pie(ses)
    CACHE["progress_hist"] = build_progress_hist(eps)
    pt_now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    CACHE["last_update"] = pt_now.strftime("%Y-%m-%d %H:%M PT")
    print(f"[dashboard] cache refreshed @ {CACHE['last_update']}")

# --------------------------------------------------------------
#  Background scheduler
# --------------------------------------------------------------
def scheduler_loop():
    while True:
        refresh_cache()
        time.sleep(REFRESH_INTERVAL_H * 3600)

threading.Thread(target=scheduler_loop, daemon=True).start()

# --------------------------------------------------------------
#  Flask application
# --------------------------------------------------------------
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Real‑Eval Dashboard</title>
  <style>
    body {font-family: Arial, sans-serif; margin:0; padding:0;
          display:flex; justify-content:center;}
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto auto auto;
      grid-gap: 20px;
      max-width: 1600px;
      padding: 20px;
    }
    .cell {border:1px solid #ccc; padding:10px; text-align:center;}
    .full {grid-column: 1 / span 2;}
    table {width:100%; border-collapse:collapse;}
    th,td {padding:4px; border:1px solid #ddd;}
  </style>
  <meta http-equiv='refresh' content='900'>
</head>
<body>
  <p style="text-align:center; font-size:0.9em; margin:8px 0">
    Last refreshed (Pacific Time): {{ last_update }}
  </p>
  <div class='grid'>
    <div class='cell'>{{ policy_bar|safe }}</div>
    <div class='cell'><h3>Elo leaderboard</h3>{{ elo_table|safe }}</div>
    <div class='cell'>{{ uni_bar|safe }}</div>
    <div class='cell'>{{ preference_pie|safe }}</div>
    <div class='cell full'>{{ progress_hist|safe }}</div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE, **CACHE)

@app.route("/refresh")
def manual_refresh():
    refresh_cache()
    return redirect(url_for("index"))

if __name__ == "__main__":
    refresh_cache()
    app.run(host="0.0.0.0", port=8080, debug=False)
