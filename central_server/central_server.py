"""
Central server for RoboArena distributed evaluation.

Changes in this revision
------------------------
• The hard-coded OPEN_SOURCE_POLICIES list is removed.
• `PolicyModel.is_in_use` is now treated as “open-source flag”.
• Leaderboard endpoint reads that flag from the DB.
"""
import contextlib
import datetime
import random
import threading
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from google.cloud import storage
from scipy.special import expit
import requests
import websockets.sync.client

from config import (
    BASE_CREDIT_CORL,
    BASE_CREDIT_DEFAULT,
    INFINITE_CREDIT_OWNER,
    UCB_TIMES_THRESHOLD,
    WEEKLY_INC_CORL,
    WEEKLY_INC_DEFAULT,
)
from database.connection import initialize_database_connection
from database.schema import (
    EpisodeModel,
    PolicyModel,
    SessionModel,
    UserModel,
)
from logger import logger

# --------------------------------------------------------------------------- #
# Globals / constants
# --------------------------------------------------------------------------- #
SERVER_VERSION = "1.1"
SESSION_TIMEOUT_HOURS = 0.5

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

# Leaderboard algorithm hyper-params
EXCLUDE = {"PI0", "PI0_FAST"}
HYBRID_NUM_T_BUCKETS = 100
EM_ITERS = 60
NUM_RANDOM_SEEDS = 100
SCALE = 15
SHIFT = 1500

LEADERBOARD_CACHE = {"timestamp": None, "board": []}
LEADERBOARD_LOCK = threading.Lock()
CACHE_TTL_SECS = 3600

POLICY_ANALYSIS_PATH = "/home/pranavatreya/real_eval/output/policy_analysis.json"

rng = np.random.default_rng(0)

# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
SessionLocal = None  # initialised in __main__


# --------------------------------------------------------------------------- #
# Utility: GCS client
# --------------------------------------------------------------------------- #
def get_gcs_client():
    return storage.Client()


# --------------------------------------------------------------------------- #
# Utility: websocket liveness
# --------------------------------------------------------------------------- #
def _ws_policy_alive(ip: str, port: int, timeout: float = 3.0) -> bool:
    try:
        conn = websockets.sync.client.connect(
            f"ws://{ip}:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
        )
        conn.close()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Cleanup stale sessions
# --------------------------------------------------------------------------- #
def cleanup_stale_sessions():
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale = (
            db.query(SessionModel)
            .filter(
                SessionModel.session_completion_timestamp.is_(None),
                SessionModel.session_creation_timestamp < cutoff,
            )
            .all()
        )
        for s in stale:
            s.session_completion_timestamp = datetime.datetime.utcnow()
            s.evaluation_notes = (s.evaluation_notes or "") + "\nTIMED_OUT"
        if stale:
            db.commit()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Credit bookkeeping helpers
# --------------------------------------------------------------------------- #
def _ensure_weekly_credit(user: UserModel) -> None:
    """
    Top-up evaluation credit with at-most-one-week rollover.

    Logic
    -----
    • If < 1 whole week has elapsed → nothing happens.
    • If ≥ 1 week has elapsed:
        – Keep *at most* one week’s worth of unused credit
          (anything beyond `inc` is considered expired).
        – Add exactly one new week’s increment `inc`.
        – Update `last_credit_update` to `now`.
    • Result: user.eval_credit ∈ [0, 2*inc].
    """
    now = datetime.datetime.utcnow()
    weeks_elapsed = int((now - user.last_credit_update).days / 7)
    if weeks_elapsed <= 0:
        return  # no full week has passed

    # weekly increment depends on CoRL participation
    inc = WEEKLY_INC_CORL if user.participating_corl else WEEKLY_INC_DEFAULT

    # Carry over at most one week's increment
    carry_over = min(user.eval_credit, inc)

    # New balance = carry_over (prev week) + one fresh week
    user.eval_credit = carry_over + inc
    user.last_credit_update = now


def _get_or_create_user(db, email: str) -> UserModel:
    user = db.query(UserModel).filter_by(email=email).first()
    if user:
        _ensure_weekly_credit(user)
        return user
    base = BASE_CREDIT_CORL if False else BASE_CREDIT_DEFAULT
    user = UserModel(
        email=email, participating_corl=False, eval_credit=base, last_credit_update=datetime.datetime.utcnow()
    )
    db.add(user)
    db.flush()
    return user


def _deduct_credit(db, email: str, amount: int = 1) -> None:
    if email == INFINITE_CREDIT_OWNER:
        return
    user = _get_or_create_user(db, email)
    user.eval_credit = max(0, user.eval_credit - amount)


def _reward_credit(db, email: str, amount: int = 1) -> None:
    user = _get_or_create_user(db, email)
    user.eval_credit += amount


# --------------------------------------------------------------------------- #
# Version check
# --------------------------------------------------------------------------- #
@app.route("/version_check", methods=["POST"])
def version_check():
    if (request.get_json() or {}).get("client_version") == SERVER_VERSION:
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "error", "message": "Version mismatch"}), 400


# --------------------------------------------------------------------------- #
# Policy-selection helpers
# --------------------------------------------------------------------------- #
def _ucb_weight(times: int) -> float:
    if times >= UCB_TIMES_THRESHOLD:
        return 1.0
    return np.sqrt(UCB_TIMES_THRESHOLD / (times + 1))


def _sample_policy_A(cands):
    weights = [_ucb_weight(p.times_in_ab_eval or 0) for p in cands]
    return random.choices(cands, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# GET /get_policies_to_compare
# ---------------------------------------------------------------------------
@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    cleanup_stale_sessions()

    evaluator_email = (
        request.args.get("evaluator_email") or request.args.get("evaluator_name")
    )
    if not evaluator_email:
        return jsonify({"error": "evaluator_email missing"}), 400

    eval_location = request.args.get("eval_location", "")
    robot_name = request.args.get("robot_name", "DROID")

    db = SessionLocal()
    try:
        # Ensure evaluator exists / credit up-to-date
        _get_or_create_user(db, evaluator_email)

        # ------------------------- alive policy pool -----------------------
        cand = (
            db.query(PolicyModel)
            .filter(PolicyModel.ip_address.isnot(None), PolicyModel.port.isnot(None))
            .all()
        )
        random.shuffle(cand)
        alive = [p for p in cand if _ws_policy_alive(p.ip_address, p.port)]
        if len(alive) < 2:
            return jsonify({"error": "Fewer than two alive policy servers."}), 400

        # ------------------------- pick UCB policy -------------------------
        elig_A = [
            p
            for p in alive
            if p.owner_name == INFINITE_CREDIT_OWNER
            or (_get_or_create_user(db, p.owner_name).eval_credit > 0)
        ]
        if not elig_A:
            return jsonify({"error": "No policy with available credit."}), 400

        ucb_policy = _sample_policy_A(elig_A)
        uniform_peer = random.choice([p for p in alive if p != ucb_policy])

        # ------------------------- decide visible labels -------------------
        if random.random() < 0.5:
            visible_A, visible_B = ucb_policy, uniform_peer
        else:
            visible_A, visible_B = uniform_peer, ucb_policy

        # ------------------------- create session --------------------------
        sess_uuid = uuid.uuid4()
        internal_note = f"UCB_POLICY={ucb_policy.unique_policy_name}"

        db.add(
            SessionModel(
                session_uuid=sess_uuid,
                evaluation_type="A/B",
                evaluation_location=eval_location,
                evaluator_name=evaluator_email,
                robot_name=robot_name,
                policyA_name=visible_A.unique_policy_name,
                policyB_name=visible_B.unique_policy_name,
                evaluation_notes=internal_note,  # preserves UCB hint
            )
        )
        db.commit()

        response = {
            "session_id": str(sess_uuid),
            "evaluation_type": "A/B",
            "policies": [
                {"label": "A", "ip": visible_A.ip_address, "port": visible_A.port},
                {"label": "B", "ip": visible_B.ip_address, "port": visible_B.port},
            ],
        }
        return jsonify(response), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# POST /upload_eval_data
# --------------------------------------------------------------------------- #
@app.route("/upload_eval_data", methods=["POST"])
def upload_eval_data():
    if not request.form:
        return jsonify({"error": "multipart form-data required"}), 400

    sess_id = request.form.get("session_id")
    if not sess_id:
        return jsonify({"error": "Missing session_id"}), 400

    policy_letter_raw = request.form.get("policy_letter", "")
    letter = policy_letter_raw.split(";", 1)[0].strip().upper()

    db = SessionLocal()
    try:
        sess = db.query(SessionModel).filter_by(session_uuid=sess_id).first()
        if not sess:
            return jsonify({"error": f"No session {sess_id}"}), 400

        # resolve policy name
        policy_name = (
            sess.policyA_name if letter == "A"
            else sess.policyB_name if letter == "B"
            else f"UNLABELED_{letter}"
        )

        # GCS uploads
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        def _upload_if_present(key: str, ext: str):
            f = request.files.get(key)
            if not f:
                return None
            gcs_path = (
                f"{BUCKET_PREFIX}/{sess_id}/{policy_name}_"
                f"{datetime.datetime.utcnow().isoformat()}_{key}.{ext}"
            )
            bucket.blob(gcs_path).upload_from_file(f)
            return gcs_path

        new_episode = EpisodeModel(
            session_id=sess.id,
            policy_name=policy_name,
            command=request.form.get("command", ""),
            binary_success=int(request.form.get("binary_success") or 0)
            if request.form.get("binary_success")
            else None,
            partial_success=float(request.form.get("partial_success") or 0.0)
            if request.form.get("partial_success")
            else None,
            duration=int(request.form.get("duration") or 0)
            if request.form.get("duration")
            else None,
            gcs_left_cam_path=_upload_if_present("video_left", "mp4"),
            gcs_right_cam_path=_upload_if_present("video_right", "mp4"),
            gcs_wrist_cam_path=_upload_if_present("video_wrist", "mp4"),
            npz_file_path=_upload_if_present("npz_file", "npz"),
            policy_ip=request.form.get("policy_ip"),
            policy_port=(
                int(request.form.get("policy_port"))
                if (request.form.get("policy_port") or "").isdigit()
                else None
            ),
            third_person_camera_type=request.form.get("third_person_camera_type"),
            third_person_camera_id=(
                int(request.form.get("third_person_camera_id"))
                if (request.form.get("third_person_camera_id") or "").isdigit()
                else None
            ),
            feedback=policy_letter_raw,
            timestamp=datetime.datetime.utcnow(),
        )
        db.add(new_episode)
        db.commit()
        return jsonify({"status": "success"}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /terminate_session
# ---------------------------------------------------------------------------
@app.route("/terminate_session", methods=["POST"])
def terminate_session():
    data = request.form if request.form else request.get_json()
    if not data:
        return jsonify({"error": "Missing form/JSON data"}), 400

    sess_id = data.get("session_id")
    new_notes = data.get("evaluation_notes", "")

    db = SessionLocal()
    try:
        sess = db.query(SessionModel).filter_by(session_uuid=sess_id).first()
        if not sess:
            return jsonify({"error": f"No session {sess_id}"}), 404

        # ------------------------------------------------------------------
        # Extract stored UCB_POLICY hint **before** modifying evaluation_notes
        # ------------------------------------------------------------------
        ucb_policy_name = None
        for line in (sess.evaluation_notes or "").splitlines():
            if line.startswith("UCB_POLICY="):
                ucb_policy_name = line.split("=", 1)[1].strip()
                break

        # ------------------------------------------------------------------
        # Append evaluator feedback (don’t overwrite internal hint)
        # ------------------------------------------------------------------
        combined_notes = (sess.evaluation_notes or "") + "\n" + new_notes
        sess.evaluation_notes = combined_notes
        sess.session_completion_timestamp = datetime.datetime.utcnow()

        # ------------------------------------------------------------------
        # If session is valid, adjust counts / credit
        # ------------------------------------------------------------------
        if "VALID_SESSION" in new_notes.upper():
            # increment counts for both visible policies
            for pname in (sess.policyA_name, sess.policyB_name):
                pol = db.query(PolicyModel).filter_by(unique_policy_name=pname).first()
                if pol:
                    pol.times_in_ab_eval = (pol.times_in_ab_eval or 0) + 1
                    pol.last_time_evaluated = datetime.datetime.utcnow()

            # debit credit from **ucb_policy’s** owner
            if ucb_policy_name:
                pol_ucb = (
                    db.query(PolicyModel)
                    .filter_by(unique_policy_name=ucb_policy_name)
                    .first()
                )
                if pol_ucb:
                    _deduct_credit(db, pol_ucb.owner_name, 1)

            # reward evaluator
            _reward_credit(db, sess.evaluator_name, 1)

        db.commit()
        return jsonify({"status": "terminated", "session_id": sess_id}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()
        

# ---------------------------------------------------------------------------
# 🆕 Canonical-university mapping + helper
# ---------------------------------------------------------------------------
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
    "UNIVERSITY OF TEXAS AT AUSTIN": "UT Austin",
}
TARGET_UNIS = [
    "Berkeley",
    "Stanford",
    "UW",
    "UPenn",
    "UMontreal",
    "Yonsei",
    "UT Austin",
]


def canonicalize_uni(raw: str | None) -> str:
    """Return canonical university name or 'Other'."""
    if not raw:
        return "Other"
    upper = raw.upper().strip()
    for k, canon in UNI_CANONICAL.items():
        if k in upper:
            return canon
    return raw # it might be a new institution not in the mapping

# ---------------------------------------------------------------------------
# 🆕 Endpoint: list completed/VALID A-B evaluation sessions
# ---------------------------------------------------------------------------
@app.route("/api/list_ab_evaluations", methods=["GET"])
def list_ab_evaluations():
    """
    Return _all_ completed and VALID A/B-evaluation sessions, **newest first**.
    Only the fields needed by the React UI are included.

    Response JSON schema
    --------------------
    {
        "evaluations": [
            {
                "session_id": str,
                "university": str,
                "completion_time": "2025-05-24T21:14:03.123456Z",
                "preference": "A" | "B" | "TIE" | null,
                "longform_feedback": str | null,
                "language_instruction": str | null,
                "policyA": {
                    "name": str,
                    "partial_success": float | null,
                    "wrist_video_url": str | null,
                    "third_person_video_url": str | null
                },
                "policyB": { …same keys as policyA… }
            },
            …
        ]
    }
    """
    db = SessionLocal()
    try:
        sessions = (
            db.query(SessionModel)
            .filter(
                SessionModel.evaluation_type == "A/B",
                SessionModel.session_completion_timestamp.isnot(None),
                SessionModel.evaluation_notes.isnot(None),
            )
            .order_by(SessionModel.session_completion_timestamp.desc())
            .all()
        )

        out = []
        for s in sessions:
            # Only keep sessions explicitly marked as VALID
            if "VALID_SESSION" not in s.evaluation_notes.upper():
                continue

            # ------------------------------------
            # Parse evaluation_notes
            # ------------------------------------
            pref = None
            feedback = None
            for line in (s.evaluation_notes or "").splitlines():
                line = line.strip()
                if line.upper().startswith("PREFERENCE="):
                    pref = line.split("=", 1)[1].strip().upper()
                elif line.upper().startswith("LONGFORM_FEEDBACK="):
                    feedback = line.split("=", 1)[1].strip()

            # ------------------------------------
            # Fetch episodes for policies A & B
            # ------------------------------------
            episodes = (
                db.query(EpisodeModel)
                .filter(
                    EpisodeModel.session_id == s.id,
                    EpisodeModel.policy_name.in_([s.policyA_name, s.policyB_name]),
                )
                .all()
            )
            ep_map = {ep.policy_name: ep for ep in episodes}
            if s.policyA_name not in ep_map or s.policyB_name not in ep_map:
                # Incomplete data – skip
                continue

            def _policy_block(policy_name: str) -> dict:
                ep = ep_map[policy_name]

                # Pick the preferred third-person camera video
                cam_type = (ep.third_person_camera_type or "").lower()
                if "left" in cam_type and ep.gcs_left_cam_path:
                    third_rel = ep.gcs_left_cam_path
                elif "right" in cam_type and ep.gcs_right_cam_path:
                    third_rel = ep.gcs_right_cam_path
                else:
                    # Fallback if camera_type missing / misspelled
                    third_rel = ep.gcs_left_cam_path or ep.gcs_right_cam_path

                def _url(rel_path: str | None) -> str | None:
                    if not rel_path:
                        return None
                    return f"https://storage.googleapis.com/{BUCKET_NAME}/{rel_path}"

                return {
                    "name": policy_name,
                    "partial_success": ep.partial_success,
                    "wrist_video_url": _url(ep.gcs_wrist_cam_path),
                    "third_person_video_url": _url(third_rel),
                }

            policyA_block = _policy_block(s.policyA_name)
            policyB_block = _policy_block(s.policyB_name)

            # Any episode will do for getting the language command
            lang_instr = ep_map[s.policyA_name].command or ep_map[s.policyB_name].command

            out.append(
                {
                    "session_id": str(s.session_uuid),
                    "university": canonicalize_uni(s.evaluation_location),
                    "completion_time": s.session_completion_timestamp.isoformat() + "Z",
                    "evaluator_name": s.evaluator_name,
                    "preference": pref,
                    "longform_feedback": feedback,
                    "language_instruction": lang_instr,
                    "policyA": policyA_block,
                    "policyB": policyB_block,
                }
            )

        return jsonify({"evaluations": out}), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

def em_hybrid(df,
              iters: int = EM_ITERS,
              step_clip: float = 1.0,
              l2_psi: float = 1e-2,
              l2_theta: float = 1e-2,
              step_decay: float = 0.99,
              tol: float = 1e-4,
              n_restarts: int = 1,
              use_partials: bool = False,
              sigma_partial: float = 0.3,
              partial_weight: float = 1.0): # 2.0 if you want to give partials more weight
    """
    EM for independent‐solve hybrid BT, with optional partial‐success signals.
    If use_partials=True, df must contain 'i_partial' and 'j_partial' in [0,1].
    """
    # ——— Precompute indices & masks ———
    pols   = pd.unique(pd.concat([df.i, df.j]))
    idmap  = {p: k for k, p in enumerate(pols)}
    P      = len(pols)
    i_idx  = df.i .map(idmap).to_numpy()
    j_idx  = df.j .map(idmap).to_numpy()
    y      = df.y .to_numpy()
    win    = (y == 2)
    loss   = (y == 0)
    tie    = (y == 1)

    if use_partials:
        s_i_par = df["i_partial"].to_numpy()
        s_j_par = df["j_partial"].to_numpy()

    best_ll, best_board = -np.inf, None

    for restart in range(n_restarts):
        rng.bit_generator.advance(restart * 1000)

        # ——— Initialize parameters ———
        θ = rng.normal(0., .1, P)
        τ = rng.normal(0., .1, HYBRID_NUM_T_BUCKETS)
        ψ = np.zeros((P, HYBRID_NUM_T_BUCKETS))
        π = np.full(HYBRID_NUM_T_BUCKETS, 1 / HYBRID_NUM_T_BUCKETS)
        ν = 0.5

        def clip_step(x, g, h, clip_val):
            if abs(h) < 1e-8:
                return x
            return x - np.clip(g/h, -clip_val, clip_val)

        # ——— EM loop ———
        for it in range(iters):
            curr_clip = step_clip * (step_decay ** it)

            # E-step: compute solve probabilities
            z_i     = θ[i_idx][:,None] + ψ[i_idx] - τ
            z_j     = θ[j_idx][:,None] + ψ[j_idx] - τ
            solve_i = expit(z_i)
            solve_j = expit(z_j)

            # A/B likelihoods
            p_win  = solve_i * (1 - solve_j)
            p_loss = (1 - solve_i) * solve_j
            p_tie  = 2 * ν * np.sqrt(p_win * p_loss)
            like_ab = (p_win*win[:,None]
                     + p_loss*loss[:,None]
                     + p_tie*tie[:,None])

            # optional partial‐success likelihood
            if use_partials:
                err_i  = (s_i_par[:,None] - solve_i)**2
                err_j  = (s_j_par[:,None] - solve_j)**2
                like_ps = np.exp(-(err_i + err_j)/(2*sigma_partial**2))**partial_weight
                like    = like_ab * like_ps
            else:
                like = like_ab

            # responsibilities γ[n,t]
            γ = π * np.clip(like, 1e-12, None)
            γ /= γ.sum(axis=1, keepdims=True)

            # M-step: update θ
            θ_prev = θ.copy()
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                g = h = 0.0

                for t in range(HYBRID_NUM_T_BUCKETS):
                    # i-slot
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = γ[mi, t]
                    w, l_, tt = win[mi], loss[mi], tie[mi]
                    g  += ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum()
                    h  -= ((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()

                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2

                    # j-slot
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = γ[mj, t]
                    wj, lj, tj = win[mj], loss[mj], tie[mj]
                    g  += ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h  -= ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum()

                    if use_partials:
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                # L2 on θ
                g -= l2_theta * θ[p]
                h -= l2_theta
                θ[p] = clip_step(θ[p], g, h, curr_clip)

            θ -= θ.mean()

            # M-step: update ψ
            for p in range(P):
                mi = (i_idx == p)
                mj = (j_idx == p)
                for t in range(HYBRID_NUM_T_BUCKETS):
                    si   = solve_i[mi, t]
                    sj_i = solve_j[mi, t]
                    gm   = γ[mi, t]
                    si_j = solve_i[mj, t]
                    sj_j = solve_j[mj, t]
                    gmj  = γ[mj, t]

                    # A/B
                    w, l_, tt   = win[mi], loss[mi], tie[mi]
                    wj, lj, tj  = win[mj], loss[mj], tie[mj]
                    g = ((w*(1-sj_i) - l_*sj_i + tt*(sj_i-si)) * gm).sum() \
                      + ((lj*(1-si_j) - wj*si_j + tj*(si_j-sj_j)) * gmj).sum()
                    h = -(((si*(1-si) + sj_i*(1-sj_i)) * gm).sum()
                         + ((si_j*(1-si_j) + sj_j*(1-sj_j)) * gmj).sum())

                    # partials
                    if use_partials:
                        g  += partial_weight * (((s_i_par[mi]-si)*si*(1-si)) * gm).sum() / sigma_partial**2
                        h  -= partial_weight * (((si*(1-si))**2) * gm).sum() / sigma_partial**2
                        g  += partial_weight * (((s_j_par[mj]-sj_j)*sj_j*(1-sj_j)) * gmj).sum() / sigma_partial**2
                        h  -= partial_weight * (((sj_j*(1-sj_j))**2) * gmj).sum() / sigma_partial**2

                    # L2 on ψ
                    g += l2_psi * ψ[p, t]
                    h -= l2_psi
                    ψ[p, t] = clip_step(ψ[p, t], g, h, curr_clip)

            ψ -= ψ.mean(axis=1, keepdims=True)

            # M-step: update τ
            for t in range(HYBRID_NUM_T_BUCKETS):
                si_t = solve_i[:, t]
                sj_t = solve_j[:, t]
                g    = (γ[:,t]*(si_t + sj_t - 1.0)).sum()
                h    = - (γ[:,t]*(si_t*(1-si_t) + sj_t*(1-sj_t))).sum()
                τ[t] = clip_step(τ[t], g, h, curr_clip)
            τ -= τ.mean()

            # update π, ν
            π = γ.mean(axis=0); π /= π.sum()
            ν = 0.5 * ((p_tie*γ).sum() / max((p_win*γ).sum(), 1e-9))

            if np.max(np.abs(θ - θ_prev)) < tol:
                break

        # finalize restart
        mixlik = (π * like).sum(axis=1)
        ll_cur = np.sum(np.log(mixlik + 1e-12))
        board  = pd.DataFrame({"policy": pols, "score": θ})\
                     .sort_values("score", ascending=False)\
                     .reset_index(drop=True)
        if ll_cur > best_ll:
            best_ll, best_board = ll_cur, board

    return best_board


def _recompute_leaderboard() -> list[dict]:
    """
    Build preference dataframe → run EM hybrid NUM_RANDOM_SEEDS times →
    return list of {policy, score(Elo), std, open_source}.
    """
    db = SessionLocal()
    try:
        # ---------- build preference dataframe ----------
        pairs, eps_lookup = [], {}

        eps_ab = (
            db.query(EpisodeModel)
              .filter(EpisodeModel.feedback.ilike("%; %"))
              .with_entities(
                  EpisodeModel.session_id,
                  EpisodeModel.feedback,
                  EpisodeModel.partial_success,
              )
              .all()
        )
        for sid, fb, ps in eps_ab:
            letter = (fb or "").strip().upper().split(";", 1)[0]
            if letter in ("A", "B"):
                eps_lookup[(sid, letter)] = ps

        sessions = (
            db.query(SessionModel)
              .filter(
                  SessionModel.evaluation_type == "A/B",
                  SessionModel.session_completion_timestamp.isnot(None),
                  SessionModel.evaluation_notes.isnot(None),
              )
              .all()
        )

        for s in sessions:
            if "VALID_SESSION" not in (s.evaluation_notes or "").upper():
                continue
            A, B = s.policyA_name.strip(), s.policyB_name.strip()
            if A.upper() in EXCLUDE or B.upper() in EXCLUDE:
                continue

            pref = None
            for line in (s.evaluation_notes or "").splitlines():
                t = line.strip().upper()
                if t.startswith("PREFERENCE="):
                    pref = {"A": 2, "B": 0, "TIE": 1}.get(
                        t.split("=", 1)[1], None
                    )
                    break
            if pref is None:
                continue

            sid = s.id
            i_par = eps_lookup.get((sid, "A"), np.nan)
            j_par = eps_lookup.get((sid, "B"), np.nan)
            pairs.append((A, B, pref, i_par, j_par))

        pref_df = pd.DataFrame(
            pairs, columns=["i", "j", "y", "i_partial", "j_partial"]
        )
        if pref_df.empty:
            return []

        # ---------- run EM hybrid NUM_RANDOM_SEEDS times ----------
        from collections import defaultdict
        score_runs = defaultdict(list)

        for seed in range(NUM_RANDOM_SEEDS):
            global rng
            rng = np.random.default_rng(seed)

            run = em_hybrid(
                pref_df,
                use_partials=False,
                n_restarts=1,
            )
            for _, row in run.iterrows():
                score_runs[row.policy].append(row.score)

        # ---------- aggregate, transform, tag ----------
        board = []
        for pol, scores in score_runs.items():
            raw_mean = float(np.mean(scores))
            raw_std  = float(np.std(scores, ddof=1))

            elo_mean = round(raw_mean * SCALE + SHIFT)
            elo_std  = round(raw_std  * SCALE, 1)

            # NEW: look up open-source flag in DB (is_in_use == True)
            pol_row = (
                db.query(PolicyModel)
                  .filter_by(unique_policy_name=pol)
                  .first()
            )
            is_open = bool(pol_row.is_in_use) if pol_row else False

            board.append({
                "policy": pol,
                "score":  elo_mean,
                "std":    elo_std,
                "open_source": is_open,
            })

        board.sort(key=lambda d: d["score"], reverse=True)
        return board

    finally:
        db.close()


def _refresh_loop():
    while True:
        try:
            board = _recompute_leaderboard()
            with LEADERBOARD_LOCK:
                LEADERBOARD_CACHE["board"] = board
                LEADERBOARD_CACHE["timestamp"] = datetime.datetime.utcnow()
            logger.info(f"Leaderboard cache refreshed ({len(board)} policies).")
        except Exception as e:
            logger.error(f"Leaderboard recompute failed: {e}")
        time.sleep(CACHE_TTL_SECS)

@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    with LEADERBOARD_LOCK:
        data = {
            "last_updated": (
                LEADERBOARD_CACHE["timestamp"].isoformat() + "Z"
                if LEADERBOARD_CACHE["timestamp"]
                else None
            ),
            "board": LEADERBOARD_CACHE["board"],
        }
    return jsonify(data), 200

@app.route("/api/policy_analysis.json", methods=["GET"])
def serve_policy_analysis():
    try:
        return send_file(POLICY_ANALYSIS_PATH, mimetype="application/json")
    except Exception as e:
        logger.error(f"Cannot serve policy_analysis.json: {e}")
        return jsonify({"error": "analysis report not available"}), 404

# --------------------------------------------------------------------------- #
# App bootstrap
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    db_url = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
    SessionLocal = initialize_database_connection(db_url)
    logger.info(f"DB connected → {db_url}")

    threading.Thread(target=_refresh_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)