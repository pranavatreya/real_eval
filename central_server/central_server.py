import uuid
import datetime
import random
import requests  # Keep for any future expansions, though we removed health checks
from flask import Flask, request, jsonify
from flask import send_file
from google.cloud import storage

from database.schema import PolicyModel, SessionModel, EpisodeModel
from database.connection import initialize_database_connection
from logger import logger

import threading, time
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.special import expit

# Make rng
rng = np.random.default_rng(0)

# Ranking algorithm hyperparameters
EXCLUDE = {"PI0", "PI0_FAST"}
HYBRID_NUM_T_BUCKETS = 100
EM_ITERS = 60
NUM_RANDOM_SEEDS = 100

# Ranking cache
LEADERBOARD_CACHE = {"timestamp": None, "board": []}
LEADERBOARD_LOCK = threading.Lock()
CACHE_TTL_SECS = 3600   # 1 h

# For displaying the AI generated analysis report
POLICY_ANALYSIS_PATH = "/home/pranavatreya/real_eval/output/policy_analysis.json"

#  Flask App Setup
app = Flask(__name__)

BUCKET_NAME = "distributed_robot_eval"
BUCKET_PREFIX = "evaluation_data"

def get_gcs_client():
    return storage.Client()

# -- Utility / Cleanup: handle stale sessions --

SESSION_TIMEOUT_HOURS = 0.5

def cleanup_stale_sessions():
    db = SessionLocal()
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=SESSION_TIMEOUT_HOURS)
    try:
        stale_sessions = db.query(SessionModel).filter(
            SessionModel.session_completion_timestamp.is_(None),
            SessionModel.session_creation_timestamp < cutoff
        ).all()

        for sess in stale_sessions:
            sess.session_completion_timestamp = datetime.datetime.utcnow()
            if sess.evaluation_notes:
                sess.evaluation_notes += "\nTIMED_OUT"
            else:
                sess.evaluation_notes = "TIMED_OUT"

        if stale_sessions:
            db.commit()
    finally:
        db.close()


# -------------------------------
# Version-check endpoint
# -------------------------------
SERVER_VERSION = "1.1"  # Make sure to bump this up whenever you update the server/client code

@app.route("/version_check", methods=["POST"])
def version_check():
    """
    The client will send e.g. JSON: {"client_version": "1.0"}
    We compare with SERVER_VERSION. If mismatch, return an error.
    """
    data = request.get_json() or {}
    client_version = data.get("client_version", None)
    if client_version == SERVER_VERSION:
        return jsonify({"status": "ok", "message": "Versions match"}), 200
    else:
        return jsonify({"status": "error", "message": "Version mismatch"}), 400
# -------------------------------


@app.route("/get_policies_to_compare", methods=["GET"])
def get_policies_to_compare():
    """
    (1) Now, instead of returning only 4 policies, we return *all* policies
    that have non-null IP/port.
    (2) We still create a new session with policyA_name / policyB_name set to the first two,
        if at least 2 exist. If fewer than 2 exist, we still error.
    """

    # Clean up stale sessions
    cleanup_stale_sessions()

    evaluation_location = request.args.get("eval_location", "")
    evaluator_name = request.args.get("evaluator_name", "")
    robot_name = request.args.get("robot_name", "DROID")

    db = SessionLocal()
    try:
        # Grab all candidate policies that have an IP/port
        candidates = db.query(PolicyModel).filter(
            PolicyModel.ip_address.isnot(None),
            PolicyModel.port.isnot(None),
        ).all()

        random.shuffle(candidates)

        if len(candidates) < 2:
            return jsonify({"error": "We need at least 2 policies to do an A/B session."}), 400

        # Increment times_in_ab_eval for each
        for pol in candidates:
            if pol.times_in_ab_eval is None:
                pol.times_in_ab_eval = 0
            pol.times_in_ab_eval += 1

        session_uuid_ = uuid.uuid4()
        # policyA_name is the first, policyB_name is the second
        # We'll do that by default
        policyA_name = candidates[0].unique_policy_name
        policyB_name = candidates[1].unique_policy_name

        new_session = SessionModel(
            session_uuid=session_uuid_,
            evaluation_type="A/B",
            evaluation_location=evaluation_location,
            evaluator_name=evaluator_name,
            robot_name=robot_name,
            policyA_name=policyA_name,
            policyB_name=policyB_name
        )
        db.add(new_session)
        db.commit()

        # Return *all* policies
        resp_data = {
            "session_id": str(session_uuid_),
            "evaluation_type": "A/B",
            "policies": []
        }

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, pol in enumerate(candidates):
            label = labels[i] if i < len(labels) else f"X{i}"
            resp_data["policies"].append({
                "label": label,
                "policy_name": pol.unique_policy_name,
                "ip": pol.ip_address,
                "port": pol.port
            })

        return jsonify(resp_data), 200

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/upload_eval_data", methods=["POST"])
def upload_eval_data():
    if not request.form:
        return jsonify({"error": "Must send data in multipart form-data."}), 400

    session_id = request.form.get("session_id", None)
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    policy_name = request.form.get("policy_name", "")
    command = request.form.get("command", "")
    binary_success_str = request.form.get("binary_success", None)
    partial_success_str = request.form.get("partial_success", None)
    duration_str = request.form.get("duration", None)
    policy_ip = request.form.get("policy_ip", None)
    policy_port_str = request.form.get("policy_port", None)
    camera_type = request.form.get("third_person_camera_type", None)
    camera_id_str = request.form.get("third_person_camera_id", None)
    # We'll store policy_letter plus average latency inside the same text
    # E.g. "B;avg_latency=0.123"
    policy_letter_and_latency = request.form.get("policy_letter", "")  
    timestamp_str = request.form.get("timestamp", "")

    try:
        binary_success = int(binary_success_str) if binary_success_str is not None else None
    except:
        binary_success = None

    try:
        partial_success = float(partial_success_str) if partial_success_str else None
    except:
        partial_success = None

    try:
        duration = int(duration_str) if duration_str else None
    except:
        duration = None

    try:
        policy_port = int(policy_port_str) if policy_port_str else None
    except:
        policy_port = None

    try:
        third_person_camera_id = int(camera_id_str) if camera_id_str else None
    except:
        third_person_camera_id = None

    try:
        if timestamp_str:
            timestamp_dt = datetime.datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
        else:
            timestamp_dt = datetime.datetime.utcnow()
    except:
        timestamp_dt = datetime.datetime.utcnow()

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {session_id}"}), 400

        storage_client = get_gcs_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        def upload_file_if_present(file_key: str, extension: str):
            f = request.files.get(file_key, None)
            if not f:
                return None
            gcs_path = f"{BUCKET_PREFIX}/{session_id}/{policy_name}_{timestamp_str}_{file_key}.{extension}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_file(f)
            return gcs_path

        gcs_left_cam_path = upload_file_if_present("video_left", "mp4")
        gcs_right_cam_path = upload_file_if_present("video_right", "mp4")
        gcs_wrist_cam_path = upload_file_if_present("video_wrist", "mp4")
        npz_file_path = upload_file_if_present("npz_file", "npz")

        new_episode = EpisodeModel(
            session_id=session_obj.id,
            policy_name=policy_name,
            command=command,
            binary_success=binary_success,
            partial_success=partial_success,
            duration=duration,
            gcs_left_cam_path=gcs_left_cam_path,
            gcs_right_cam_path=gcs_right_cam_path,
            gcs_wrist_cam_path=gcs_wrist_cam_path,
            npz_file_path=npz_file_path,
            timestamp=timestamp_dt,
            policy_ip=policy_ip,
            policy_port=policy_port,
            third_person_camera_type=camera_type,
            third_person_camera_id=third_person_camera_id,
            feedback=policy_letter_and_latency  # Store the combined policy letter + average latency
        )
        db.add(new_episode)
        db.commit()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/terminate_session", methods=["POST"])
def terminate_session():
    form_data = request.form if request.form else request.json
    if not form_data:
        return jsonify({"error": "Missing form or JSON data."}), 400

    form_session_id = form_data.get("session_id")
    if not form_session_id:
        return jsonify({"error": "Missing session_id"}), 400

    final_notes = form_data.get("evaluation_notes", "")

    db = SessionLocal()
    try:
        session_obj = db.query(SessionModel).filter_by(session_uuid=form_session_id).first()
        if not session_obj:
            return jsonify({"error": f"No session with ID {form_session_id}"}), 404

        session_obj.session_completion_timestamp = datetime.datetime.utcnow()
        session_obj.evaluation_notes = final_notes

        used_policy_names = []
        if session_obj.policyA_name:
            used_policy_names.append(session_obj.policyA_name)
        if session_obj.policyB_name:
            used_policy_names.append(session_obj.policyB_name)

        for pname in used_policy_names:
            pol = db.query(PolicyModel).filter_by(unique_policy_name=pname).first()
            if pol:
                pol.last_time_evaluated = datetime.datetime.utcnow()

        db.commit()
        return jsonify({
            "status": "terminated",
            "session_id": form_session_id
        }), 200

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
    return "Other"


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

def _recompute_leaderboard():
    """
    Build preference dataframe → run EM hybrid NUM_RANDOM_SEEDS times →
    return list[ {policy, score(<mean>), std} ] sorted by score desc.
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

        # ---------- run EM hybrid multiple seeds ----------
        from collections import defaultdict

        score_runs = defaultdict(list)

        for seed in range(NUM_RANDOM_SEEDS):
            # reset global rng used inside em_hybrid
            global rng
            rng = np.random.default_rng(seed)

            board_run = em_hybrid(
                pref_df,
                use_partials=False,
                n_restarts=1,   # one restart per seed; we are averaging
            )
            for _, row in board_run.iterrows():
                score_runs[row.policy].append(row.score)

        # ---------- mean & std ----------
        board_avg = [
            {
                "policy": pol,
                "score": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
            }
            for pol, scores in score_runs.items()
        ]
        board_avg.sort(key=lambda d: d["score"], reverse=True)

        # round for transmission
        for d in board_avg:
            d["score"] = round(d["score"], 3)
            d["std"] = round(d["std"], 3)

        return board_avg

    finally:
        db.close()

def _refresh_loop():
    while True:
        try:
            board = _recompute_leaderboard()
            with LEADERBOARD_LOCK:
                LEADERBOARD_CACHE["board"] = board
                LEADERBOARD_CACHE["timestamp"] = datetime.datetime.utcnow()
            logger.info("Leaderboard cache refreshed; {} policies".format(len(board)))
        except Exception as e:
            logger.error(f"Leaderboard recompute failed: {e}")
        time.sleep(CACHE_TTL_SECS)

@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    with LEADERBOARD_LOCK:
        data = {
            "last_updated": (LEADERBOARD_CACHE["timestamp"].isoformat() + "Z")
                            if LEADERBOARD_CACHE["timestamp"] else None,
            "board": LEADERBOARD_CACHE["board"],
        }
    return jsonify(data), 200

@app.route("/api/policy_analysis.json", methods=["GET"])
def serve_policy_analysis():
    """
    Serve the cached AI-generated policy analysis report.
    """
    try:
        return send_file(POLICY_ANALYSIS_PATH, mimetype="application/json")
    except Exception as e:
        logger.error(f"Cannot serve policy_analysis.json: {e}")
        return jsonify({"error": "analysis report not available"}), 404

if __name__ == "__main__":
    # 1) Initialize the database connection
    database_url: str = "postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval"
    SessionLocal = initialize_database_connection(database_url)
    logger.info(f"Database connection to {database_url} initialized.")

    # Start the leaderboard computing thread
    threading.Thread(target=_refresh_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=True)
