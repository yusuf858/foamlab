"""
FoamLab — Flask Application
Integrated Detection and Analysis System for Foam/Oil Pollution in Lake Surface Images
"""
import os
import base64
import logging
import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template, Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("models", exist_ok=True)

_last_result = {}


def _read_image(field="image"):
    if field in request.files:
        data = request.files[field].read()
    else:
        b64 = request.json.get("image_b64", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    global _last_result
    try:
        img = _read_image("image")
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400

        filter_type    = request.form.get("filter_type", "gaussian")
        enhance_method = request.form.get("enhance", "clahe")
        seg_method     = request.form.get("segment", "otsu")
        se_shape       = request.form.get("se_shape", "ellipse")
        se_size        = int(request.form.get("se_size", 9))
        location       = request.form.get("location", "")

        from utils.pipeline import run_pipeline
        result = run_pipeline(img, filter_type, enhance_method,
                               seg_method, se_shape, se_size, location)
        _last_result = result

        try:
            from utils.database import log_analysis
            fname = request.files["image"].filename if "image" in request.files else ""
            log_analysis(fname, result, location)
        except Exception as e:
            logger.warning(f"DB log failed: {e}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"/api/analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_analyze", methods=["POST"])
def batch_analyze():
    from utils.pipeline import run_pipeline
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    filter_type    = request.form.get("filter_type", "gaussian")
    enhance_method = request.form.get("enhance", "clahe")
    seg_method     = request.form.get("segment", "otsu")
    se_shape       = request.form.get("se_shape", "ellipse")
    se_size        = int(request.form.get("se_size", 9))

    results = []
    total_ok = 0
    total_fail = 0

    for f in files:
        try:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                total_fail += 1
                results.append({"filename": f.filename, "error": "decode failed"})
                continue
            r = run_pipeline(img, filter_type, enhance_method,
                              seg_method, se_shape, se_size)
            results.append({
                "filename":     f.filename,
                "severity":     r["severity"],
                "dominant":     r["dominant_label"],
                "pollution_pct":r["pollution_area_pct"],
                "total_regions":r["total_regions"],
                "process_time": r["process_time"],
                "preview":      r["images"]["visualised"],
            })
            total_ok += 1
        except Exception as e:
            total_fail += 1
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({
        "results": results,
        "summary": {
            "total_processed": total_ok,
            "total_failed": total_fail,
            "total_images": len(files),
        }
    })


@app.route("/api/webcam_analyze", methods=["POST"])
def webcam_analyze():
    global _last_result
    try:
        data = request.get_json(force=True)
        b64  = data.get("image_b64", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400
        from utils.pipeline import run_pipeline
        result = run_pipeline(img)
        _last_result = result
        return jsonify(result)
    except Exception as e:
        logger.error(f"/api/webcam_analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/combination_study", methods=["POST"])
def combination_study():
    try:
        img = _read_image("image")
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400
        from utils.pipeline import run_combination_study
        results = run_combination_study(img)
        return jsonify({"combos": results})
    except Exception as e:
        logger.error(f"/api/combination_study: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/robustness", methods=["POST"])
def robustness():
    try:
        img = _read_image("image")
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400
        from utils.pipeline import run_robustness_study
        results = run_robustness_study(img)
        return jsonify({"variants": results})
    except Exception as e:
        logger.error(f"/api/robustness: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparative", methods=["POST"])
def comparative():
    try:
        img = _read_image("image")
        if img is None:
            return jsonify({"error": "Cannot decode image"}), 400
        from utils.pipeline import run_comparative_analysis
        results = run_comparative_analysis(img)
        return jsonify({"systems": results})
    except Exception as e:
        logger.error(f"/api/comparative: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    try:
        from utils.database import get_history
        return jsonify({"rows": get_history(200)})
    except Exception:
        return jsonify({"rows": []})


@app.route("/api/trend_data", methods=["GET"])
def trend_data():
    try:
        from utils.database import get_trend_data
        return jsonify({"data": get_trend_data()})
    except Exception:
        return jsonify({"data": []})


@app.route("/api/generate_pdf", methods=["GET"])
def generate_pdf():
    global _last_result
    try:
        from utils.report import generate_pdf_report
        if not _last_result:
            return jsonify({"error": "No analysis available"}), 400
        pdf = generate_pdf_report(_last_result)
        return Response(pdf, mimetype="application/pdf",
                        headers={"Content-Disposition":
                                 "attachment; filename=foamlab_report.pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/db_status", methods=["GET"])
def db_status():
    try:
        import utils.database as db_mod
        ok = db_mod.check_db_status()
        return jsonify({"connected": ok, "error": db_mod.get_last_error() if not ok else ""})
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)})


@app.route("/api/db_config", methods=["POST"])
def db_config_endpoint():
    """Allow UI to update DB credentials at runtime and test connection."""
    try:
        import utils.database as db_mod
        data = request.get_json(force=True)
        host     = data.get("host", "localhost")
        user     = data.get("user", "root")
        password = data.get("pass", "")
        name     = data.get("name", "foamlab_db")

        # Update module-level config at runtime
        db_mod.DB_CONFIG["host"]     = host
        db_mod.DB_CONFIG["user"]     = user
        db_mod.DB_CONFIG["password"] = password
        db_mod.DB_NAME               = name

        # Try to init (creates DB + tables if needed)
        ok = db_mod.init_db()
        if ok:
            return jsonify({"connected": True})
        else:
            try:
                import pymysql
                c = pymysql.connect(host=host, user=user, password=password,
                                    connect_timeout=5, charset="utf8mb4")
                c.close()
                return jsonify({"connected": False,
                                "error": f"Server OK but failed to create '{name}'. Check user permissions."})
            except Exception as e2:
                return jsonify({"connected": False, "error": str(e2)})
    except Exception as e:
        logger.error(f"/api/db_config: {e}", exc_info=True)
        return jsonify({"connected": False, "error": str(e)})


@app.route("/api/db_test", methods=["GET"])
def db_test():
    """Return detailed DB connection info for debugging."""
    try:
        import pymysql
        import utils.database as db_mod
        cfg = dict(db_mod.DB_CONFIG)
        result = {"host": cfg["host"], "user": cfg["user"],
                  "db_name": db_mod.DB_NAME, "status": "unknown", "error": None}
        try:
            # Try without database first
            c = pymysql.connect(
                host=cfg["host"], port=cfg["port"],
                user=cfg["user"], password=cfg["password"],
                connect_timeout=5, charset="utf8mb4"
            )
            c.close()
            result["server_reachable"] = True
        except Exception as e:
            result["server_reachable"] = False
            result["error"] = str(e)
            result["status"] = "Cannot reach MySQL server"
            return jsonify(result)
        # Try with database
        try:
            c = pymysql.connect(
                host=cfg["host"], port=cfg["port"],
                user=cfg["user"], password=cfg["password"],
                database=db_mod.DB_NAME,
                connect_timeout=5, charset="utf8mb4"
            )
            c.close()
            result["status"] = "connected"
            result["connected"] = True
        except Exception as e:
            result["status"] = "server_ok_db_missing"
            result["error"] = str(e)
            result["connected"] = False
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "connected": False})


if __name__ == "__main__":
    try:
        from utils.database import init_db
        init_db()
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")
    app.run(debug=True, host="0.0.0.0", port=5000)
