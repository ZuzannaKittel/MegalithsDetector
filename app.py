from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model.predict import scan_map


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "static" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# --------------------------------------------------
# Pages
# --------------------------------------------------

@app.route("/")
def index():
    dem_files = sorted(
        path.name
        for path in RAW_DIR.glob("*.asc")
    )

    return render_template(
        "index.html",
        dem_files=dem_files,
    )


# --------------------------------------------------
# API: run model inference
# --------------------------------------------------

@app.route("/api/scan", methods=["POST"])
def run_scan():
    data = request.get_json(silent=True) or {}

    filename = data.get("filename")
    threshold = data.get("threshold", 0.65)

    if not filename:
        return jsonify({"error": "Please select a DEM tile."}), 400

    # Only allow files that actually exist directly in data/raw.
    dem_path = RAW_DIR / Path(filename).name

    if dem_path.name != filename or not dem_path.is_file():
        return jsonify({"error": "Invalid DEM file selection."}), 400

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "Threshold must be a number."}), 400

    if not 0.0 <= threshold <= 1.0:
        return jsonify({"error": "Threshold must be between 0 and 1."}), 400

    try:
        result = scan_map(
            dem_path,
            threshold=threshold,
            output_dir=RESULTS_DIR,
        )

        return jsonify(result)

    except Exception as exc:
        app.logger.exception("Inference failed")
        return jsonify({
            "error": f"Inference failed: {str(exc)}"
        }), 500


# --------------------------------------------------
# Development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )