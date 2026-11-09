from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model.predict import DEFAULT_THRESHOLD, scan_map


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.route("/")
def index():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dem_files = sorted(path.name for path in RAW_DIR.glob("*.asc"))
    return render_template("index.html", dem_files=dem_files)


@app.route("/api/scan", methods=["POST"])
def run_scan():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    filename = data.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        return jsonify({"error": "Please select a DEM tile."}), 400

    # Only accept a single filename directly inside data/raw/.
    if Path(filename).name != filename or not filename.lower().endswith(".asc"):
        return jsonify({"error": "Invalid DEM file selection."}), 400

    dem_path = RAW_DIR / filename
    if not dem_path.is_file():
        return jsonify({"error": "The selected DEM file was not found in data/raw/."}), 404

    try:
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
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
    except FileNotFoundError:
        app.logger.exception("Required file missing during inference")
        return jsonify({
            "error": "The model checkpoint is missing. Train the model first."
        }), 503
    except Exception:
        app.logger.exception("Inference failed")
        return jsonify({
            "error": "The scan failed. Check the Flask terminal for details."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
