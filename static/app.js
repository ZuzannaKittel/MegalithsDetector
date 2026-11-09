const scanButton = document.getElementById("scan-button");
const scanButtonLabel = document.getElementById("scan-button-label");
const demSelect = document.getElementById("dem-select");
const thresholdSlider = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");

const scanMessage = document.getElementById("scan-message");
const resultsSection = document.getElementById("results-section");
const resultImage = document.getElementById("result-image");
const mapPlaceholder = document.getElementById("map-placeholder");
const candidateList = document.getElementById("candidate-list");
const reviewCount = document.getElementById("review-count");

const DEFAULT_THRESHOLD = 0.50;

let currentFilename = null;
let currentDetections = [];


/* --------------------------------------------------
   Threshold control
-------------------------------------------------- */

function updateThresholdLabel() {
    thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
}

thresholdSlider.value = String(DEFAULT_THRESHOLD);
updateThresholdLabel();
thresholdSlider.addEventListener("input", updateThresholdLabel);


/* --------------------------------------------------
   Review persistence

   Use source filename + original window coordinates, not the
   temporary candidate number assigned after NMS. The same map
   window therefore keeps its review when the threshold changes.
-------------------------------------------------- */

function getReviewKey(detection) {
    return `megalith-review:${encodeURIComponent(currentFilename)}:${detection.x}:${detection.y}`;
}

function getReview(detection) {
    return localStorage.getItem(getReviewKey(detection));
}

function saveReview(detection, decision) {
    const key = getReviewKey(detection);

    if (getReview(detection) === decision) {
        localStorage.removeItem(key);
    } else {
        localStorage.setItem(key, decision);
    }

    renderCandidates(currentDetections);
}


/* --------------------------------------------------
   Candidate cards
-------------------------------------------------- */

function renderCandidates(detections) {
    currentDetections = detections;
    candidateList.innerHTML = "";

    if (detections.length === 0) {
        candidateList.innerHTML = `
            <div class="empty-review">
                <h3>No candidates found</h3>
                <p>Try a lower threshold or inspect another elevation dataset.</p>
            </div>
        `;
        reviewCount.textContent = "0/0";
        return;
    }

    let reviewed = 0;

    detections.forEach((detection) => {
        const decision = getReview(detection);
        if (decision) reviewed += 1;

        const card = document.createElement("div");
        card.className = "candidate-card";

        const topLine = document.createElement("div");
        topLine.className = "candidate-topline";

        const candidateId = document.createElement("span");
        candidateId.className = "candidate-id";
        candidateId.textContent = `CANDIDATE #${String(detection.id).padStart(2, "0")}`;

        const score = document.createElement("span");
        score.className = "candidate-score";
        score.textContent = `MODEL SCORE ${Number(detection.score_percent).toFixed(1)}%`;
        topLine.append(candidateId, score);

        const coordinates = document.createElement("div");
        coordinates.className = "candidate-coordinates";
        coordinates.textContent =
            `WINDOW CENTRE — PL-1992\n` +
            `X: ${Number(detection.map_x).toFixed(1)}\n` +
            `Y: ${Number(detection.map_y).toFixed(1)}\n` +
            `WINDOW: ${Number(detection.width_m).toFixed(1)} × ` +
            `${Number(detection.height_m).toFixed(1)} m`;
        coordinates.style.whiteSpace = "pre-line";

        const actions = document.createElement("div");
        actions.className = "candidate-actions";

        const acceptButton = document.createElement("button");
        acceptButton.type = "button";
        acceptButton.className = "review-button accept";
        acceptButton.textContent = "✓ Accept";
        acceptButton.classList.toggle("is-selected", decision === "accepted");
        acceptButton.setAttribute("aria-pressed", String(decision === "accepted"));
        acceptButton.addEventListener("click", () => saveReview(detection, "accepted"));

        const rejectButton = document.createElement("button");
        rejectButton.type = "button";
        rejectButton.className = "review-button reject";
        rejectButton.textContent = "✕ Reject";
        rejectButton.classList.toggle("is-selected", decision === "rejected");
        rejectButton.setAttribute("aria-pressed", String(decision === "rejected"));
        rejectButton.addEventListener("click", () => saveReview(detection, "rejected"));

        actions.append(acceptButton, rejectButton);

        const status = document.createElement("div");
        status.className = "review-status";

        if (decision === "accepted") {
            status.textContent = "STATUS: ACCEPTED FOR REVIEW";
            status.classList.add("is-accepted");
        } else if (decision === "rejected") {
            status.textContent = "STATUS: REJECTED";
            status.classList.add("is-rejected");
        } else {
            status.textContent = "AWAITING HUMAN REVIEW";
        }

        card.append(topLine, coordinates, actions, status);
        candidateList.appendChild(card);
    });

    reviewCount.textContent = `${reviewed}/${detections.length}`;
}


/* --------------------------------------------------
   Result metrics
-------------------------------------------------- */

function updateMetrics(result) {
    document.getElementById("metric-windows").textContent = result.windows_evaluated;
    document.getElementById("metric-raw").textContent = result.raw_hits;
    document.getElementById("metric-detections").textContent = result.detections_after_nms;
    document.getElementById("metric-device").textContent = String(result.device).toUpperCase();
    document.getElementById("result-filename").textContent = result.source_file;
}


/* --------------------------------------------------
   Run inference
-------------------------------------------------- */

scanButton.addEventListener("click", async () => {
    const filename = demSelect.value;
    const threshold = Number(thresholdSlider.value);

    if (!filename) {
        scanMessage.textContent = "No DEM files are available. Add an .asc file to data/raw/.";
        scanMessage.className = "scan-message error";
        return;
    }

    scanButton.disabled = true;
    scanButtonLabel.textContent = "Scanning terrain...";
    scanMessage.textContent = "Calculating local relief and evaluating terrain windows...";
    scanMessage.className = "scan-message";

    try {
        const response = await fetch("/api/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, threshold })
        });

        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.error || `Scan request failed (${response.status}).`);
        }

        currentFilename = result.source_file;
        updateMetrics(result);

        resultImage.onload = () => {
            resultImage.classList.remove("hidden");
            mapPlaceholder.classList.add("hidden");
        };
        resultImage.onerror = () => {
            scanMessage.textContent = "The scan completed, but the annotated map image could not be loaded.";
            scanMessage.className = "scan-message error";
        };
        resultImage.alt = `Annotated terrain map for ${result.source_file}`;
        resultImage.src = `/static/${result.image_path}?t=${Date.now()}`;

        resultsSection.classList.remove("hidden");
        renderCandidates(result.detections || []);

        scanMessage.textContent =
            `Analysis complete. ${result.detections_after_nms} candidate(s) ready for human review.`;
        scanMessage.className = "scan-message success";

    } catch (error) {
        scanMessage.textContent = error.message;
        scanMessage.className = "scan-message error";
        console.error("Scan failed:", error);
    } finally {
        scanButton.disabled = false;
        scanButtonLabel.textContent = "Run detection";
    }
});
