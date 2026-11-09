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

let currentFilename = null;
let currentDetections = [];


/* --------------------------------------------------
   Threshold control
-------------------------------------------------- */

thresholdSlider.addEventListener("input", () => {
    thresholdValue.textContent = Number(
        thresholdSlider.value
    ).toFixed(2);
});


/* --------------------------------------------------
   Review persistence
-------------------------------------------------- */

function getReviewKey(detectionId) {
    return `megalith-review:${currentFilename}:${detectionId}`;
}

function getReview(detectionId) {
    return localStorage.getItem(getReviewKey(detectionId));
}

function saveReview(detectionId, decision) {
    const key = getReviewKey(detectionId);

    if (getReview(detectionId) === decision) {
        localStorage.removeItem(key);
    } else {
        localStorage.setItem(key, decision);
    }

    renderCandidates(currentDetections);
}


/* --------------------------------------------------
   Render candidate cards
-------------------------------------------------- */

function renderCandidates(detections) {
    currentDetections = detections;
    candidateList.innerHTML = "";

    if (detections.length === 0) {
        candidateList.innerHTML = `
            <div class="empty-review">
                <h3>No candidates found</h3>
                <p>Lower the threshold or change the dataset to detect structures.</p>
            </div>
        `;

        reviewCount.textContent = "0";
        return;
    }

    let reviewed = 0;

    detections.forEach((detection) => {
        const decision = getReview(detection.id);

        if (decision) {
            reviewed += 1;
        }

        const card = document.createElement("div");
        card.className = "candidate-card";

        const topLine = document.createElement("div");
        topLine.className = "candidate-topline";

        const candidateId = document.createElement("span");
        candidateId.className = "candidate-id";
        candidateId.textContent =
            `CANDIDATE #${String(detection.id).padStart(2, "0")}`;

        const score = document.createElement("span");
        score.className = "candidate-score";
        score.textContent = `${detection.score_percent.toFixed(1)}%`;

        topLine.append(candidateId, score);

        const coordinates = document.createElement("div");
        coordinates.className = "candidate-coordinates";

        coordinates.textContent =
            `PL-1992 X: ${detection.map_x.toFixed(1)}\n` +
            `PL-1992 Y: ${detection.map_y.toFixed(1)}\n` +
            `WINDOW: ${detection.width} × ${detection.height} m`;

        coordinates.style.whiteSpace = "pre-line";

        const actions = document.createElement("div");
        actions.className = "candidate-actions";

        const acceptButton = document.createElement("button");
        acceptButton.className = "review-button accept";
        acceptButton.textContent = "✓ Accept";
        acceptButton.classList.toggle("is-selected", decision === "accepted");
        acceptButton.setAttribute("aria-pressed", String(decision === "accepted"));

        acceptButton.addEventListener("click", () => {
            saveReview(detection.id, "accepted");
        });

        const rejectButton = document.createElement("button");
        rejectButton.className = "review-button reject";
        rejectButton.textContent = "✕ Reject";
        rejectButton.classList.toggle("is-selected", decision === "rejected");
        rejectButton.setAttribute("aria-pressed", String(decision === "rejected"));

        rejectButton.addEventListener("click", () => {
            saveReview(detection.id, "rejected");
        });

        actions.append(acceptButton, rejectButton);

        const status = document.createElement("div");
        status.className = "review-status";

        if (decision === "accepted") {
            status.textContent = "STATUS: ACCEPTED";
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
   Update result metrics
-------------------------------------------------- */

function updateMetrics(result) {
    document.getElementById("metric-windows").textContent =
        result.windows_evaluated;

    document.getElementById("metric-raw").textContent =
        result.raw_hits;

    document.getElementById("metric-detections").textContent =
        result.detections_after_nms;

    document.getElementById("metric-device").textContent =
        result.device;

    document.getElementById("result-filename").textContent =
        result.source_file;
}


/* --------------------------------------------------
   Run inference
-------------------------------------------------- */

scanButton.addEventListener("click", async () => {
    const filename = demSelect.value;
    const threshold = Number(thresholdSlider.value);

    if (!filename) {
        scanMessage.textContent = "No DEM files are available.";
        scanMessage.className = "scan-message error";
        return;
    }

    // Toggle button state for the CSS glowing spinner
    scanButton.disabled = true;
    scanButtonLabel.textContent = "Scanning terrain...";

    scanMessage.textContent =
        "Calculating local relief and evaluating terrain windows. Please wait.";
    scanMessage.className = "scan-message";

    try {
        const response = await fetch("/api/scan", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                filename: filename,
                threshold: threshold
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Something went wrong.");
        }

        currentFilename = result.source_file;

        updateMetrics(result);

        resultImage.src =
            `/static/${result.image_path}?t=${Date.now()}`;

        // Swap out placeholder for actual image map
        resultImage.classList.remove("hidden");
        mapPlaceholder.classList.add("hidden");

        // Reveal the metrics bento box
        resultsSection.classList.remove("hidden");

        renderCandidates(result.detections);

        scanMessage.textContent =
            `Analysis complete. ${result.detections_after_nms} candidate(s) isolated.`;
        scanMessage.className = "scan-message success";

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        scanMessage.textContent = error.message;
        scanMessage.className = "scan-message error";
        console.error("Scan failed:", error);

    } finally {
        scanButton.disabled = false;
        scanButtonLabel.textContent = "Execute Scan";
    }
});