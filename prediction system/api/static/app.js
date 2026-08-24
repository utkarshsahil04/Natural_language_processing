/**
 * PriorityPulse - Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let presets = [];
    let history = JSON.parse(localStorage.getItem("priority_history") || "[]");

    // Elements
    const form = document.getElementById("prediction-form");
    const subjectInput = document.getElementById("ticket-subject");
    const bodyInput = document.getElementById("ticket-body");
    const subjectCount = document.getElementById("subject-count");
    const bodyCount = document.getElementById("body-count");
    const clearBtn = document.getElementById("clear-btn");
    const predictBtn = document.getElementById("predict-btn");
    const btnSpinner = document.getElementById("btn-spinner");
    const presetsContainer = document.getElementById("presets-container");

    const emptyState = document.getElementById("empty-state");
    const resultDisplay = document.getElementById("result-display");
    const priorityHero = document.getElementById("priority-hero");
    const badgeText = document.getElementById("badge-text");
    const badgeIcon = document.getElementById("badge-icon");
    const latencyTag = document.getElementById("latency-tag");
    const latencyVal = document.getElementById("latency-val");
    const circleFill = document.getElementById("circle-fill");
    const circlePercentage = document.getElementById("circle-percentage");
    const formattedInputText = document.getElementById("formatted-input-text");
    const copyInputBtn = document.getElementById("copy-input-btn");

    const pctHigh = document.getElementById("pct-high");
    const barHigh = document.getElementById("bar-high");
    const pctMedium = document.getElementById("pct-medium");
    const barMedium = document.getElementById("bar-medium");
    const pctLow = document.getElementById("pct-low");
    const barLow = document.getElementById("bar-low");

    const gpuName = document.getElementById("gpu-name");
    const historyCount = document.getElementById("history-count");
    const historyTbody = document.getElementById("history-tbody");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const runBatchBtn = document.getElementById("run-batch-btn");
    const batchTbody = document.getElementById("batch-tbody");

    // 1. Initialize API and load GPU info & presets
    async function init() {
        renderHistory();
        try {
            const [infoRes, samplesRes] = await Promise.all([
                fetch("/api/info"),
                fetch("/api/samples"),
            ]);

            if (infoRes.ok) {
                const info = await infoRes.json();
                gpuName.textContent = info.device_name || "CPU";
            }

            if (samplesRes.ok) {
                presets = await samplesRes.json();
                renderPresets();
            }
        } catch (err) {
            console.warn("Could not connect to API:", err);
            gpuName.textContent = "Backend Offline";
        }
    }

    // 2. Render preset buttons
    function renderPresets() {
        presetsContainer.innerHTML = "";
        presets.forEach((sample, idx) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = `preset-chip ${idx === 0 ? "active" : ""}`;
            btn.innerHTML = `
                <span class="preset-title">${escapeHtml(sample.subject)}</span>
                <span class="preset-sub">${sample.type} • ${sample.category || "Sample"}</span>
            `;
            btn.addEventListener("click", () => {
                document.querySelectorAll(".preset-chip").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                loadPreset(sample);
            });
            presetsContainer.appendChild(btn);
        });

        // Preload first preset
        if (presets.length > 0) {
            loadPreset(presets[0]);
        }
    }

    function loadPreset(sample) {
        // Set type radio
        const radio = document.querySelector(`input[name="ticket_type"][value="${sample.type}"]`);
        if (radio) radio.checked = true;

        subjectInput.value = sample.subject;
        bodyInput.value = sample.body;
        updateCharCounts();
    }

    // 3. Form input & char counters
    function updateCharCounts() {
        subjectCount.textContent = `${subjectInput.value.length}/100`;
        bodyCount.textContent = `${bodyInput.value.length}/1000`;
    }

    subjectInput.addEventListener("input", updateCharCounts);
    bodyInput.addEventListener("input", updateCharCounts);

    clearBtn.addEventListener("click", () => {
        form.reset();
        updateCharCounts();
        document.querySelectorAll(".preset-chip").forEach(b => b.classList.remove("active"));
        emptyState.style.display = "flex";
        resultDisplay.style.display = "none";
        latencyTag.style.display = "none";
    });

    // 4. Handle prediction request
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const type = document.querySelector('input[name="ticket_type"]:checked')?.value || "Incident";
        const subject = subjectInput.value.trim();
        const body = bodyInput.value.trim();

        if (!subject && !body) return;

        setLoading(true);

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ type, subject, body }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Prediction failed");
            }

            const data = await res.json();
            displayResult(data, type, subject);
            addToHistory(data, type, subject);
        } catch (err) {
            alert("Prediction Error: " + err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        predictBtn.disabled = isLoading;
        btnSpinner.style.display = isLoading ? "inline-block" : "none";
    }

    // 5. Render prediction result
    function displayResult(data, type, subject) {
        emptyState.style.display = "none";
        resultDisplay.style.display = "flex";

        const priority = data.predicted_priority.toLowerCase();
        const confPct = Math.round(data.confidence * 100);

        // Reset theme classes
        priorityHero.className = `priority-hero theme-${priority}`;

        // Badge Text & Icon
        badgeText.textContent = priority.toUpperCase();
        if (priority === "high") {
            badgeIcon.textContent = "??";
        } else if (priority === "medium") {
            badgeIcon.textContent = "??";
        } else {
            badgeIcon.textContent = "?";
        }

        // Circular Chart
        circleFill.setAttribute("stroke-dasharray", `${confPct}, 100`);
        circlePercentage.textContent = `${confPct}%`;

        // Breakdown bars
        const lowVal = (data.probabilities.low * 100).toFixed(1);
        const medVal = (data.probabilities.medium * 100).toFixed(1);
        const highVal = (data.probabilities.high * 100).toFixed(1);

        pctHigh.textContent = `${highVal}%`;
        barHigh.style.width = `${highVal}%`;

        pctMedium.textContent = `${medVal}%`;
        barMedium.style.width = `${medVal}%`;

        pctLow.textContent = `${lowVal}%`;
        barLow.style.width = `${lowVal}%`;

        // Latency
        latencyVal.textContent = `${data.latency_ms} ms (${data.device || 'GPU'})`;
        latencyTag.style.display = "inline-flex";

        // Formatted code
        formattedInputText.textContent = data.text;
    }

    // Copy formatted input
    copyInputBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(formattedInputText.textContent);
        copyInputBtn.textContent = "Copied!";
        setTimeout(() => { copyInputBtn.textContent = "Copy"; }, 1500);
    });

    // 6. Prediction History
    function addToHistory(data, type, subject) {
        const item = {
            time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
            type: type,
            subject: subject || "Untitled Ticket",
            priority: data.predicted_priority,
            confidence: Math.round(data.confidence * 100),
            latency: data.latency_ms,
            text: data.text,
        };

        history.unshift(item);
        if (history.length > 50) history.pop();
        localStorage.setItem("priority_history", JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        historyCount.textContent = history.length;
        if (history.length === 0) {
            historyTbody.innerHTML = `
                <tr class="no-history-row">
                    <td colspan="7" class="text-center">No prediction history yet. Run a prediction above!</td>
                </tr>
            `;
            return;
        }

        historyTbody.innerHTML = history.map((item, idx) => `
            <tr>
                <td style="font-family: var(--font-mono); font-size: 0.75rem;">${item.time}</td>
                <td><span class="chip-label" style="display:inline-block; padding: 0.2rem 0.5rem; font-size: 0.75rem;">${item.type}</span></td>
                <td><strong>${escapeHtml(item.subject)}</strong></td>
                <td><span class="tag-badge tag-${item.priority.toLowerCase()}">${item.priority}</span></td>
                <td><span style="font-family: var(--font-mono);">${item.confidence}%</span></td>
                <td style="font-family: var(--font-mono); color: var(--text-muted);">${item.latency}ms</td>
                <td>
                    <button class="ghost-btn" style="padding: 0.25rem 0.5rem;" onclick="reloadHistoryItem(${idx})">Load</button>
                </td>
            </tr>
        `).join("");
    }

    window.reloadHistoryItem = function(idx) {
        const item = history[idx];
        if (!item) return;
        const radio = document.querySelector(`input[name="ticket_type"][value="${item.type}"]`);
        if (radio) radio.checked = true;
        subjectInput.value = item.subject;
        bodyInput.value = "";
        updateCharCounts();
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    clearHistoryBtn.addEventListener("click", () => {
        history = [];
        localStorage.removeItem("priority_history");
        renderHistory();
    });

    // 7. Tabs logic
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId)?.classList.add("active");
        });
    });

    // 8. Batch Benchmark execution
    runBatchBtn.addEventListener("click", async () => {
        if (presets.length === 0) return;
        runBatchBtn.disabled = true;
        runBatchBtn.innerHTML = `<span>Benchmarking...</span>`;
        batchTbody.innerHTML = `<tr><td colspan="7" class="text-center">Running GPU batch benchmark across 6 scenarios...</td></tr>`;

        try {
            const batchPayload = {
                tickets: presets.map(p => ({ type: p.type, subject: p.subject, body: p.body }))
            };

            const res = await fetch("/api/predict/batch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(batchPayload),
            });

            const data = await res.json();
            renderBatchResults(data.results);
        } catch (err) {
            batchTbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color: var(--priority-high)">Batch execution error: ${err.message}</td></tr>`;
        } finally {
            runBatchBtn.disabled = false;
            runBatchBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Re-Run Batch Test
            `;
        }
    });

    function renderBatchResults(results) {
        let correctCount = 0;
        const rowsHtml = results.map((r, idx) => {
            const preset = presets[idx] || {};
            const isMatch = r.predicted_priority.toLowerCase() === (preset.expected || "").toLowerCase();
            if (isMatch) correctCount++;

            return `
                <tr>
                    <td style="color: var(--text-muted);">${preset.category || "Scenario " + (idx+1)}</td>
                    <td><span class="chip-label" style="display:inline-block; padding: 0.15rem 0.4rem; font-size: 0.75rem;">${preset.type || ""}</span></td>
                    <td><strong>${escapeHtml(preset.subject || "")}</strong></td>
                    <td><span class="tag-badge tag-${r.predicted_priority.toLowerCase()}">${r.predicted_priority}</span></td>
                    <td style="text-transform: uppercase; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">${preset.expected || "N/A"}</td>
                    <td style="font-family: var(--font-mono); font-weight: 600;">${(r.confidence * 100).toFixed(1)}%</td>
                    <td>
                        <span class="tag-badge ${isMatch ? 'tag-match' : 'tag-mismatch'}">
                            ${isMatch ? '? MATCH' : '? MISMATCH'}
                        </span>
                    </td>
                </tr>
            `;
        }).join("");

        const accuracyPct = ((correctCount / results.length) * 100).toFixed(0);
        batchTbody.innerHTML = `
            <tr style="background: rgba(99, 102, 241, 0.08);">
                <td colspan="7" style="padding: 0.85rem 1rem; font-weight: 600;">
                    Benchmark Summary: <strong style="color: #a5b4fc;">${correctCount}/${results.length} Scenarios Matched (${accuracyPct}%)</strong>
                </td>
            </tr>
            ${rowsHtml}
        `;
    }

    // Keyboard Shortcut (Ctrl + Enter or Cmd + Enter)
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            form.dispatchEvent(new Event("submit"));
        }
    });

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Run initialization
    init();
});
