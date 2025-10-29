// public/app.js (FINAL for Laptop Server setup - Dual Video)

// Helper function to reload video streams with cache busting
function reloadVideoStreams() {
    const rawVideo = document.getElementById('rawVideo');
    const processedVideo = document.getElementById('processedVideo');
    const timestamp = new Date().getTime(); // Cache buster

    if (rawVideo) {
        rawVideo.src = `/raw_video_feed?ts=${timestamp}`; // Reload raw feed
        console.log("Reloading raw video feed...");
    }
    if (processedVideo) {
        processedVideo.src = `/processed_video_feed?ts=${timestamp}`; // Reload processed feed
        console.log("Reloading processed video feed...");
    }
}


// Run when the DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    const startStopBtn = document.getElementById('startStopBtn');

    // --- Sync initial state from server ---
    if (startStopBtn) {
        fetch('/api/state')
            .then(res => { if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`); return res.json(); })
            .then(data => {
                if (data.power) {
                    startStopBtn.textContent = 'Stop'; startStopBtn.classList.remove('start-btn','bg-green-500','hover:bg-green-600'); startStopBtn.classList.add('stop-btn','bg-red-500','hover:bg-red-600');
                } else {
                    startStopBtn.textContent = 'Start'; startStopBtn.classList.remove('stop-btn','bg-red-500','hover:bg-red-600'); startStopBtn.classList.add('start-btn','bg-green-500','hover:bg-green-600');
                }
                // --- Initial Load of Video Streams ---
                reloadVideoStreams(); // Load streams after checking state
            })
            .catch(err => {
                 console.error("Error fetching initial state:", err);
                 if(startStopBtn) { startStopBtn.textContent = 'State Error'; startStopBtn.disabled = true; startStopBtn.classList.add('bg-gray-400'); }
                 const statusElement = document.getElementById('status'); if (statusElement) statusElement.textContent = `Error: ${err.message}`;
            });
    } else {
        // If no start/stop button (e.g., other pages), still try loading streams if they exist
        reloadVideoStreams();
    }
});

// ------------------------------------
// Control Functions
// ------------------------------------
function sendCmd(cmd) {
    // ... (Keep the exact same sendCmd function as before) ...
    fetch(`/control/${cmd}`)
        .then(res => {
             if (!res.ok) { return res.json().then(errData => { throw new Error(errData.message || `Server error: ${res.status}`); }).catch(() => { throw new Error(`Server error: ${res.status} ${res.statusText}`); }); }
             return res.json();
        })
        .then(data => {
            const statusElement = document.getElementById('status');
            if (statusElement && data && data.message) { statusElement.textContent = `Boat status: ${data.message}`; }
            console.log(`Command '${cmd}' sent. Response: ${data ? data.message : 'N/A'}`);
        })
        .catch(e => {
            console.error(`Error sending command '${cmd}':`, e);
            const statusElement = document.getElementById('status'); if (statusElement) statusElement.textContent = `Error: ${e.message || 'Network Error!'}`;
        });
}

// Start/Stop Button Logic
function toggleStartStop() {
    const startStopBtn = document.getElementById('startStopBtn');
    if (!startStopBtn || startStopBtn.disabled) return;

    const isStarting = startStopBtn.textContent.trim().toUpperCase() === 'START';
    const cmd = isStarting ? 'start' : 'stop';
    sendCmd(cmd); // Send command to Laptop server

    // Optimistic UI update
    if (isStarting) {
        startStopBtn.textContent = 'Stop'; startStopBtn.classList.remove('start-btn','bg-green-500','hover:bg-green-600'); startStopBtn.classList.add('stop-btn','bg-red-500','hover:bg-red-600');
        // --- Reload streams on Start ---
        reloadVideoStreams();
    } else {
        startStopBtn.textContent = 'Start'; startStopBtn.classList.remove('stop-btn','bg-red-500','hover:bg-red-600'); startStopBtn.classList.add('start-btn','bg-green-500','hover:bg-green-600');
        // Optionally stop/blank streams on Stop, or just let them show "PAUSED" / "DISCONNECTED"
        // reloadVideoStreams(); // Or maybe set src to "" to stop them?
    }
}

// Attach functions globally
window.sendCmd = sendCmd;
window.toggleStartStop = toggleStartStop;