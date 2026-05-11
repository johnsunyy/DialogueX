/**
 * sign-detection.js
 *
 * Handles reading from the local video stream, extracting full-body landmarks
 * using MediaPipe Holistic, and sending a strict 225-float vector to the
 * backend SocketIO server for dual-model sign language recognition.
 *
 * Feature layout (matches backend/sign_translation.py and the trained model):
 *   [0   : 63 ]  Left-hand  landmarks  (21 keypoints × xyz)
 *   [63  : 126]  Right-hand landmarks  (21 keypoints × xyz)
 *   [126 : 225]  Pose       landmarks  (33 keypoints × xyz)
 *
 * Missing landmarks are filled with exactly 0.0 so the vector is always 225
 * floats regardless of occlusion.
 *
 * Preprocessing matches training pipeline (sign_model/scripts/01_extract_landmarks.py):
 *   - x and y coordinates are clipped to [0.0, 1.0]
 *   - z (depth) is left as-is
 */

window.initSignDetection = function(videoElement, socket, room, localUid) {
    console.log("[SignDetection] Initialising MediaPipe Holistic (225-feature mode)");
    let isDetecting  = true;
    let lastFrameTime = 0;

    // Offscreen canvas for frame capture
    const canvas = document.createElement("canvas");
    canvas.width  = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");

    // ── Initialise MediaPipe Holistic ─────────────────────────────────────────
    const holistic = new window.Holistic({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
        modelComplexity:        0,     // 0 = fastest for live use
        smoothLandmarks:        true,
        enableSegmentation:     false,
        refineFaceLandmarks:    false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence:  0.5
    });

    // ── Result handler ────────────────────────────────────────────────────────
    holistic.onResults((results) => {
        if (!isDetecting) return;

        // Helper: extract (x, y, z) for every landmark in a landmark list,
        // or return an array of `count * 3` zeros if the list is missing.
        // x and y are clipped to [0.0, 1.0] to match training-time preprocessing
        // (sign_model/scripts/01_extract_landmarks.py lines 194-195).
        function extractLandmarks(landmarkList, count) {
            const out = new Array(count * 3).fill(0.0);
            if (!landmarkList) return out;
            for (let i = 0; i < Math.min(landmarkList.length, count); i++) {
                const lm = landmarkList[i];
                // Clip x and y to [0, 1]; leave z as-is (relative depth)
                out[i * 3]     = Math.min(1.0, Math.max(0.0, lm.x || 0.0));
                out[i * 3 + 1] = Math.min(1.0, Math.max(0.0, lm.y || 0.0));
                out[i * 3 + 2] = lm.z || 0.0;
            }
            return out;
        }

        // Build the 225-float feature vector
        const leftHand  = extractLandmarks(results.leftHandLandmarks,  21); // 63 floats
        const rightHand = extractLandmarks(results.rightHandLandmarks, 21); // 63 floats
        const pose      = extractLandmarks(results.poseLandmarks,      33); // 99 floats

        const landmarks = leftHand.concat(rightHand, pose);  // 225 floats total

        // Send every result that arrives — the 10 FPS cap on holistic.send()
        // below already limits the rate. Adding a second gate here would further
        // starve the sliding-window buffer and hurt temporal accuracy.
        const hasData = landmarks.some(v => v !== 0.0);
        if (hasData && socket && socket.connected) {
            socket.emit('sign_landmarks', {
                room:      room,
                uid:       localUid,
                landmarks: landmarks          // 225-float array
            });
        }
    });

    // ── 10 FPS processing loop ────────────────────────────────────────────────
    // The 10 FPS cap here limits MediaPipe CPU cost.
    // Every frame that MediaPipe processes is forwarded immediately (no second gate).
    async function detectionLoop() {
        if (!isDetecting) return;

        const now = Date.now();
        // ~10 FPS cap to keep CPU/GPU load manageable
        if (now - lastFrameTime >= 100 &&
            videoElement.readyState >= 2 &&
            !videoElement.paused) {
            lastFrameTime = now;
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            await holistic.send({ image: canvas }).catch(err => {
                console.warn("[SignDetection] Holistic processing error:", err);
            });
        }

        requestAnimationFrame(detectionLoop);
    }

    requestAnimationFrame(detectionLoop);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    window.stopSignDetection = function() {
        console.log("[SignDetection] Stopping MediaPipe Holistic");
        isDetecting = false;
        try {
            holistic.close();
        } catch (e) {
            console.error("[SignDetection] Error closing Holistic:", e);
        }
    };
};
