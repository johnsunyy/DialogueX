/**
 * sign-detection.js
 * 
 * Handles reading from local video stream, extracting hand landmarks 
 * using MediaPipe, and sending them to the backend SocketIO server
 * for sign language recognition.
 */

window.initSignDetection = function(videoElement, socket, room, localUid) {
    console.log("Initializing Sign Detection with MediaPipe Hands");
    let isDetecting = true;
    let lastFrameTime = 0;
    
    // Create an offscreen canvas for frame capture
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");

    // Initialize MediaPipe Hands
    const hands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5
    });

    // Handle results from MediaPipe
    hands.onResults((results) => {
        if (!isDetecting) return;
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const hand = results.multiHandLandmarks[0];
            const landmarks = [];
            
            // Extract all 21 landmarks (x, y, z)
            for (let i = 0; i < hand.length; i++) {
                landmarks.push(hand[i].x, hand[i].y, hand[i].z);
            }
            
            // Send to backend via Socket.IO
            if (socket && socket.connected) {
                socket.emit('sign_landmarks', {
                    room: room,
                    uid: localUid,
                    landmarks: landmarks
                });
            }
        }
    });

    // 10 FPS Processing Loop to avoid overloading the CPU/GPU
    async function detectionLoop() {
        if (!isDetecting) return;
        
        const now = Date.now();
        // Run roughly every 100ms (10 FPS)
        if (now - lastFrameTime >= 100 && videoElement.readyState >= 2 && !videoElement.paused) {
            lastFrameTime = now;
            
            // Render video frame to canvas then send to MediaPipe
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            await hands.send({ image: canvas }).catch(err => {
                console.warn("MediaPipe Hands processing error:", err);
            });
        }
        
        // Schedule next frame
        requestAnimationFrame(detectionLoop);
    }
    
    requestAnimationFrame(detectionLoop);

    // Provide a cleanup function to the window space so `leaveCall` or disable can call it
    window.stopSignDetection = function() {
        console.log("Stopping Sign Detection");
        isDetecting = false;
        try {
            hands.close();
        } catch (e) {
            console.error("Error closing MediaPipe:", e);
        }
    };
};
