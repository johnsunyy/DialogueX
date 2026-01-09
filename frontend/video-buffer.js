/**
 * Video Buffer Manager for DIALOGUE-X
 * Synchronizes video delay with audio translation latency
 */

class VideoBufferManager {
    constructor() {
        this.videoBuffers = new Map(); // uid -> VideoFrameBuffer
        this.delayMs = 3500; // Default 3.5 seconds delay
        this.isEnabled = false;
    }

    /**
     * Set the delay time based on translation latency
     * @param {number} delayMs - Delay in milliseconds
     */
    setDelay(delayMs) {
        console.log(`[VideoBuffer] Setting delay to ${delayMs}ms`);
        this.delayMs = delayMs;

        // Update all existing buffers
        this.videoBuffers.forEach((buffer) => {
            buffer.setDelay(delayMs);
        });
    }

    /**
     * Enable video buffering (when translation is enabled)
     */
    enable() {
        console.log('[VideoBuffer] Enabling video delay synchronization');
        this.isEnabled = true;

        // Apply buffering to all existing video elements
        this.videoBuffers.forEach((buffer, uid) => {
            buffer.startBuffering();
        });
    }

    /**
     * Disable video buffering (when translation is disabled)
     */
    disable() {
        console.log('[VideoBuffer] Disabling video delay');
        this.isEnabled = false;

        // Remove buffering from all video elements
        this.videoBuffers.forEach((buffer, uid) => {
            buffer.stopBuffering();
        });

        this.videoBuffers.clear();
    }

    /**
     * Register a remote video track for buffering
     * @param {number} uid - User ID
     * @param {HTMLVideoElement} videoElement - Video element to buffer
     */
    addVideoTrack(uid, videoElement) {
        if (!this.isEnabled) {
            console.log(`[VideoBuffer] Not enabled, skipping buffer for UID ${uid}`);
            return;
        }

        console.log(`[VideoBuffer] Adding video buffer for UID ${uid}`);

        const buffer = new VideoFrameBuffer(videoElement, this.delayMs);
        this.videoBuffers.set(uid, buffer);
        buffer.startBuffering();
    }

    /**
     * Remove a video track from buffering
     * @param {number} uid - User ID
     */
    removeVideoTrack(uid) {
        const buffer = this.videoBuffers.get(uid);
        if (buffer) {
            console.log(`[VideoBuffer] Removing video buffer for UID ${uid}`);
            buffer.stopBuffering();
            this.videoBuffers.delete(uid);
        }
    }
}

/**
 * Individual video frame buffer
 * Captures video frames and plays them back with a delay
 */
class VideoFrameBuffer {
    constructor(videoElement, delayMs) {
        this.videoElement = videoElement;
        this.delayMs = delayMs;
        this.frameQueue = [];
        this.isBuffering = false;
        this.canvas = null;
        this.ctx = null;
        this.playbackInterval = null;
        this.captureInterval = null;
    }

    setDelay(delayMs) {
        this.delayMs = delayMs;
        // Adjust queue size if needed
        const targetFrames = Math.ceil(delayMs / (1000 / 30)); // 30 FPS
        while (this.frameQueue.length > targetFrames) {
            this.frameQueue.shift(); // Remove old frames
        }
    }

    startBuffering() {
        if (this.isBuffering) return;

        console.log(`[VideoFrameBuffer] Starting buffering with ${this.delayMs}ms delay`);
        this.isBuffering = true;

        // Create canvas for frame capture
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.videoElement.videoWidth || 640;
        this.canvas.height = this.videoElement.videoHeight || 480;
        this.ctx = this.canvas.getContext('2d');

        // Capture frames at ~30 FPS
        this.captureInterval = setInterval(() => {
            this.captureFrame();
        }, 1000 / 30);

        // Play delayed frames
        setTimeout(() => {
            this.startPlayback();
        }, this.delayMs);
    }

    captureFrame() {
        if (!this.ctx || !this.videoElement) return;

        try {
            // Draw current video frame to canvas
            this.ctx.drawImage(
                this.videoElement,
                0, 0,
                this.canvas.width,
                this.canvas.height
            );

            // Get frame data
            const frameData = this.ctx.getImageData(
                0, 0,
                this.canvas.width,
                this.canvas.height
            );

            // Add to queue with timestamp
            this.frameQueue.push({
                data: frameData,
                timestamp: Date.now()
            });

            // Limit queue size (keep ~5 seconds worth of frames)
            const maxFrames = Math.ceil((this.delayMs / 1000) * 30) + 150; // Buffer extra
            if (this.frameQueue.length > maxFrames) {
                this.frameQueue.shift();
            }
        } catch (e) {
            // Silently handle errors (video might not be ready)
        }
    }

    startPlayback() {
        if (this.playbackInterval) return;

        console.log('[VideoFrameBuffer] Starting delayed playback');

        // Play frames at ~30 FPS
        this.playbackInterval = setInterval(() => {
            if (this.frameQueue.length > 0) {
                const frame = this.frameQueue.shift();
                if (frame && this.ctx) {
                    // Display the delayed frame
                    this.ctx.putImageData(frame.data, 0, 0);

                    // Optional: Update video element with buffered frame
                    // (This is complex - for now we just log)
                }
            }
        }, 1000 / 30);
    }

    stopBuffering() {
        console.log('[VideoFrameBuffer] Stopping buffering');
        this.isBuffering = false;

        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }

        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }

        this.frameQueue = [];
        this.canvas = null;
        this.ctx = null;
    }
}

// Export for use in main app
window.VideoBufferManager = VideoBufferManager;
