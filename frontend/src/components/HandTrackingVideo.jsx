import React, { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Define the connections between hand landmarks (based on MediaPipe Hands)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],   // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],   // Index finger
  [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
  [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky finger
];

// Helper function to draw lines connecting hand landmarks
const drawConnectors = (ctx, landmarks, connections, options) => {
  const { color = '#b885ca', lineWidth = 5 } = options;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  connections.forEach(([startIdx, endIdx]) => {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];

    ctx.beginPath();
    ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
    ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
    ctx.stroke();
  });
};

// Helper function to draw circles at hand landmarks
const drawLandmarks = (ctx, landmarks, options) => {
  const { color = '#b885ca', radius = 2 } = options;
  ctx.fillStyle = color;

  landmarks.forEach((landmark) => {
    const x = landmark.x * ctx.canvas.width;
    const y = landmark.y * ctx.canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  });
};

const HandTrackingWebcam = () => {
  const videoRef = useRef(null); // Reference for the webcam video
  const canvasRef = useRef(null); // Reference for the canvas overlay
  const [handLandmarker, setHandLandmarker] = useState(null); // Hand Landmarker instance

  useEffect(() => {
    const initializeHandLandmarker = async () => {
      // Load the vision files and initialize the hand landmarker
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
        numHands: 2, // Limit to detecting 1 hand
        runningMode: 'VIDEO', // Set running mode to video
        minHandDetectionConfidence: 0.7, // Increased confidence threshold
        minHandPresenceConfidence: 0.7,
        minTrackingConfidence: 0.7
      });
      setHandLandmarker(handLandmarker);
    };

    // Initialize the hand landmarker
    initializeHandLandmarker();
  }, []);

  useEffect(() => {
    // Access the webcam and stream video to the video element
    const setupWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } // Lower resolution for faster processing
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing webcam:', err);
      }
    };

    setupWebcam();
  }, []);

  useEffect(() => {
    const handleVideoLoaded = () => {
      if (videoRef.current && canvasRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
      }
    };

    if (videoRef.current) {
      videoRef.current.addEventListener('loadeddata', handleVideoLoaded);
    }

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('loadeddata', handleVideoLoaded);
      }
    };
  }, []);

  useEffect(() => {
    if (handLandmarker && videoRef.current && canvasRef.current) {
      let frameCount = 0; // To track frames and implement frame skipping

      const processVideo = async () => {
        if (videoRef.current.readyState === 4) {
          frameCount++;
          if (frameCount % 3 !== 0) { // Skip 2 out of 3 frames for faster performance
            requestAnimationFrame(processVideo);
            return;
          }

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');

          // Resize canvas to match the video size
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          // Detect hands from video
          const handLandmarkerResult = await handLandmarker.detectForVideo(video, Date.now());

          // Clear the canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Draw landmarks and connectors if hands are detected
          if (handLandmarkerResult && handLandmarkerResult.landmarks) {
            handLandmarkerResult.landmarks.forEach((hand) => {
              // Draw the hand connections
              drawConnectors(ctx, hand, HAND_CONNECTIONS, {
                color: '#b885ca',
                lineWidth: 5,
              });

              // Draw each landmark with a small circle
              drawLandmarks(ctx, hand, { color: '#b885ca', radius: 2 });
            });
          }

          frameCount = 0; // Reset frame count after processing a frame
        }

        requestAnimationFrame(processVideo);
      };

      requestAnimationFrame(processVideo); // Start processing the video
    }
  }, [handLandmarker]);

  return (
    <div className = "camera-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: '100%', height: 'auto', position: 'absolute', top: 0, left: 0, zIndex: 1 }}
      />
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: 'auto', position: 'absolute', top: 0, left: 0, zIndex: 2 }}
      />
    </div>
  );
};

export default HandTrackingWebcam;