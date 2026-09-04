import { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver, type HandLandmarkerResult, } from "@mediapipe/tasks-vision";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],           
  [0, 5], [5, 6], [6, 7], [7, 8],           
  [5, 9], [9, 10], [10, 11], [11, 12],      
  [9, 13], [13, 14], [14, 15], [15, 16],    
  [13, 17], [17, 18], [18, 19], [19, 20],   
  [0, 17],                                  
];

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const [status, setStatus] = useState("Завантаження моделі…");

  useEffect(() => {
    let animationFrameId: number;
    let stream: MediaStream;

    async function setup() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU", 
        },
        runningMode: "VIDEO",
        numHands: 2,
      });

      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStatus("Готово — покажи руку в кадр");
      predictLoop();
    }

    function predictLoop() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !canvas || !landmarker || video.readyState < 2) {
        animationFrameId = requestAnimationFrame(predictLoop);
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;

      const results = landmarker.detectForVideo(video, performance.now());

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLandmarks(ctx, results, canvas.width, canvas.height);

      animationFrameId = requestAnimationFrame(predictLoop);
    }

    setup();

    return () => {
      cancelAnimationFrame(animationFrameId);
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  function drawLandmarks(
    ctx: CanvasRenderingContext2D,
    results: HandLandmarkerResult,
    width: number,
    height: number
  ) {
    for (const hand of results.landmarks) {
      ctx.strokeStyle = "#00e5a0";
      ctx.lineWidth = 2;
      for (const [start, end] of HAND_CONNECTIONS) {
        const p1 = hand[start];
        const p2 = hand[end];
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
      ctx.fillStyle = "#ff3d81";
      for (const point of hand) {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  return (
    <div style={{ display: "grid", placeItems: "center", height: "100vh", background: "#111" }}>
      <div style={{ position: "relative", width: 640, height: 480 }}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ position: "absolute", width: 640, height: 480, transform: "scaleX(-1)" }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", width: 640, height: 480, transform: "scaleX(-1)" }}
        />
        <p style={{ position: "absolute", bottom: -32, color: "#fff", width: "100%", textAlign: "center" }}>
          {status}
        </p>
      </div>
    </div>
  );
}

export default App;