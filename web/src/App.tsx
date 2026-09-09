import { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver, type HandLandmarkerResult, } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-web";
import { GLOSSES } from "./glosses";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// Mirrors ml/data/normalize.py + ml/data/dataset.py so the live landmark
// stream is preprocessed exactly like the training data.
const MAX_SEQ_LEN = 64;
const WRIST_IDX = 0;
const MIDDLE_MCP_IDX = 9;
const INFER_EVERY_N_FRAMES = 6;
const MIN_FRAMES_FOR_INFERENCE = 8;

type HandFrame = number[][][]; // [2 hands][21 landmarks][x, y, z]

function buildHandFrame(results: HandLandmarkerResult): HandFrame {
  const frame: HandFrame = [
    Array.from({ length: 21 }, () => [0, 0, 0]),
    Array.from({ length: 21 }, () => [0, 0, 0]),
  ];
  for (let h = 0; h < Math.min(2, results.landmarks.length); h++) {
    const hand = results.landmarks[h];
    for (let p = 0; p < 21; p++) {
      frame[h][p] = [hand[p].x, hand[p].y, hand[p].z];
    }
  }
  return frame;
}

function normalizeAndFlatten(buffer: HandFrame[]): Float32Array {
  const data = new Float32Array(MAX_SEQ_LEN * 126);
  buffer.forEach((frame, t) => {
    for (let h = 0; h < 2; h++) {
      const wrist = frame[h][WRIST_IDX];
      const mcp = frame[h][MIDDLE_MCP_IDX];
      const tx = mcp[0] - wrist[0];
      const ty = mcp[1] - wrist[1];
      const tz = mcp[2] - wrist[2];
      const scale = Math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-6;
      for (let p = 0; p < 21; p++) {
        const pt = frame[h][p];
        const base = t * 126 + h * 63 + p * 3;
        data[base] = (pt[0] - wrist[0]) / scale;
        data[base + 1] = (pt[1] - wrist[1]) / scale;
        data[base + 2] = (pt[2] - wrist[2]) / scale;
      }
    }
  });
  return data;
}

function buildMask(len: number): Uint8Array {
  const mask = new Uint8Array(MAX_SEQ_LEN);
  for (let i = 0; i < Math.min(len, MAX_SEQ_LEN); i++) mask[i] = 1;
  return mask;
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exps = Float32Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps, (e) => e / sum);
}

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

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const bufferRef = useRef<HandFrame[]>([]);
  const frameCountRef = useRef(0);
  const isInferringRef = useRef(false);
  const [status, setStatus] = useState("Завантаження моделі…");

  useEffect(() => {
    let animationFrameId: number;
    let stream: MediaStream;

    async function runInference() {
      const session = sessionRef.current;
      if (!session || isInferringRef.current) return;
      isInferringRef.current = true;
      try {
        const buffer = bufferRef.current;
        const data = normalizeAndFlatten(buffer);
        const mask = buildMask(buffer.length);
        const feeds = {
          landmarks: new ort.Tensor("float32", data, [1, MAX_SEQ_LEN, 126]),
          mask: new ort.Tensor("bool", mask, [1, MAX_SEQ_LEN]),
        };
        const output = await session.run(feeds);
        const probs = softmax(output.logits.data as Float32Array);
        let bestIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestIdx]) bestIdx = i;
        }
        setStatus(`${GLOSSES[bestIdx]} — ${(probs[bestIdx] * 100).toFixed(0)}%`);
      } catch (err) {
        console.error("Inference failed:", err);
      } finally {
        isInferringRef.current = false;
      }
    }

    async function setup() {
      const [vision, session] = await Promise.all([
        FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        ),
        ort.InferenceSession.create("/models/transformer.onnx"),
      ]);
      sessionRef.current = session;

      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
      } catch (err) {
        console.error("Camera access failed:", err);
        setStatus("Немає доступу до камери");
        return;
      }
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

      const buffer = bufferRef.current;
      buffer.push(buildHandFrame(results));
      if (buffer.length > MAX_SEQ_LEN) buffer.shift();

      frameCountRef.current += 1;
      if (
        frameCountRef.current % INFER_EVERY_N_FRAMES === 0 &&
        buffer.length >= MIN_FRAMES_FOR_INFERENCE
      ) {
        void runInference();
      }

      animationFrameId = requestAnimationFrame(predictLoop);
    }

    setup();

    return () => {
      cancelAnimationFrame(animationFrameId);
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
      void sessionRef.current?.release();
    };
  }, []);

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