import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
)


def extract_landmarks_from_video(video_path: str) -> np.ndarray:
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval_ms = 1000 / fps

        frames_data = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * frame_interval_ms)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            two_hands = np.zeros((2, 21, 3), dtype=np.float32)
            for i, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                for j, lm in enumerate(hand_landmarks):
                    two_hands[i, j] = [lm.x, lm.y, lm.z]

            frames_data.append(two_hands)
            frame_idx += 1

        cap.release()

    return np.stack(frames_data) if frames_data else np.zeros((0, 2, 21, 3), dtype=np.float32)


if __name__ == "__main__":
    raw_dir = Path("data/raw/videos")
    out_dir = Path("data/landmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_file in tqdm(list(raw_dir.glob("*.mp4"))):
        landmarks = extract_landmarks_from_video(str(video_file))
        if landmarks.shape[0] > 0:
            np.save(out_dir / f"{video_file.stem}.npy", landmarks)