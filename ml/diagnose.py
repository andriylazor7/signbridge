import json
from collections import Counter
from pathlib import Path

import numpy as np

split = json.load(open("data/wlasl100_split.json", encoding="utf-8"))

# 1. Скільки прикладів на клас у train
by_class_train = Counter(e["gloss"] for e in split if e["split"] == "train")
counts = list(by_class_train.values())
print(f"Класів у train: {len(by_class_train)}")
print(f"Прикладів на клас (train): мін={min(counts)}, макс={max(counts)}, середнє={sum(counts)/len(counts):.1f}")

# 2. Чи виглядає розмітка осмислено - перші 10 пар video_id -> слово
print("\nПерші 10 записів (перевір хоча б 2-3 з них, відкривши відео вручну):")
for e in split[:10]:
    print(f"  data/raw/wlasl-processed/videos/{e['video_id']}.mp4  ->  {e['gloss']}")

# 3. Чи landmark-файли реально містять дані, а не нулі
print("\nЧастка ненульових значень у landmark-файлах (низьке число = MediaPipe не бачив руку):")
landmarks_dir = Path("data/landmarks")
for e in split[:8]:
    path = landmarks_dir / f"{e['video_id']}.npy"
    if not path.exists():
        print(f"  {e['video_id']}: ФАЙЛ ВІДСУТНІЙ")
        continue
    arr = np.load(path)
    nonzero_frac = np.count_nonzero(arr) / arr.size if arr.size else 0
    print(f"  {e['video_id']} ({e['gloss']}): shape={arr.shape}, ненульових={nonzero_frac:.1%}")