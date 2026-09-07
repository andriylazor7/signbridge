import argparse
import json
from pathlib import Path


def load_class_list(path: Path) -> dict[int, str]:
    class_id_to_gloss = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id_str, gloss = line.split(None, 1)
            class_id_to_gloss[int(class_id_str)] = gloss.strip()
    return class_id_to_gloss


def main(nslt_path: Path, classes_path: Path, videos_dir: Path, out_path: Path):
    class_id_to_gloss = load_class_list(classes_path)

    with open(nslt_path, encoding="utf-8") as f:
        nslt = json.load(f)

    entries = []
    skipped_missing_video = 0

    for video_id, info in nslt.items():
        video_file = videos_dir / f"{video_id}.mp4"
        if not video_file.exists():
            skipped_missing_video += 1
            continue

        class_id, start_frame, end_frame = info["action"]
        entries.append({
            "video_id": video_id,
            "gloss": class_id_to_gloss[class_id],
            "split": info["subset"],
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Written {len(entries)} examples to {out_path}")
    print(f"Skipped (video missing on disk): {skipped_missing_video}")
    print(f"Classes (unique words) in result: {len({e['gloss'] for e in entries})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nslt", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.nslt, args.classes, args.videos_dir, args.out)