"""
Export a square-cropped, WhatsApp/LinkedIn-ready MP4 from the raw animation output.

Runs ffmpeg cropdetect to find the tightest non-black bounding box, then re-encodes
with that crop applied. Output is a standard MP4 (faststart, H.264) with no black borders.

Usage:
    python animate/export_square.py
    python animate/export_square.py --input animate/_output/warsaw_transit_full.mp4
    python animate/export_square.py --crf 28   # higher = smaller file, lower quality
"""

import subprocess
import re
import argparse
from pathlib import Path

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
DEFAULT_INPUT = Path(__file__).parent / "_output" / "warsaw_transit_full.mp4"
DEFAULT_CRF = 26


def detect_crop(input_path: Path) -> str:
    """Run cropdetect and return the most common crop string (e.g. '1200:1200:200:0')."""
    result = subprocess.run(
        [FFMPEG, "-i", str(input_path), "-vf", "cropdetect=24:2:0", "-f", "null", "-"],
        capture_output=True, text=True
    )
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
    if not crops:
        raise RuntimeError("cropdetect found no crop values — is the input path correct?")
    # Use the last detected value (stable at end of video)
    crop = crops[-1]
    print(f"Detected crop: {crop}")
    return crop


def export(input_path: Path, output_path: Path, crop: str, crf: int):
    """Apply crop and re-encode to a clean standard MP4."""
    cmd = [
        FFMPEG,
        "-i", str(input_path),
        "-vf", f"crop={crop}",
        "-vcodec", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", str(output_path),
    ]
    print(f"Encoding to {output_path} ...")
    subprocess.run(cmd, check=True)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Done: {output_path.name} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and re-encode animation to square MP4")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--crf",    type=int,  default=DEFAULT_CRF,
                        help="libx264 CRF (18=lossless-ish, 26=default, 32=smaller)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    output = args.input.parent / (args.input.stem + "_1x1.mp4")
    crop   = detect_crop(args.input)
    export(args.input, output, crop, args.crf)
