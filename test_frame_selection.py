"""Regression: capture the slide AFTER the final element, even when that
completed state is on screen only briefly.

Slide builds ALPHA (6s) -> +BRAVO (6s) -> +CHARLIE (3s, shorter than
FRAME_MIN_STABLE_SECONDS) then hard-cuts away. The old code applied the
minimum-duration filter per build stage, so the short final stage was
discarded and the capture showed only "ALPHA BRAVO".
"""
import subprocess
import tempfile
from pathlib import Path

import yt2notion as y

FONT = "/System/Library/Fonts/Helvetica.ttc"


def clip(path, lines, bg, fg, dur):
    """Render `lines` stacked, each a big block so adding one is a real change."""
    draws = ",".join(
        f"drawtext=fontfile={FONT}:text='{t}':fontcolor={fg}:fontsize=54:"
        f"x=40:y={60 + n * 90}"
        for n, t in enumerate(lines))
    subprocess.run(
        [y.FFMPEG_PATH, "-y", "-f", "lavfi",
         "-i", f"color=c={bg}:size=640x360:duration={dur}:rate=2",
         "-vf", draws, "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        # Progressive build; the COMPLETED state lasts only 3s — shorter than
        # FRAME_MIN_STABLE_SECONDS, which the old code required per stage
        for i, (lines, dur) in enumerate([
            (["ALPHA"], 6),
            (["ALPHA", "BRAVO"], 6),
            (["ALPHA", "BRAVO", "CHARLIE"], 3),
        ]):
            p = tmp / f"a{i}.mp4"
            clip(p, lines, "white", "black", dur)
            parts.append(p)
        nxt = tmp / "b.mp4"
        clip(nxt, ["OTHER", "TOPIC"], "black", "white", 8)
        parts.append(nxt)

        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        vid = tmp / "v.mp4"
        subprocess.run([y.FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(vid)],
                       capture_output=True, check=True)

        keyframes = y._extract_keyframes(vid, tmp / "frames", ["en-US"])
        for path, ts, state, text in keyframes:
            print(f"  slide start={ts:.0f}s capture={state['capture']:.1f}s "
                  f"ocr={text!r}")

        slides = [{"start": ts, "image_path": p, "ocr_text": t, "state": st}
                  for p, ts, st, t in keyframes]
        merged = y._dedup_slides(slides)
        print(f"slides: {len(merged)}")

        built = merged[0]
        assert "CHARLIE" in built["ocr_text"], (
            f"captured before the final element landed: {built['ocr_text']!r}")
        assert "BRAVO" in built["ocr_text"] and "ALPHA" in built["ocr_text"]
        assert built["start"] < 3, (
            f"timestamp should be the build's start, got {built['start']}")
        assert any("OTHER" in s["ocr_text"] for s in merged), \
            "the following slide must stay separate (not over-merged)"
        assert "CHARLIE" not in merged[-1]["ocr_text"], \
            "the next slide must not be merged into the build"
        print("PASS captured the completed build, timestamped at its start")


if __name__ == "__main__":
    main()
