import os
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt
from matplotlib import animation

ASSETS = Path(__file__).resolve().parent.parent / "assets"

PATCH = np.array(
    [
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        [0.1, 0.9, 0.9, 0.1, 0.9, 0.9],
        [0.1, 0.9, 0.9, 0.1, 0.9, 0.9],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
    ]
)

KERNEL = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])


def build_pipeline_figure():
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Three boxes of BOX_W with GAP between them, centred in the 0..10 axes.
    # The arrows live in the gaps, so both values are named rather than
    # spelled out per box: nudging one no longer strands the arrows.
    BOX_W, GAP = 2.6, 0.7
    left = (10 - (3 * BOX_W + 2 * GAP)) / 2

    stages = [
        ("lecture.py", "your script,\nunchanged"),
        ("sys.settrace", "one step per line,\nvariables captured"),
        ("trace.json", "static site,\nno server"),
    ]
    for i, (title, subtitle) in enumerate(stages):
        x = left + i * (BOX_W + GAP)
        ax.add_patch(
            plt.Rectangle((x, 0.7), BOX_W, 1.5, facecolor="#eef2f7", edgecolor="#33415c", linewidth=1.4)
        )
        ax.text(x + BOX_W / 2, 1.72, title, ha="center", fontsize=13, family="monospace")
        ax.text(x + BOX_W / 2, 1.12, subtitle, ha="center", fontsize=9, color="#33415c")

    for i in range(len(stages) - 1):
        start = left + (i + 1) * BOX_W + i * GAP
        ax.annotate(
            "",
            xy=(start + GAP - 0.12, 1.45),
            xytext=(start + 0.12, 1.45),
            arrowprops={"arrowstyle": "-|>", "color": "#33415c"},
        )

    ax.text(5.0, 0.25, "python lecture.py still runs it as an ordinary script", ha="center", fontsize=9, style="italic", color="#5c6b82")
    fig.tight_layout()
    fig.savefig(ASSETS / "pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_convolution_clip():
    size, k = PATCH.shape[0], KERNEL.shape[0]
    out_size = size - k + 1
    output = np.full((out_size, out_size), np.nan)

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.6))
    positions = [(r, c) for r in range(out_size) for c in range(out_size)]

    def draw(frame):
        row, col = positions[frame]
        window = PATCH[row : row + k, col : col + k]
        output[row, col] = float((window * KERNEL).sum())

        left.clear()
        left.imshow(PATCH, cmap="gray", vmin=0, vmax=1)
        left.add_patch(
            plt.Rectangle((col - 0.5, row - 0.5), k, k, fill=False, edgecolor="#e63946", linewidth=2.5)
        )
        left.set_title("input", fontsize=10)
        left.set_xticks([])
        left.set_yticks([])

        right.clear()
        right.imshow(np.nan_to_num(output), cmap="magma", vmin=-4, vmax=4)
        right.set_title("feature map", fontsize=10)
        right.set_xticks([])
        right.set_yticks([])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(positions), interval=380, blit=False)
    # dpi=200 doubles the 7.5x3.6in figure to 1500x720, so the clip still looks
    # sharp at the width the lecture displays it. The bitrate has to rise with
    # the pixels or the extra resolution just buys compression mush.
    anim.save(ASSETS / "convolution.mp4", writer=animation.FFMpegWriter(fps=2.6, bitrate=1800), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    build_pipeline_figure()
    build_convolution_clip()
    for path in sorted(ASSETS.iterdir()):
        print(path.name, f"{path.stat().st_size / 1024:.0f} KB")
