import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lectrace import image, link, note, plot, system_text, text, url_reference, video

from _helpers import (
    FEATURES,
    TOKENS,
    attention_chart,
    attention_heatmap,
    attention_table,
)

ASSETS = Path(__file__).parent / "assets"

D_MODEL = 8
NUM_HEADS = 2
D_K = D_MODEL // NUM_HEADS


def main():
    torch.set_grad_enabled(False)  # @invisible

    text("# You are looking at a Python file")
    text(
        "Not a slide deck and not a notebook. The thing on your screen is "
        "`01_attention.py`, and the arrow keys are walking through it one line at a "
        "time. The panel on the right is the state of the program at the line you "
        "are on."
    )
    image(str(ASSETS / "pipeline.png"), width=840)
    text(
        "That is the whole of lectrace. It loads your script, runs `main()` under "
        "`sys.settrace`, records the value of every variable you asked for at every "
        "line, and writes the result to a JSON file that a static viewer reads. "
        "Nothing is hosted, nothing is uploaded, and anyone you share it with "
        "needs nothing installed to open the result."
    )
    text("Running on:")
    system_text([sys.executable, "--version"])

    note(
        "Pause here and press the left arrow twice so the class sees the program "
        "run backwards. That reversal is what gets a reaction, so show it before "
        "you explain it. Do not announce that stepping backwards is possible; just "
        "do it and let them notice."
    )

    text("## The example")
    text(
        "Six words of a clinical note, and a self-attention layer small enough that "
        "every number fits on the screen: `d_model=8`, `num_heads=2`, so `d_k=4`."
    )
    tokens = TOKENS  # @inspect tokens
    E = embed(tokens)  # @inspect E
    text(
        "Row *i* of `E` is the meaning of word *i*. We wrote these eight numbers per "
        "word by hand instead of learning them, so that the arithmetic stays "
        "readable: column 0 marks a negation word, column 1 a symptom, column 2 a "
        "drug, column 3 the subject, column 4 a linking word."
    )

    text("## Three questions about every word")
    text(
        "Attention asks each word to play three roles at once. What am I looking "
        "for (the query), what do I advertise about myself (the key), and what do I "
        "hand over if somebody attends to me (the value). Three projections of the "
        "same eight numbers:"
    )
    W_q, W_k, W_v = projections()  # @inspect W_q
    Q = E @ W_q  # @inspect Q
    K = E @ W_k  # @inspect K
    V = E @ W_v  # @inspect V

    text("## Split into heads")
    text(
        "One head can only look for one kind of thing, so the eight columns are cut "
        "into two groups of four. Head 0 takes columns 0 to 3, head 1 takes columns "
        "4 to 7. In PyTorch this is `.view(seq, heads, d_k).transpose(0, 1)`."
    )
    link(split_heads)
    Qh = split_heads(Q)  # @inspect Qh
    Kh = split_heads(K)
    Vh = split_heads(V)  # @clear E Q K V @inspect Vh
    text(
        "`E`, `Q`, `K` and `V` have just left the panel. They are still in memory, "
        "we are done reading them, and a panel with four dead matrices in it is a "
        "panel nobody reads."
    )

    text("## Score, scale, softmax")
    text(
        "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}"
        "{\\sqrt{d_k}}\\right)V$$"
    )
    scores = Qh @ Kh.transpose(-2, -1)  # @inspect scores
    text(
        "`scores[h, i, j]` is how much word *i* wants to hear from word *j* in head "
        "*h*. Two heads, six words, so two 6 by 6 grids."
    )
    scaled = scores / (D_K**0.5)  # @inspect scaled
    text(
        "Dot products grow with dimension. Without the $\\sqrt{d_k}$ the softmax "
        "below saturates and the gradients go to nothing, which is a lot of "
        "consequence for one division."
    )
    weights = F.softmax(scaled, dim=-1)  # @inspect weights
    assert torch.allclose(weights.sum(dim=-1), torch.ones(NUM_HEADS, len(tokens)))  # @invisible
    text("Every row now sums to exactly 1. Attention is a weighted average.")

    text("## Reading head 0")
    attention_table(weights, head=0)  # @stepover
    text(
        "Look at the row for **denies**. It puts 0.728 on *pain* and 0.219 on "
        "*chest*, and almost nothing anywhere else. This head went looking for the "
        "symptom that is being denied."
    )
    attention_chart(weights, head=0, query_index=1)  # @stepover
    text(
        "That distinction carries the clinical meaning. *Patient denies chest pain* "
        "and *patient reports chest pain* are the same six words. Only the attention "
        "pattern separates them."
    )
    text(
        "Now look at the row for **patient**: 0.167 six times, a flat average. Its "
        "query vector is all zeros, so every score is 0, and the softmax of equal "
        "scores is the uniform distribution. A head with no opinion about a word "
        "falls back to averaging. There is no special case in the code for that, it "
        "drops out of the arithmetic."
    )

    text("## Reading head 1")
    attention_table(weights, head=1)  # @stepover
    text(
        "Same layer, same forward pass, different job. Row **on** puts 0.916 on "
        "*aspirin*. That is the argument for having more than one head."
    )
    plot(attention_heatmap(weights))  # @stepover

    note(
        "Say plainly that we set these weights by hand. In a trained model nobody "
        "does, you get the heads you get, and reading them is how you find out what "
        "the model keyed on. Do not promise the class their heads will look this "
        "tidy."
    )

    text("## Collecting the values")
    context = weights @ Vh  # @inspect context
    text(
        "Each word is now the weighted average of the values it attended to. Row 1 "
        "of head 0 is no longer the negation vector, it is mostly *pain* with a bit "
        "of *chest*. The word has been rewritten in terms of what it points at."
    )
    merged = merge_heads(context)  # @inspect merged
    output = merged @ torch.eye(D_MODEL)  # @inspect output
    text(
        "Glue the heads back together and pass them through an output projection, "
        "which is what lets the two heads mix. We used the identity here so the "
        "numbers stay recognisable."
    )

    text("## The same treatment, for a convolution")
    text(
        "None of this is specific to attention. Here is a 3 by 3 kernel crossing a "
        "6 by 6 image, which is the same idea applied to a convolution."
    )
    video(str(ASSETS / "convolution.mp4"), width=900)
    text(
        "That clip was a `for` loop with two inspect comments on it. Whatever "
        "you are building, you can do this to it."
    )

    link(url_reference("https://arxiv.org/abs/1706.03762"))
    text("Next file: how to write one of these.")


def embed(tokens):
    rows = []
    for token in tokens:
        rows.append(FEATURES[token])
    return torch.tensor(rows)


def projections():
    """Set by hand so that two readable heads fall out of the arithmetic."""
    W_q = torch.zeros(D_MODEL, D_MODEL)
    W_q[0, 1] = 8.0
    W_q[4, 6] = 8.0

    W_k = torch.eye(D_MODEL)
    W_k[2, 6] = 1.0

    W_v = torch.eye(D_MODEL)
    return W_q, W_k, W_v


def split_heads(X):
    """[seq, d_model] to [heads, seq, d_k]."""
    seq_len = X.shape[0]
    reshaped = X.view(seq_len, NUM_HEADS, D_K)
    return reshaped.transpose(0, 1)


def merge_heads(X):
    """[heads, seq, d_k] back to [seq, d_model]."""
    seq_len = X.shape[1]
    swapped = X.transpose(0, 1).contiguous()
    return swapped.view(seq_len, D_MODEL)


if __name__ == "__main__":
    main()
