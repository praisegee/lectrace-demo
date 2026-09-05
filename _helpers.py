import matplotlib.pyplot as plt
from lectrace import plot, table

TOKENS = ["patient", "denies", "chest", "pain", "on", "aspirin"]

# Eight dimensions, four of which we have given a meaning:
# 0 negation, 1 symptom, 2 drug, 3 subject, 4 linking word.
FEATURES = {
    "patient": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "denies": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "chest": [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "pain": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "on": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "aspirin": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def attention_table(weights, head):
    rows = []
    for i, query_word in enumerate(TOKENS):
        row = {"from / to": query_word}
        for j, key_word in enumerate(TOKENS):
            row[key_word] = round(float(weights[head, i, j]), 3)
        rows.append(row)
    table(rows, caption=f"Head {head}. Every row is one word's attention and sums to 1.")


def attention_chart(weights, head, query_index):
    values = [
        {"word": word, "attention": round(float(weights[head, query_index, j]), 3)}
        for j, word in enumerate(TOKENS)
    ]
    plot(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": f"What '{TOKENS[query_index]}' is looking at, head {head}",
            "width": {"step": 84},
            "height": 300,
            "data": {"values": values},
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {
                    "field": "word",
                    "type": "nominal",
                    "sort": None,
                    "axis": {"labelAngle": 0, "labelFontSize": 13, "labelPadding": 6, "titleFontSize": 13},
                },
                "y": {
                    "field": "attention",
                    "type": "quantitative",
                    "scale": {"domain": [0, 1]},
                    "axis": {"labelFontSize": 12, "titleFontSize": 13},
                },
                "color": {"field": "attention", "type": "quantitative", "legend": None},
                "tooltip": [{"field": "word"}, {"field": "attention"}],
            },
        }
    )


def attention_heatmap(weights):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for head, ax in enumerate(axes):
        ax.imshow(weights[head].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(TOKENS)))
        ax.set_xticklabels(TOKENS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(TOKENS)))
        ax.set_yticklabels(TOKENS, fontsize=9)
        ax.set_title(f"head {head}", fontsize=10)
    fig.tight_layout()
    return fig


def dose_chart(schedule):
    plot(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Daily paracetamol schedule",
            "width": {"step": 60},
            "height": 300,
            "data": {"values": schedule},
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {
                    "field": "hour",
                    "type": "ordinal",
                    "title": "hour",
                    "axis": {"labelAngle": 0, "labelFontSize": 13, "labelPadding": 6, "titleFontSize": 13},
                },
                "y": {
                    "field": "mg",
                    "type": "quantitative",
                    "title": "mg",
                    "axis": {"labelFontSize": 12, "titleFontSize": 13},
                },
                "tooltip": [{"field": "hour"}, {"field": "mg"}],
            },
        }
    )
