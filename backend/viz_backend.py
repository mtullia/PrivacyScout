# viz_backend.py
import requests, textwrap
import matplotlib.pyplot as plt

BASE = "http://127.0.0.1:8000"
SAMPLES = [
    "We may share personal information with third parties for analytics and advertising.",
    "We never sell your data and only collect minimal information necessary to provide the service.",
    "You may opt out of certain data sharing by adjusting your preferences."
]

def analyze(text):
    r = requests.post(f"{BASE}/analyze", json={"text": text})
    r.raise_for_status()
    return r.json()

def pretty(title):
    return "\n".join(textwrap.wrap(title, 60))

def plot_probabilities(ax, proba_dict, title):
    # Back-compat: if your /analyze doesn’t return per-class proba, synthesize from label/score.
    if not proba_dict:
        ax.text(0.5, 0.5, "No probabilities available.\n(Add /analyze_proba)", ha='center', va='center')
        ax.set_axis_off()
        return
    labels = list(proba_dict.keys())
    vals = [proba_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_title(pretty(title))
    ax.set_ylabel("Probability")

def plot_keywords(ax, keywords, title):
    if not keywords:
        ax.text(0.5, 0.5, "No keywords returned", ha='center', va='center')
        ax.set_axis_off()
        return
    labels = [k["text"] for k in keywords][:8]
    vals   = [k["weight"] for k in keywords][:8]
    ax.barh(labels[::-1], vals[::-1])
    ax.set_title(pretty(title))
    ax.set_xlabel("RAKE weight")

def main():
    # Try to call a richer endpoint if you add it later (see Option B below)
    has_proba = False

    fig_count = 0
    for text in SAMPLES:
        data = analyze(text)
        label = data.get("label", "?")
        score = data.get("score", 0.0)
        keywords = data.get("keywords", [])
        proba = data.get("proba", None)
        if proba: has_proba = True

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        title = f"Label: {label} (score={score:.2f})"
        plot_probabilities(ax1, proba, title)   # shows message if proba missing
        plot_keywords(ax2, keywords, "Top Keywords")
        plt.suptitle(pretty(text), fontsize=10, y=1.03)
        plt.tight_layout()
        fig_count += 1

    plt.show()
    if not has_proba:
        print("\nTip: add an /analyze_proba endpoint (Option B) to see per-class bars.\n")

if __name__ == "__main__":
    main()

