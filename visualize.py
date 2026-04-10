"""
Visualization module -- produces the four figures described in the paper.

Fig 4.1  Top 15 in-demand skills (bar chart)
Fig 4.2  2-D projection of job embeddings (scatter plot via PCA)
Fig 4.3  Skill extraction performance vs baselines (grouped bar chart)
Fig 4.4  Skill demand heatmap by normalized job role
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")           # headless rendering (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_FIXED_COLORS = [
    "#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3",
    "#CCB974","#64B5CD","#E377C2","#7F7F7F","#BCBD22","#17BECF","#AEC7E8",
    "#FFBB78","#98DF8A","#FF9896","#C5B0D5","#C49C94","#F7B6D2","#DBDB8D",
    "#9EDAE5","#393B79","#637939","#8C6D31","#843C39","#7B4173","#5254A3",
    "#8CA252","#BD9E39","#AD494A","#A55194","#6B6ECF","#B5CF6B","#E7BA52",
    "#D6616B","#CE6DBD","#9C9EDE","#CEDB9C","#E7CB94","#E7969C","#DE9ED6",
    "#3182BD","#E6550D","#31A354","#756BB1","#636363","#6BAED6","#FD8D3C",
    "#74C476","#9E9AC8","#969696","#9ECAE1","#FDAE6B","#A1D99B","#BCBDDC",
    "#BDBDBD","#C6DBEF","#FDD0A2","#C7E9C0","#DADAEB","#D9D9D9",
    "#F7FBFF","#FFF5EB","#F7FCF5","#F2F0F7","#FFFFFF",
]

def _role_color(role: str, all_roles: list) -> str:
    idx = sorted(set(all_roles)).index(role) % len(_FIXED_COLORS)
    return _FIXED_COLORS[idx]


def fig41_top_skills(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 4.1 - Top 15 In-Demand Skills Across the Job Corpus."""
    all_skills = []
    for skills in df["pred_skills"]:
        all_skills.extend(skills)

    counts = Counter(all_skills).most_common(15)
    skills, freqs = zip(*counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(skills)), list(reversed(freqs)), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Number of Job Postings", fontsize=12)
    ax.set_title("Fig 4.1 - Top 15 In-Demand Skills Across the Job Corpus", fontsize=13, pad=12)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig41_top_skills.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    return fig


def fig42_embeddings(embeddings: np.ndarray, roles: list, save: bool = True) -> plt.Figure:
    """Fig 4.2 - 2-D Projection of Job Description Embeddings by Canonical Role."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_roles = sorted(set(roles))
    for role in unique_roles:
        idx = [i for i, r in enumerate(roles) if r == role]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   label=role, color=_role_color(role, roles), alpha=0.75, s=55, edgecolors="white", linewidths=0.4)

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("Fig 4.2 - 2-D Projection of Job Description Embeddings by Canonical Role",
                 fontsize=12, pad=10)
    n_roles = len(set(roles))
    ncol = max(1, n_roles // 20)
    ax.legend(fontsize=6, framealpha=0.8, loc="best", ncol=ncol,
              markerscale=0.8, handlelength=1)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig42_embeddings.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    return fig


def fig43_performance(metrics: dict, save: bool = True) -> plt.Figure:
    """Fig 4.3 - Skill Extraction Performance vs Baselines."""
    baselines = {
        "Keyword Match": {"Precision": 0.61, "Recall": 0.58, "F1": 0.59},
        "SkillSpan":     {"Precision": 0.72, "Recall": 0.68, "F1": 0.70},
        "LightXML":      {"Precision": 0.79, "Recall": 0.75, "F1": 0.77},
        "LLM Zero-Shot": {"Precision": 0.83, "Recall": 0.81, "F1": 0.82},
        "Proposed":      {"Precision": metrics["precision"],
                          "Recall":    metrics["recall"],
                          "F1":        metrics["f1"]},
    }

    models = list(baselines.keys())
    metrics_keys = ["Precision", "Recall", "F1"]
    x = np.arange(len(models))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, color) in enumerate(zip(metrics_keys, colors)):
        vals = [baselines[m][key] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=key, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Fig 4.3 - Skill Extraction Performance: Proposed vs Baselines", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig43_performance.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    return fig


def fig44_heatmap(df: pd.DataFrame, esco_skills: list, save: bool = True) -> plt.Figure:
    """Fig 4.4 - Skill Demand Heatmap by Normalized Job Role."""
    roles = df["normalized_role"].unique()
    all_skills = []
    for s in df["pred_skills"]:
        all_skills.extend(s)
    top_skills = [s for s, _ in Counter(all_skills).most_common(20)]

    matrix = pd.DataFrame(0.0, index=sorted(roles), columns=top_skills)
    for role in roles:
        role_df = df[df["normalized_role"] == role]
        role_skills = []
        for s in role_df["pred_skills"]:
            role_skills.extend(s)
        counts = Counter(role_skills)
        total = len(role_df)
        for skill in top_skills:
            matrix.loc[role, skill] = counts.get(skill, 0) / total

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, linecolor="white", ax=ax,
                cbar_kws={"label": "Demand Ratio"})
    ax.set_title("Fig 4.4 - Skill Demand Heatmap by Normalized Job Role", fontsize=12, pad=10)
    ax.set_xlabel("Skill", fontsize=11)
    ax.set_ylabel("Normalized Role", fontsize=11)
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig44_heatmap.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved -> {path}")
    return fig
