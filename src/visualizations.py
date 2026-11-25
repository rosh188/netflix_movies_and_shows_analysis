from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from clean_eda import explode_column, get_cleaned_df
from network_analysis import build_collaboration_graph

# Consistent dark Netflix-inspired styling
NETFLIX_RED = "#E50914"
NETFLIX_DARK = "#141414"
NETFLIX_CARD = "#1F1F1F"
NETFLIX_GREY = "#8C8C8C"
ACCENT_BLUE = "#3B82F6"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": NETFLIX_DARK,
            "axes.facecolor": NETFLIX_CARD,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "white",
            "xtick.color": "#CCCCCC",
            "ytick.color": "#CCCCCC",
            "text.color": "white",
            "grid.color": "#2C2C2C",
            "font.sans-serif": "DejaVu Sans",
        }
    )


def plot_tv_vs_movie_trend(df: pd.DataFrame, out_dir: Path) -> None:
    trend_df = df.dropna(subset=["year_added"]).copy()
    trend_df["year_added"] = trend_df["year_added"].astype(int)
    trend_df = trend_df[trend_df["year_added"] >= 2015]

    counts = pd.crosstab(trend_df["year_added"], trend_df["type"])
    totals = counts.sum(axis=1)
    tv_share = counts.get("TV Show", 0) / totals * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Stacked Bar of Additions
    years = counts.index.astype(str)
    movies = counts.get("Movie", 0)
    tv_shows = counts.get("TV Show", 0)

    ax1.bar(years, movies, label="Movie", color=NETFLIX_RED, width=0.6)
    ax1.bar(years, tv_shows, bottom=movies, label="TV Show", color=ACCENT_BLUE, width=0.6)
    ax1.set_title("Titles Added by Year (2015–2021)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Year Added")
    ax1.set_ylabel("Number of Titles")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=False)

    # Right: TV Show Share % Line
    ax2.plot(years, tv_share, marker="o", linewidth=2.8, markersize=8, color=ACCENT_BLUE)
    for x_idx, val in enumerate(tv_share):
        ax2.annotate(f"{val:.1f}%", (x_idx, val + 1.2), ha="center", fontsize=9, color="white")
    ax2.set_ylim(0, max(tv_share) + 10)
    ax2.set_title("TV Show Share of Annual Additions (%)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Year Added")
    ax2.set_ylabel("TV Show Share (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "1_tv_vs_movie_trend.png", dpi=200)
    plt.close()


def plot_top_countries_split(df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:
    countries_df = explode_column(df, "country_exploded", "country_clean").reset_index(drop=True)
    known = countries_df[countries_df["country_exploded"] != "Unknown"]

    top_names = known["country_exploded"].value_counts().head(top_n).index.tolist()
    sub = known[known["country_exploded"].isin(top_names)]

    pivot = pd.crosstab(sub["country_exploded"], sub["type"]).loc[top_names][::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(pivot.index, pivot.get("Movie", 0), color=NETFLIX_RED, label="Movie")
    ax.barh(
        pivot.index,
        pivot.get("TV Show", 0),
        left=pivot.get("Movie", 0),
        color=ACCENT_BLUE,
        label="TV Show",
    )

    ax.set_title(f"Top {top_n} Producing Countries: Movies vs. TV Shows", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Titles")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / "2_top_countries_split.png", dpi=200)
    plt.close()


def plot_top_genres(df: pd.DataFrame, out_dir: Path, top_n: int = 12) -> None:
    genres_df = explode_column(df, "genre", "listed_in").reset_index(drop=True)
    top_genres = genres_df["genre"].value_counts().head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top_genres.index, top_genres.values, color=NETFLIX_RED)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 35, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)

    ax.set_xlim(0, top_genres.max() * 1.12)
    ax.set_title(f"Top {top_n} Genres on Netflix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Title Count")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "3_top_genres.png", dpi=200)
    plt.close()


def plot_collaboration_network(df: pd.DataFrame, out_dir: Path, min_collabs: int = 6) -> None:
    # Filter to strong repeat collaborators so the visual stays clean and readable
    g = build_collaboration_graph(df, min_collaborations=min_collabs)

    fig, ax = plt.subplots(figsize=(13, 10))
    pos = nx.spring_layout(g, k=0.65, seed=42)

    degrees = dict(g.degree())
    node_sizes = [180 + degrees[n] * 95 for n in g.nodes()]
    edge_weights = [g[u][v]["weight"] * 0.35 for u, v in g.edges()]

    nx.draw_networkx_edges(g, pos, ax=ax, width=edge_weights, edge_color="#555555", alpha=0.75)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=NETFLIX_RED,
        edgecolors="white",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8.5, font_color="white", font_weight="bold")

    ax.set_title(
        f"Frequent Cast & Director Collaborations (≥ {min_collabs} Shared Titles)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "4_collaboration_network.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    setup_style()
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    cleaned_df = get_cleaned_df()
    plot_tv_vs_movie_trend(cleaned_df, output_folder)
    plot_top_countries_split(cleaned_df, output_folder)
    plot_top_genres(cleaned_df, output_folder)
    plot_collaboration_network(cleaned_df, output_folder, min_collabs=6)

    print("Saved 4 charts to outputs/:")
    for img in sorted(output_folder.glob("*.png")):
        print(" -", img)