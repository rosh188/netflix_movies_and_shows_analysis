from pathlib import Path
import pandas as pd


def load_and_clean(csv_path: str = "data/netflix_titles.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    str_cols = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df["date_added_dt"] = pd.to_datetime(
        df["date_added"].fillna("").astype(str).str.strip().replace("", pd.NA),
        format="mixed",
        errors="coerce",
    )
    df["year_added"] = df["date_added_dt"].dt.year
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    df["duration_min"] = pd.to_numeric(
        df["duration"].str.extract(r"(\d+)\s*min", expand=False), errors="coerce"
    )
    df["seasons"] = pd.to_numeric(
        df["duration"].str.extract(r"(\d+)\s*Season", expand=False), errors="coerce"
    )

    df["country_clean"] = df["country"].replace("", "Unknown")
    df["director_clean"] = df["director"].replace("", "Unknown")
    df["cast_clean"] = df["cast"].replace("", "Unknown")

    return df


def get_cleaned_df() -> pd.DataFrame:
    parquet_path = Path("data/netflix_cleaned.parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    df = load_and_clean("data/netflix_titles.csv")
    df.to_parquet(parquet_path, index=False)
    return df


def explode_column(df: pd.DataFrame, col: str, clean_name: str | None = None) -> pd.DataFrame:
    target = clean_name or col
    exploded = df.assign(_items=df[target].str.split(",")).explode("_items")
    exploded[col] = exploded["_items"].astype(str).str.strip()
    exploded = exploded[exploded[col] != ""]
    return exploded.drop(columns=["_items"])


def analyze_countries_and_genres(df: pd.DataFrame, top_n: int = 10) -> None:
    countries_df = explode_column(df, "country_exploded", "country_clean").reset_index(drop=True)
    known_countries = countries_df[countries_df["country_exploded"] != "Unknown"]

    print(f"\n--- Top {top_n} Producing Countries (Known) ---")
    country_counts = known_countries["country_exploded"].value_counts().head(top_n)
    print(country_counts)

    print(f"\n--- Movie vs TV Show Split (Top {top_n} Countries) ---")
    top_country_names = country_counts.index.tolist()
    top_df = known_countries[known_countries["country_exploded"].isin(top_country_names)]

    pivot_type = pd.crosstab(top_df["country_exploded"], top_df["type"])
    pivot_type["total"] = pivot_type.sum(axis=1)
    pivot_type["tv_share_pct"] = (
        pivot_type.get("TV Show", 0) / pivot_type["total"] * 100
    ).round(1)
    sort_col = "Movie" if "Movie" in pivot_type.columns else pivot_type.columns[0]
    print(pivot_type.sort_values(by=sort_col, ascending=False))

    genres_df = explode_column(df, "genre", "listed_in").reset_index(drop=True)
    print(f"\n--- Top {top_n} Genres Overall ---")
    print(genres_df["genre"].value_counts().head(top_n))


def analyze_tv_vs_movie_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend_df = df.dropna(subset=["year_added"]).copy()
    trend_df["year_added"] = trend_df["year_added"].astype(int)

    counts = pd.crosstab(trend_df["year_added"], trend_df["type"])
    counts["total_added"] = counts.sum(axis=1)
    counts["movie_pct"] = (counts.get("Movie", 0) / counts["total_added"] * 100).round(1)
    counts["tv_show_pct"] = (counts.get("TV Show", 0) / counts["total_added"] * 100).round(1)

    print("\n--- Titles Added by Year (Movies vs TV Shows) ---")
    print(counts.tail(10))
    return counts


def summarize_catalog(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print("\nType breakdown:")
    print(df["type"].value_counts())
    print((df["type"].value_counts(normalize=True) * 100).round(2))

    key_cols = ["director", "cast", "country", "date_added", "rating"]
    print("\nEmpty/missing counts in key columns:")
    print({col: int((df[col] == "").sum()) for col in key_cols})

    if df["year_added"].notna().any():
        print(
            "\nYear added range:",
            int(df["year_added"].min()),
            "to",
            int(df["year_added"].max()),
        )


if __name__ == "__main__":
    csv_file = Path("data/netflix_titles.csv")
    if not csv_file.exists():
        raise FileNotFoundError("Place netflix_titles.csv inside data/ before running.")

    cleaned_df = load_and_clean(str(csv_file))
    summarize_catalog(cleaned_df)
    cleaned_df.to_parquet("data/netflix_cleaned.parquet", index=False)
    print("\nSaved data/netflix_cleaned.parquet")

    analyze_countries_and_genres(cleaned_df)
    analyze_tv_vs_movie_trend(cleaned_df)