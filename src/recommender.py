import re
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 45)  # caps cell width so one long genre string doesn't push others off-screen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clean_eda import get_cleaned_df


def _clean_token_list(raw_text: str, limit: int | None = None) -> list[str]:
    if not raw_text:
        return []
    items = [item.strip().lower().replace(" ", "_") for item in raw_text.split(",") if item.strip()]
    return items[:limit] if limit else items


def build_content_soup(row: pd.Series) -> str:
    genres = _clean_token_list(row.get("listed_in", ""))
    directors = _clean_token_list(row.get("director", ""), limit=3)
    cast = _clean_token_list(row.get("cast", ""), limit=5)
    title_clean = re.sub(r"[^a-z0-9\s]", " ", str(row.get("title", "")).lower())
    desc_clean = re.sub(r"[^a-z0-9\s]", " ", str(row.get("description", "")).lower())
    content_type = str(row.get("type", "")).lower().replace(" ", "_")

    # Repeat high-signal metadata so genres, cast, and directors carry stronger weight
    weighted_parts = (
        [content_type] * 2
        + genres * 3
        + directors * 2
        + cast * 2
        + [title_clean, desc_clean]
    )
    return " ".join(part for part in weighted_parts if part)


class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.df["soup"] = self.df.apply(build_content_soup, axis=1)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        self._title_to_idx: dict[str, int] = {}
        for idx, title in enumerate(self.df["title"]):
            key = title.strip().lower()
            if key not in self._title_to_idx:
                self._title_to_idx[key] = idx

    def resolve_title(self, query: str) -> int | None:
        q = query.strip().lower()
        if q in self._title_to_idx:
            return self._title_to_idx[q]
        matches = self.df[self.df["title"].str.lower().str.contains(re.escape(q), na=False)]
        if not matches.empty:
            return int(matches.index[0])
        return None

    def recommend(
        self,
        title: str,
        top_n: int = 5,
        same_type_only: bool = False,
    ) -> pd.DataFrame:
        idx = self.resolve_title(title)
        if idx is None:
            raise ValueError(f"Title not found matching: '{title}'")

        source_type = self.df.loc[idx, "type"]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        rows = []
        for candidate_idx, score in scores:
            if candidate_idx == idx:
                continue
            if same_type_only and self.df.loc[candidate_idx, "type"] != source_type:
                continue
            rows.append(
                {
                    "title": self.df.loc[candidate_idx, "title"],
                    "type": self.df.loc[candidate_idx, "type"],
                    "listed_in": self.df.loc[candidate_idx, "listed_in"],
                    "release_year": self.df.loc[candidate_idx, "release_year"],
                    "similarity": round(float(score), 4),
                }
            )
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows)


if __name__ == "__main__":
    df = get_cleaned_df()
    recommender = ContentRecommender(df)

    sample_titles = ["Stranger Things", "Inception", "Sacred Games"]
    for sample in sample_titles:
        idx = recommender.resolve_title(sample)
        if idx is not None:
            matched_name = df.loc[idx, "title"]
            print(f"\n=== Top 5 Similar to: {matched_name} ({df.loc[idx, 'type']}) ===")
            print(recommender.recommend(matched_name, top_n=5).to_string(index=False))