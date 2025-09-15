# netflix_analysis.py
# Complete EDA + stats + numpy + calculus + feature engineering for the Netflix dataset.
# Assumes file 'netflix_titles.csv' is in the working directory. If not, change DATA_PATH below.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import scipy.stats as stats
import warnings
import sympy as sp


warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook")

# ---------- Configuration ----------
DATA_PATH = "netflix_titles.csv"  # change if needed
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Utility functions ----------
def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

def ensure_loaded(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Place 'netflix_titles.csv' there or update DATA_PATH.")
    return pd.read_csv(path)

# ---------- Part 1: Basic Data Understanding ----------
def basic_data_understanding(df):
    print("----- Basic Data Understanding -----\n")
    print("First 10 rows:\n", df.head(10), "\n")
    print("Shape:", df.shape)
    print("Columns & dtypes:\n", df.dtypes, "\n")
    print("Missing values per column:\n", df.isnull().sum(), "\n")
    print("Duplicate rows count:", df.duplicated().sum(), "\n")
    print("Describe (numerical):\n", df.describe(include=[np.number]), "\n")
    print("Describe (categorical):\n", df.describe(include=[object]), "\n")
    print("Unique values per column:\n", df.nunique(), "\n")
    # Most frequent value in country (handle NaNs)
    most_freq_country = df['country'].dropna().mode()
    print("Most frequent country (mode):", most_freq_country.values if not most_freq_country.empty else "No data")
    return

# ---------- Part 2: EDA ----------
def eda_plots(df):
    print("----- EDA Plots -----")
    # Clean minor columns
    df_local = df.copy()
    # Ensure release_year numeric
    df_local['release_year'] = pd.to_numeric(df_local['release_year'], errors='coerce')

    # 1. Bar chart: Movies vs TV Shows
    counts = df_local['type'].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Count: Movies vs TV Shows")
    ax.set_ylabel("Count")
    save_fig(fig, "movies_vs_tv_count.png")
    plt.show()

    # 2. Top 10 countries producing most content (split multi-country entries)
    # Expand countries by splitting on comma
    country_series = df_local['country'].dropna().astype(str).str.split(",")
    country_exploded = country_series.explode().str.strip()
    top_countries = country_exploded.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(y=top_countries.index, x=top_countries.values, ax=ax)
    ax.set_title("Top 10 Countries by Count of Titles")
    ax.set_xlabel("Number of Titles")
    save_fig(fig, "top10_countries.png")
    plt.show()

    # 3. Top 10 most common genres in listed_in
    # listed_in has comma-separated categories (genres)
    genres_series = df_local['listed_in'].dropna().astype(str).str.split(",").explode().str.strip()
    top_genres = genres_series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(y=top_genres.index, x=top_genres.values, ax=ax)
    ax.set_title("Top 10 Genres (listed_in)")
    ax.set_xlabel("Count")
    save_fig(fig, "top10_genres.png")
    plt.show()

    # 4. Scatter plot: release_year vs count of titles (grouped by year)
    yearly_counts = df_local.groupby('release_year').size().reset_index(name='count').dropna()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=yearly_counts, x='release_year', y='count', ax=ax)
    ax.set_title("Release Year vs Number of Titles")
    save_fig(fig, "release_year_vs_count.png")
    plt.show()

    # 5. Histogram of movie durations (in minutes)
    # 'duration' column may be like "90 min" for movies, "3 Seasons" for shows.
    movie_durations = df_local[df_local['type'].str.lower() == 'movie']['duration'].dropna().astype(str)
    # extract numeric minutes
    movie_minutes = movie_durations.str.extract(r'(\d+)').astype(float).squeeze()
    movie_minutes = movie_minutes.dropna()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(movie_minutes, bins=30, kde=False, ax=ax)
    ax.set_title("Histogram of Movie Durations (minutes)")
    ax.set_xlabel("Duration (minutes)")
    save_fig(fig, "movie_duration_histogram.png")
    plt.show()

    # 6. Bar chart: number of titles added per year (based on date_added)
    # parse date_added
    df_local['date_added_parsed'] = pd.to_datetime(df_local['date_added'], errors='coerce')
    df_local['added_year'] = df_local['date_added_parsed'].dt.year
    added_per_year = df_local['added_year'].value_counts().sort_index().dropna()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=added_per_year.index.astype(str), y=added_per_year.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Titles Added per Year (based on date_added)")
    ax.set_xlabel("Year added")
    ax.set_ylabel("Number of Titles")
    save_fig(fig, "titles_added_per_year.png")
    plt.show()

    # 7. Year with highest number of releases (by release_year)
    top_release_year = yearly_counts.loc[yearly_counts['count'].idxmax(), 'release_year']
    top_release_count = yearly_counts['count'].max()
    print(f"Year with highest number of releases: {int(top_release_year)} ({top_release_count} titles)")

    return {
        "counts_type": counts,
        "top_countries": top_countries,
        "top_genres": top_genres,
        "yearly_counts": yearly_counts,
        "movie_minutes": movie_minutes,
        "added_per_year": added_per_year,
        "top_release_year": (top_release_year, top_release_count)
    }

# ---------- Part 3: Statistics ----------
def statistics_section(df):
    print("----- Statistics -----")
    ry = pd.to_numeric(df['release_year'], errors='coerce').dropna().astype(int)

    mean_ = ry.mean()
    median_ = ry.median()
    # mode might be multimodal; use scipy.stats.mode (returns smallest mode)
    try:
        mode_val = stats.mode(ry, keepdims=True).mode[0]
    except Exception:
        mode_val = ry.mode().iloc[0] if not ry.mode().empty else np.nan
    var_ = ry.var(ddof=0)
    std_ = ry.std(ddof=0)
    print(f"Release year -> mean: {mean_:.2f}, median: {median_}, mode: {mode_val}, variance: {var_:.2f}, std: {std_:.2f}")

    # Percentage of titles added in last 5 years (assuming current year = 2025)
    current_year = 2025
    added_years = pd.to_datetime(df['date_added'], errors='coerce').dt.year
    total_titles = len(df)
    last5_count = ((added_years >= (current_year - 5)) & added_years.notna()).sum()
    pct_last5 = (last5_count / total_titles) * 100 if total_titles > 0 else 0
    print(f"Percentage of titles added in last 5 years (>= {current_year-5}): {pct_last5:.2f}% (count: {last5_count}/{total_titles})")

    # Most common rating for Movies and TV Shows separately
    mode_movie_rating = df[df['type'].str.lower() == 'movie']['rating'].dropna().mode()
    mode_show_rating = df[df['type'].str.lower() == 'tv show']['rating'].dropna().mode()
    print("Most common rating (Movies):", mode_movie_rating.values if not mode_movie_rating.empty else "No data")
    print("Most common rating (TV Shows):", mode_show_rating.values if not mode_show_rating.empty else "No data")

    return {
        "mean": mean_,
        "median": median_,
        "mode": mode_val,
        "variance": var_,
        "std": std_,
        "pct_last5": pct_last5,
        "mode_movie_rating": mode_movie_rating,
        "mode_tv_rating": mode_show_rating
    }

# ---------- Part 4: Linear Algebra & NumPy ----------
def linear_algebra_numpy(df):
    print("----- Linear Algebra & NumPy -----")
    release_year = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int).values
    is_movie = (df['type'].str.lower() == 'movie').astype(int).values

    # 1. create numpy arrays (already)
    arr_release = np.array(release_year)
    arr_is_movie = np.array(is_movie)

    # 3. vector addition
    vec_add = arr_release + arr_is_movie

    # 4. dot product
    dot_prod = np.dot(arr_release, arr_is_movie)

    # 5. feature matrix multiplication with weights [0.6, 0.4]
    feature_matrix = np.vstack([arr_release, arr_is_movie]).T  # shape (n,2)
    weights = np.array([0.6, 0.4])
    weighted = feature_matrix.dot(weights)

    # 6. normalize release_year using L2 normalization
    norm = np.linalg.norm(arr_release)
    normalized_release = arr_release / norm if norm != 0 else arr_release

    print("Vector addition (first 10):", vec_add[:10])
    print("Dot product (release_year · is_movie):", int(dot_prod))
    print("Weighted feature sample (first 10):", weighted[:10])
    print("Normalized release_year sample (first 10):", normalized_release[:10])

    return {
        "arr_release": arr_release,
        "arr_is_movie": arr_is_movie,
        "vec_add": vec_add,
        "dot_prod": dot_prod,
        "weighted": weighted,
        "normalized_release": normalized_release
    }

# ---------- Part 5: Calculus ----------
def calculus_part():
    print("----- Calculus -----")
    # Popularity_Score = (is_movie * release_year) + 0.5 * (release_year - 2000)^2
    r = sp.symbols('r')
    m = sp.symbols('m')  # is_movie treated like constant (0 or 1)
    Popularity_Score = m * r + sp.Rational(1,2) * (r - 2000)**2
    derivative = sp.diff(Popularity_Score, r)
    print("Popularity_Score:", Popularity_Score)
    print("Derivative wrt release_year r:", derivative)
    # derivative expands to: m + (r - 2000)
    return derivative

# ---------- Part 6: Feature Engineering ----------
def feature_engineering(df):
    print("----- Feature Engineering -----")
    df2 = df.copy()
    df2['release_year'] = pd.to_numeric(df2['release_year'], errors='coerce')
    df2['content_age'] = 2025 - df2['release_year']  # as asked
    df2['is_movie'] = (df2['type'].str.lower() == 'movie').astype(int)
    df2['recent_release'] = (df2['release_year'] >= 2020).astype(int)
    # num_genres: count of genres in 'listed_in'
    df2['num_genres'] = df2['listed_in'].fillna('').apply(lambda s: len([x for x in s.split(',') if x.strip()])) 
    # quartile ranking of release_year
    df2['release_year_rank_quartile'] = pd.qcut(df2['release_year'].fillna(-9999), q=4, labels=False, duplicates='drop')
    # Save engineered features snapshot
    fe_cols = ['title', 'release_year', 'content_age', 'is_movie', 'recent_release', 'num_genres', 'release_year_rank_quartile']
    df2[fe_cols].head(10).to_csv(OUTPUT_DIR / "feature_engineered_sample.csv", index=False)
    print("Saved feature engineered sample to outputs/feature_engineered_sample.csv")
    return df2

# ---------- Part 7: SQL Simulation in Pandas ----------
def sql_simulation(df):
    print("----- SQL-like Pandas Queries -----")
    df_local = df.copy()
    df_local['release_year'] = pd.to_numeric(df_local['release_year'], errors='coerce')

    q1 = df_local[(df_local['type'].str.lower() == 'movie') & (df_local['release_year'] > 2015)]
    q2 = q1.sort_values(by=['release_year', 'title'], ascending=[False, True])
    # top 5 countries by number of titles (explode country field)
    country_exploded = df_local['country'].dropna().astype(str).str.split(",").explode().str.strip()
    top5_countries = country_exploded.value_counts().head(5)
    q3 = top5_countries
    q4 = df_local[(df_local['release_year'] >= 2000) & (df_local['release_year'] <= 2010)]
    # Count how many titles have "Drama" in listed_in
    drama_count = df_local['listed_in'].dropna().str.contains('Drama', case=False, na=False).sum()

    print(f"Movies with release_year > 2015: {len(q1)}")
    print("Top 5 countries by number of titles:\n", q3)
    print(f"Titles released between 2000 and 2010: {len(q4)}")
    print(f"Titles with 'Drama' in listed_in: {drama_count}")

    # Save some results
    q1.head(20).to_csv(OUTPUT_DIR / "movies_after_2015_sample.csv", index=False)
    q2.head(20).to_csv(OUTPUT_DIR / "movies_after_2015_sorted_sample.csv", index=False)
    q4.head(20).to_csv(OUTPUT_DIR / "titles_2000_2010_sample.csv", index=False)
    return {
        "movies_after_2015": q1,
        "movies_after_2015_sorted": q2,
        "top5_countries": q3,
        "titles_2000_2010": q4,
        "drama_count": drama_count
    }

# ---------- Part 8: Insights ----------
def generate_insights(df, eda_results):
    print("----- Insights -----")
    # Which country produces the most Netflix content?
    country_series = df['country'].dropna().astype(str).str.split(",").explode().str.strip()
    top_country = country_series.value_counts().idxmax()
    top_country_count = country_series.value_counts().max()
    print(f"Country producing most content: {top_country} ({top_country_count} titles)")

    # Which year had the highest content release?
    top_release_year, top_release_count = eda_results['top_release_year']
    print(f"Year with highest content releases: {int(top_release_year)} ({top_release_count} titles)")

    # Movies or TV Shows more frequent?
    type_counts = df['type'].value_counts()
    more_freq = type_counts.idxmax()
    print(f"More frequent: {more_freq} ({type_counts.max()} titles)")

    # Most frequent genre
    top_genre = eda_results['top_genres'].idxmax()
    top_genre_count = eda_results['top_genres'].max()
    print(f"Most frequent genre: {top_genre} ({top_genre_count} occurrences)")

    insights = {
        "top_country": (top_country, top_country_count),
        "top_release_year": (int(top_release_year), top_release_count),
        "more_freq_type": (more_freq, int(type_counts.max())),
        "top_genre": (top_genre, int(top_genre_count))
    }
    # Save to CSV
    pd.DataFrame({
        "insight": ["top_country", "top_release_year", "more_freq_type", "top_genre"],
        "value": [f"{top_country} ({top_country_count})", f"{int(top_release_year)} ({top_release_count})", f"{more_freq} ({type_counts.max()})", f"{top_genre} ({top_genre_count})"]
    }).to_csv(OUTPUT_DIR / "insights_summary.csv", index=False)
    print("Saved insights_summary.csv")
    return insights

# ---------- Main flow ----------
def main():
    df = ensure_loaded(DATA_PATH)
    basic_data_understanding(df)
    eda_results = eda_plots(df)
    stats_results = statistics_section(df)
    la_results = linear_algebra_numpy(df)
    derivative = calculus_part()
    df_fe = feature_engineering(df)
    sql_results = sql_simulation(df)
    insights = generate_insights(df, eda_results)
    print("\n--- Done. All outputs (plots, CSV samples) are in the 'outputs/' folder. ---")

if __name__ == "__main__":
    main()
