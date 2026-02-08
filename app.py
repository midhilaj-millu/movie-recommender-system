import streamlit as st
import pickle
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ===============================
# Load TMDB API Key
# ===============================
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# ===============================
# Helper: Fetch Poster
# ===============================
def fetch_poster(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    poster_path = data.get("poster_path")
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return None


# ===============================
# Load Movies List (small file)
# ===============================
movies = pickle.load(open("movies_list.pkl", "rb"))
movies_list = movies["title"].values


# ===============================
# SAFE Similarity Handling (Cloud)
# ===============================
if "similarity" not in st.session_state:

    if os.path.exists("similarity.pkl"):
        # Load if already generated
        similarity = pickle.load(open("similarity.pkl", "rb"))
        st.session_state["similarity"] = similarity

    else:
        # Generate on first cloud run
        st.warning("Generating similarity matrix (first run only)… ⏳")

        df = pd.read_csv("dataset.csv")
        df = df[["id", "title", "overview"]]
        df.dropna(inplace=True)

        cv = CountVectorizer(
            max_features=5000,
            stop_words="english"
        )
        vectors = cv.fit_transform(df["overview"]).toarray()

        similarity = cosine_similarity(vectors)

        pickle.dump(similarity, open("similarity.pkl", "wb"))
        st.session_state["similarity"] = similarity

        st.success("Similarity matrix generated successfully ✅")

# Use cached similarity
similarity = st.session_state["similarity"]


# ===============================
# UI
# ===============================
st.header("🎬 Movie Recommender System")
st.subheader("Find movies similar to your favorite 🍿")

selectvalue = st.selectbox(
    "Select a movie",
    movies_list
)


# ===============================
# Recommendation Logic
# ===============================
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []
    recommended_posters = []

    for i in distances[1:10]:
        movie_id = movies.iloc[i[0]].id
        poster = fetch_poster(movie_id)

        if poster:
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_posters.append(poster)

        if len(recommended_movies) == 5:
            break

    return recommended_movies, recommended_posters


# ===============================
# Display Recommendations
# ===============================
if st.button("Show Recommendations"):
    movie_names, movie_posters = recommend(selectvalue)

    cols = st.columns(5)

    for i in range(len(movie_names)):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    height: 60px;
                    text-align: center;
                    font-weight: 600;
                    font-size: 16px;
                    line-height: 1.2;
                    overflow: hidden;
                ">
                    {movie_names[i]}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(movie_posters[i], use_container_width=True)
