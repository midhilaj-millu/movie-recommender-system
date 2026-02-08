import streamlit as st
import pickle
import requests
import os

if not os.path.exists("similarity.pkl"):
    st.warning("Generating similarity file. This may take a moment...")
    import subprocess
    subprocess.run(["python", "main.py"])


# Load API key from Streamlit secrets
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

def fetch_poster(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    response = requests.get(url)
    data = response.json()

    poster_path = data.get("poster_path")
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return None


# Load pickled files
movies = pickle.load(open("movies_list.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

movies_list = movies["title"].values

st.header("🎬 Movie Recommender System")


st.subheader("Popular Movies 🍿")


# Movie selection
selectvalue = st.selectbox("Select movie from dropdown", movies_list)

def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []
    recommended_posters = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        poster = fetch_poster(movie_id)

        # Only include movies WITH posters
        if poster:
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_posters.append(poster)

    return recommended_movies[:5], recommended_posters[:5]



if st.button("Show Recommend"):
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
