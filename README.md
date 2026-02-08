# 🎬 Movie Recommender System

A **content-based movie recommendation system** built using **Machine Learning, NLP, and Streamlit**.  
The system recommends movies similar to a selected movie based on textual similarity of movie descriptions.

---

## 📌 Project Overview

This project analyzes movie descriptions and recommends the **top 5 most similar movies** using
Natural Language Processing techniques and cosine similarity.

The application is deployed as an interactive **Streamlit web app** and fetches movie posters using the **TMDB API**.

---

## 🚀 Features

- Content-based movie recommendations
- NLP-based similarity using movie overviews
- Cosine similarity for recommendation ranking
- Displays movie posters using TMDB API
- Clean, responsive Streamlit UI
- Fast performance using precomputed data

---

## 🧠 Recommendation Approach

- **Type:** Content-Based Filtering  
- **Technique:** Natural Language Processing (NLP)  
- **Similarity Metric:** Cosine Similarity  

Movie overviews are converted into numerical vectors using `CountVectorizer`, and similarity is calculated between movies based on vector distance.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Web Framework:** Streamlit  
- **API:** The Movie Database (TMDB)  
- **ML Concepts:** NLP, Vectorization, Similarity Metrics  

---

## 📂 Project Structure

