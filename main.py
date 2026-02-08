import pandas as pd

# Load TMDB dataset (10k movies)
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print(data.head())
