import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/housing_large.csv")

os.makedirs("static/graphs", exist_ok=True)
graph_path = "static/graphs/"

def save(name):
    plt.savefig(graph_path + name, bbox_inches="tight")
    plt.close()

sns.histplot(df["price"], kde=True)
plt.title("Price Distribution")
save("price_distribution.png")

sns.histplot(df["area"], kde=True, color="green")
plt.title("Area Distribution")
save("area_distribution.png")

sns.countplot(x=df["bedrooms"])
plt.title("Bedrooms Count")
save("bedrooms_count.png")

sns.countplot(x=df["bathrooms"], color="orange")
plt.title("Bathrooms Count")
save("bathrooms_count.png")

sns.countplot(x=df["balcony"], color="purple")
plt.title("Balcony Count")
save("balcony_count.png")

plt.figure(figsize=(12, 5))
df["location"].value_counts().plot(kind="bar")
plt.title("Location Frequency")
save("location_frequency.png")

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
save("correlation_heatmap.png")

sns.scatterplot(x=df["area"], y=df["price"])
plt.title("Price vs Area")
save("price_vs_area.png")

sns.boxplot(x=df["bedrooms"], y=df["price"])
plt.title("Price vs Bedrooms")
save("price_vs_bedrooms.png")

sns.boxplot(x=df["bathrooms"], y=df["price"], color="gold")
plt.title("Price vs Bathrooms")
save("price_vs_bathrooms.png")

print("✔ Graphs generated and saved to static/graphs/")
