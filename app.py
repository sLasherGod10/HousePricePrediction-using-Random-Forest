from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("model/regression_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Define realistic annual growth rates by city (in %)
CITY_GROWTH_RATES = {
    'Mumbai': 9.5,
    'Delhi': 8.8,
    'Bangalore': 10.2,
    'Hyderabad': 9.0,
    'Pune': 8.5,
    'Chennai': 7.8,
    'Nagpur': 6.5
}

def generate_future_price_graph(current_price, location, area, save_path="static/graphs/future_trend.png"):
    years = [2025, 2026, 2027, 2028, 2029, 2030]
    rate = CITY_GROWTH_RATES.get(location, 7.0) / 100  # default 7%
    
    prices = [current_price]
    for _ in range(5):
        next_price = prices[-1] * (1 + rate)
        prices.append(round(next_price))

    plt.figure(figsize=(10, 6))
    plt.plot(years, prices, marker='o', linewidth=3, markersize=8, color='#667eea')
    plt.fill_between(years, prices, alpha=0.15, color='#667eea')
    plt.title(f"Future Price Trend in {location} ({area} sq ft house)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Price (₹ in Lakhs)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x/100000:.1f}L"))

    for i, price in enumerate(prices):
        plt.text(years[i], price + max(prices)*0.02, f"₹{price/100000:.1f}L", 
                ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    balcony = int(request.form["balcony"])
    location = request.form["location"]

    locations = ['Pune','Mumbai','Nagpur','Bangalore','Hyderabad','Chennai','Delhi']
    loc_encoded = [1 if location == city else 0 for city in locations]

    features = np.array([[area, bedrooms, bathrooms, balcony] + loc_encoded])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    current_price = int(prediction)

    # Generate future trend graph
    os.makedirs("static/graphs", exist_ok=True)
    graph_path = "static/graphs/future_trend.png"
    generate_future_price_graph(current_price, location, area, graph_path)

    # Future prices (next 5 years)
    rate = CITY_GROWTH_RATES.get(location, 7.0) / 100
    future_prices = []
    price = current_price
    for year in range(2026, 2031):
        price = int(price * (1 + rate))
        future_prices.append((year, price))

    return render_template("result.html",
                           price=current_price,
                           location=location,
                           area=area,
                           future_prices=future_prices,
                           growth_rate=CITY_GROWTH_RATES.get(location, 7.0))

@app.route("/graphs")
def graphs():
    os.makedirs("static/graphs", exist_ok=True)
    graph_files = [f"graphs/{f}" for f in os.listdir("static/graphs") if f.endswith(('.png', '.jpg', '.jpeg'))]
    # Always include future trend if exists
    if "graphs/future_trend.png" in graph_files:
        graph_files.remove("graphs/future_trend.png")
        graph_files.insert(0, "graphs/future_trend.png")  # Show latest prediction first

    return render_template("graphs.html", graphs=graph_files)

if __name__ == "__main__":
    app.run(debug=True)