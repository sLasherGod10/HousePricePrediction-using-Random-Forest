

# 🏡 **House Price Prediction using Random Forest (Flask Web App)**

This project is a **Machine Learning–powered House Price Prediction Web Application** built using **Flask**, **Python**, and **Random Forest Regression**.  
Users can input house features through a web interface, and the trained model predicts the price in real-time.

The project also includes:
- Data preprocessing & EDA  
- Model training script  
- Visualization graphs  
- Web-based UI (HTML + CSS)  

---

# 📁 **Project Folder Structure**

```
HousePricePrediction/
│
├── data/
│   └── housing_large.csv            # Dataset used for training
│
├── model/
│   ├── columns.pkl                  # Stores feature column names
│   ├── regression_model.pkl         # Trained Random Forest model (ignored in git)
│   ├── scaler.pkl                   # Scaler for feature normalization
│
├── static/
│   ├── style.css                    # Frontend CSS styles
│   └── graphs/
│       └── future_trend.png         # Trend graph for UI display
│
├── templates/
│   ├── index.html                   # Home page (input form)
│   ├── result.html                  # Prediction result page
│   └── graphs.html                  # Graph visualization page
│
├── app.py                           # Main Flask app (runs the website)
├── train_model.py                   # Model training script
├── eda.py                           # Exploratory Data Analysis (optional)
├── requirements.txt                 # Project dependencies
└── .gitignore                       # Ignore model & large files
```

---

# 🚀 **Features**

### ✔ **Machine Learning**
- Random Forest Regression model  
- Handles missing values & scaling  
- Uses saved model artifacts (`.pkl` files)  

### ✔ **Web Application (Flask)**
- Form-based user input  
- Predicts house price in real time  
- Displays graphs and trends  
- Clean UI with HTML/CSS  

### ✔ **Data Visualization**
- Trend graphs  
- Model insights  
- PCA / distribution charts (optional)

---

# 🧠 **Model Training**

To retrain the model, run:

```
python train_model.py
```

This script:
- Loads the dataset  
- Cleans & preprocesses data  
- Trains Random Forest  
- Saves files inside `model/`  
  - `regression_model.pkl`  
  - `scaler.pkl`  
  - `columns.pkl`  

---

# 🌐 **Run the Flask Application**

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

Then open your browser:

```
http://127.0.0.1:5000/
```

---

# 📸 **Web Interface Screens**

### **1️⃣ Home Page — Input Features**

Users enter house features such as:
- Bedrooms  
- Bathrooms  
- Area  
- Location  
- Age of property  
- More…

### **2️⃣ Prediction Result Page**

Displays:
- Estimated price  
- Feature summary  

### **3️⃣ Graph Visualization Page**

Shows:
- Trend graph (`future_trend.png`)  
- Any additional PCA or EDA graphs  

---

# 📦 **Requirements**

All dependencies are listed in `requirements.txt`:

Example:
```
Flask
pandas
numpy
scikit-learn
matplotlib
```

Install them using:

```
pip install -r requirements.txt
```

---

# 🔧 **.gitignore**

Large ML files such as `.pkl` models are ignored:

```
*.pkl
model/*.pkl
```

---

# 🤝 **Contributors**
- **Atharva Khaire**


✅ Better screenshots section  
✅ Badges (Python version, Flask, License, etc.)  
Just tell me!
