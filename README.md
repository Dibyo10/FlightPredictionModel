# ✈️ Flight Price Prediction Model - Model Card

## 📌 Model Summary

This model is a **Linear Regression-based flight price prediction system**, trained to estimate airline ticket prices using structured flight-related features. It was developed entirely from scratch using `scikit-learn`, and trained on a cleaned and feature-engineered version of the popular flight fare dataset.

**Key details:**
- **Algorithm:** `LinearRegression` (from sklearn)
- **Input:** Airline, Source, Destination, Journey details, Stops, Duration, Additional Info
- **Output:** Estimated flight price (in INR)
- **Evaluation Metric:** R² ≈ 0.70

---

## 💡 Usage

### Example

```python
import pickle
import numpy as np

# Load trained model
with open("flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input must match training pipeline's feature order and scaling
input_features = np.array([[...]]).reshape(1, -1)

# Predict
predicted_price = model.predict(input_features)[0]
print(f"Estimated Flight Price: ₹{round(predicted_price)}")
```

**Input:**  
A 2D numpy array of shape **(1, 47)** with encoded and scaled values

**Output:**  
A single float: predicted flight fare in INR

**Note:**  
This model requires that inputs are already:
- One-hot encoded (for categorical features)
- Scaled (e.g., via RobustScaler)
- Include engineered time features (sin/cos of hours, days, months)

---

## 🛠️ System

**Standalone model**  
Can be embedded in:
- Streamlit frontend  
- Flask API backend  
- Jupyter/Kaggle Notebook  

Requires input pre-processing pipeline (as used during training)  
**Dependencies:** numpy, sklearn, pandas

---

## ⚙️ Implementation Details

**Hardware & Software**
- Trained on: Google Colab (free tier)
- Framework: Python 3.10 + scikit-learn
- Runtime: ~2 seconds to train
- Inference time: < 50ms per prediction

---

## 🧠 Model Characteristics

- Trained from scratch (no pre-trained weights)
- Not quantized or pruned
- Total features: 47 (after encoding, scaling, and engineering)
- Uses cyclical time encoding (sin/cos) for hours, days, months
- Scaled using RobustScaler for better performance on outliers

---

## 📊 Data Overview

**Source:**
- Dataset: Data_Train.xlsx (public Kaggle dataset)
- Records: 10,681 flight listings
- Columns: Airline, Source, Destination, Date of Journey, Duration, Stops, Additional Info, and Price

**Preprocessing Summary:**
- Extracted day, month, dep_hour, arr_hour, duration components
- Applied cyclic encoding for time-based fields
- One-hot encoded Airline, Source, Destination, Additional_Info
- Scaled continuous features using RobustScaler

---

## 📈 Evaluation Results

- **Train/Test split:** 70/30
- **MAE:** ₹1758.39
- **RMSE:** ₹2537.03
- **R² Score:** 0.6968

**Evaluation Remarks:**
- Good generalization across airlines and cities
- Slightly less accurate on rare or underrepresented routes

---

## 🔍 Fairness & Limitations

### Fairness:
- Dataset does not contain personal or demographic attributes
- No demographic fairness evaluation required

### Known Limitations:
- Cannot adapt to real-time price surges or airline dynamic pricing
- May underperform for rare or unseen route-airline combinations
- Model is linear — complex nonlinearities in airline pricing aren't captured

### Ethics:
- This model is for educational and demonstrative purposes only
- **Do not use for actual pricing decisions in production without rigorous validation**

---

## ✅ Summary

This model showcases a full machine learning pipeline including:
- Feature engineering
- One-hot encoding
- Cyclical time transformations
- Robust scaling
- Linear regression with proper evaluation

_Perfect for beginners building their first structured ML regression project!_
