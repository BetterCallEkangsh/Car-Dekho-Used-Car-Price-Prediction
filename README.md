# 🚗 Car Dekho Used Car Price Prediction

Predicting the resale price of used cars listed on [CarDekho](https://www.cardekho.com/) using regression models, so sellers can get a data-driven price suggestion based on market conditions.

## 📌 Problem Statement

This dataset comprises used cars sold on CarDekho.com in India, along with key features of each car. The goal is to predict a car's selling price from its input features — helping a new seller price their car appropriately based on current market conditions.

## 📊 Data Collection

- Data was scraped from the CarDekho website.
- **15,411 rows** and **13 columns**, including car name, brand, model, vehicle age, kilometers driven, seller type, fuel type, transmission type, mileage, engine capacity, max power, seats, and selling price (target).

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

## 🔍 Project Workflow

1. **Data Cleaning** — removed duplicate records, checked for null values.
2. **EDA** — separated categorical/numerical features, visualized feature correlations with a heatmap.
3. **Preprocessing** — feature/target split, train-test split (80/20), `StandardScaler` on numerical features via `ColumnTransformer`.
4. **Model Building** — trained and compared multiple regressors:
   - Linear Regression
   - Lasso Regression
   - Ridge Regression
   - K-Neighbors Regressor
   - Decision Tree Regressor
   - Random Forest Regressor
5. **Hyperparameter Tuning** — `RandomizedSearchCV` on KNN and Random Forest.
6. **Evaluation** — MAE, RMSE, and R² Score on train/test sets.

## 📈 Results

| Model | Test R² Score |
|---|---|
| Linear Regression | 0.663 |
| Lasso | 0.663 |
| Ridge | 0.663 |
| Tuned KNN Regressor | **0.832** |
| Tuned Random Forest | **0.832** |

The tuned KNN and Random Forest models performed best, explaining ~83% of the variance in selling price.

## 📂 Project Structure

```
├── Datasets/
│   └── cardekho_imputated.csv
├── Car_Dekho_Price_Prediction.ipynb
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Usage
```bash
git clone https://github.com/<your-username>/car-dekho-price-prediction.git
cd car-dekho-price-prediction
jupyter notebook Car_Dekho_Price_Prediction.ipynb
```

## 🔮 Future Improvements
- Feature engineering on categorical columns (brand, model) with target/frequency encoding
- Try gradient boosting models (XGBoost, LightGBM, CatBoost)
- Deploy as a web app (Flask/Streamlit) for interactive price prediction
- Log/scale the target variable to handle price skewness

## 📝 License
This project is open source and available under the [MIT License](LICENSE).
