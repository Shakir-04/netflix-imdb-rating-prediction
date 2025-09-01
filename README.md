# netflix-imdb-rating-prediction
My first machine learning project predicting IMDb ratings of Netflix shows and movies using datasets, regression models (Linear Regression &amp; Random Forest) with Python, Pandas, and Scikit-learn written on Jupyter Notebook 

This project applies **machine learning** to predict IMDb ratings of Netflix shows and movies using metadata such as genre, release year, number of votes, and availability across countries.  
The models were implemented in **Python** using **Pandas, Scikit-learn, Matplotlib, and Seaborn**.

---

## 📊 Dataset
- **Source:** [Netflix Titles on Kaggle](https://www.kaggle.com/shivamb/netflix-shows)  
- Includes metadata: `title`, `genre`, `release year`, `imdb score`, `number of votes`, `available countries`, `type (movie/TV show)`  
- Target variable: **IMDb Average Rating**  

---

## ⚙️ Methods
1. **Data preprocessing**  
   - Dropped non-predictive columns (`title`, `imdbId`)  
   - One-hot encoded `genre` and `available countries`  
   - Label encoded `type`  

2. **Models**  
   - **Linear Regression** (baseline)  
   - **Random Forest Regression** (non-linear, more robust)  

3. **Visualization**  
   - Scatter plots and heatmaps to identify feature correlations  
   - Explored relationships between release year, votes, and ratings  

---

## 📈 Results
- **Linear Regression**: MSE = 0.8875, R² = 0.2784  
- **Random Forest Regression**: MSE = 0.7992, R² = 0.4147  

👉 Random Forest performed better, capturing non-linear relationships and explaining ~41% of the variance in IMDb ratings.  

---

## 🔧 Tools & Libraries
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  

