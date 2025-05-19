# House Price Prediction using Linear Regression

This repository contains a Jupyter Notebook that demonstrates a complete pipeline for predicting house prices based on features like `Bedrooms` and `Area` using Linear Regression. The notebook includes both the **Normal Equation** and **Gradient Descent** methods to compute regression weights and evaluates them using **Mean Squared Error (MSE)**.

---

## 📊 Dataset

The dataset used (`data.csv`) includes columns:
- `Bedrooms`
- `Area` (in square feet)
- `Price` (target variable)

---

## 🚀 Workflow

1. **Importing Libraries**  
   Load required libraries like `numpy`, `pandas`, and `matplotlib`.

2. **Data Loading & Cleaning**
   - Load `data.csv` into a DataFrame.
   - Remove rows with `0` values in critical columns.
   - Check for and report missing values.

3. **Data Visualization**  
   Visualize the relationship between `Area` and `Price` using a scatter plot.

4. **Feature Scaling**  
   Apply **Min-Max Scaling** to `Area` and `Price` columns.

5. **Train-Test Split**  
   Use a custom `train_test_split` function to divide the data.

6. **Linear Regression using Normal Equation**
   - Add a bias column.
   - Compute weights using matrix algebra:  
     $$ \theta = (X^T X)^{-1} X^T y $$
   - Predict and calculate MSE.
   - Inverse transform scaled predictions back to original scale.

7. **Linear Regression using Gradient Descent**
   - Implement **Batch Gradient Descent** manually.
   - Update weights iteratively.
   - Predict on the test set and calculate MSE.

8. **Comparison**
   - Compare **MSE** of Normal Equation vs. Gradient Descent.
   - Discuss **accuracy** and **speed** of both methods.

---

## 📈 Results

The notebook prints:
- MSE from both methods.
- Sample predicted vs actual prices.
- Final learned weights (`theta`).

---

## ✅ Requirements

Make sure the following Python packages are installed:
- `numpy`
- `pandas`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy pandas matplotlib
