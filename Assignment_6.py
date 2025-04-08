import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("cleaned_ebay_deals.csv")

# Drop rows with missing values in key columns we care about
df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])

# Plot to explore distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df["discount_percentage"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Discount Percentage")
plt.xlabel("Discount (%)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------
# Balance the dataset
# --------------------

# Add a column to categorize discount levels
def assign_discount_bin(discount):
    if discount <= 10:
        return 'Low'
    elif discount <= 30:
        return 'Medium'
    else:
        return 'High'

df['discount_bin'] = df['discount_percentage'].apply(assign_discount_bin)

# Check how many samples per bin
bin_counts = df['discount_bin'].value_counts()
print("Original counts per discount bin:\n", bin_counts)

# Under-sample each bin to match the smallest group
min_bin_size = bin_counts.min()
balanced_df = (
    df.groupby('discount_bin', group_keys=False)
      .apply(lambda x: x.sample(min_bin_size, random_state=42))
      .reset_index(drop=True)
)

# Drop the bin column since we don't need it anymore
balanced_df = balanced_df.drop(columns=['discount_bin'])

# --------------------
# Prepare features and target
# --------------------

X = balanced_df[["price", "original_price", "shipping"]]
y = balanced_df["discount_percentage"]

# Encode the 'shipping' column (categorical) using OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["shipping"])
    ],
    remainder="passthrough"
)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline with preprocessing + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# --------------------
# Evaluate the model
# --------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Evaluation Metrics ---")
print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
print(f"MSE  (Mean Squared Error):       {mse:.2f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
print(f"R² Score:                         {r2:.2f}")

# MAE (Mean Absolute Error): 9.74
# MAE represents the average absolute difference between the actual and predicted values.
# In this case, the model's predictions are off by approximately 9.74 percentage points on average.
# A lower MAE indicates better model performance, and here it shows a moderate level of accuracy.

# MSE (Mean Squared Error): 157.43
# MSE measures the average squared difference between the actual and predicted values.
# It penalizes larger errors more than MAE due to the squaring of differences.
# A lower MSE indicates better performance. Here, the relatively high value indicates that the model has significant errors, though it’s not exceptionally bad.

# RMSE (Root Mean Squared Error): 12.55
# RMSE is the square root of MSE, which brings the error metric back to the original unit (percent in our case).
# It is more sensitive to large errors than MAE. In this case, the RMSE of 12.55 suggests that our predictions are, on average, off by around 12.55 percentage points.
# RMSE provides a clearer sense of model performance when compared to MAE, especially in scenarios where larger errors are more critical.

# R² Score: 0.67
# The R² Score (Coefficient of Determination) measures how well the model explains the variance in the target variable (discount_percentage).
# It ranges from 0 to 1, where a higher value means a better fit. Here, an R² of 0.67 suggests that our model explains about 67% of the variance in the discount percentage.
# This is a decent value, indicating that the model is performing reasonably well but has room for improvement, especially with data imbalance and potential feature issues.




# --------------------
# Visualizations
# --------------------

# Scatter plot: actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.xlabel("Actual Discount (%)")
plt.ylabel("Predicted Discount (%)")
plt.title("Actual vs Predicted Discount")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual plot to check error patterns
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="darkorange")
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Predicted Discount (%)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------
# Simulate real-world usage on unseen data
# --------------------

# Pretend we scraped data that doesn't have discount info yet
incomplete_df = df.drop(columns=['discount_percentage'])

# Randomly pick 20 products to "predict"
sample_df = incomplete_df[["title", "price", "original_price", "shipping"]].sample(n=20, random_state=42)

# Predict discounts using the trained model
X_new = sample_df[["price", "original_price", "shipping"]]
predicted_discounts = model.predict(X_new)

# Add predictions to the sample
sample_df["Predicted Discount (%)"] = predicted_discounts.round(2)

# Reorder and display as a clean table
result_table = sample_df[["title", "price", "original_price", "shipping", "Predicted Discount (%)"]]

print("\nPredicted Discount Percentages for 20 Products:\n")
print(result_table.to_string(index=False))
