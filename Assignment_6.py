import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


# Load cleaned data
df = pd.read_csv("cleaned_ebay_deals.csv")

# Drop rows with missing values in the key columns
df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])

# --- Plot: Histogram of Discount Percentage ---
plt.figure(figsize=(10, 6))
sns.histplot(df["discount_percentage"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Discount Percentage (Histogram)")
plt.xlabel("Discount Percentage")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_ebay_deals.csv")

# Step 1: Drop rows with missing values in required columns
df = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"])

# Step 2: Create discount_bin column
def assign_discount_bin(discount):
    if discount <= 10:
        return 'Low'
    elif discount <= 30:
        return 'Medium'
    else:
        return 'High'

df['discount_bin'] = df['discount_percentage'].apply(assign_discount_bin)

# Step 3: Count samples per bin
bin_counts = df['discount_bin'].value_counts()
print("Original counts per discount bin:\n", bin_counts)

# Step 4: Balance the dataset via random under-sampling
min_bin_size = bin_counts.min()

balanced_df = (
    df.groupby('discount_bin', group_keys=False)
    .apply(lambda x: x.sample(min_bin_size, random_state=42))
    .reset_index(drop=True)
)

# Step 5: Drop the bin column (we only needed it for balancing)
balanced_df = balanced_df.drop(columns=['discount_bin'])

X = balanced_df[["price", "original_price", "shipping"]]
y = balanced_df["discount_percentage"]

# Step 2: Preprocess shipping (categorical) using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["shipping"])
    ],
    remainder="passthrough"  # keep 'price' and 'original_price' as-is
)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Build pipeline and train the model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# Step 5: Generate predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Step 7: Display metrics
print("\n--- Regression Evaluation Metrics ---")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"MSE  (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Visualization: Actual vs Predicted Scatter Plot ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel("Actual Discount Percentage")
plt.ylabel("Predicted Discount Percentage")
plt.title("Actual vs Predicted Discount Percentage")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Visualization: Residual Plot ---
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="darkorange")
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Predicted Discount Percentage")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Part 6: Apply Model to Incomplete Data ---

# Step 1: Create a new version without discount_percentage
incomplete_df = df.drop(columns=['discount_percentage'])

# Step 2: Randomly select 20 products with essential columns
sample_df = incomplete_df[["title", "price", "original_price", "shipping"]].sample(n=20, random_state=42)

# Step 3: Predict discount_percentage using the trained model
X_new = sample_df[["price", "original_price", "shipping"]]
predicted_discounts = model.predict(X_new)

# Step 4: Create a result table
sample_df["Predicted Discount (%)"] = predicted_discounts.round(2)

# Reorder columns for display
result_table = sample_df[["title", "price", "original_price", "shipping", "Predicted Discount (%)"]]

# Display the result table
print("\nPredicted Discount Percentages for 20 Products:\n")
print(result_table.to_string(index=False))
