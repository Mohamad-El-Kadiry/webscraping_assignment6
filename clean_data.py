import pandas as pd
import re
# Load the dataset with all columns read as strings
df = pd.read_csv("ebay_tech_deals.csv", dtype=str)

# --- Step 1: Clean Price Columns ---
#this helps in encapsulation, convert_value handles the conversion of a single element,
# and clean_currency applies that logic to the entire Series.
def clean_currency(prices):
    def convert_value(x):
        if pd.isna(x):
            return x
        if isinstance(x, str):
            cleaned = re.sub(r'[^\d.-]', '', x.strip())
            if cleaned == '':
                return None
            return cleaned
        return x
    return prices.apply(convert_value)

df['price'] = clean_currency(df['price'])
df['original_price'] = clean_currency(df['original_price'])
df['original_price'] = df['original_price'].fillna(df['price'])
# --- Step 3: Clean Shipping Information ---

# Replace "N/A" or empty values in the shipping column with a default message
# Replace "N/A" or empty strings with the default message
df["shipping"] = df["shipping"].replace(["N/A", ""], "Shipping info unavailable")

# Replace any string that is only whitespace with the default message
df["shipping"] = df["shipping"].replace(r'^\s*$', 'Shipping info unavailable', regex=True)

# Ensure any NaN values are also replaced
df["shipping"] = df["shipping"].fillna("Shipping info unavailable")

# --- Step 4: Convert Price Columns to Numeric ---
# Convert the cleaned 'price' and 'original_price' columns to numeric (float)
# Use errors="coerce" to handle any non-numeric values safely (they'll become NaN)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")

# --- Step 5: Calculate Discount Percentage ---
df["discount_percentage"] = ((1 - (df["price"] / df["original_price"])) * 100).round(2)
df["discount_percentage"] = df["discount_percentage"].fillna(0)

# --- Step 6: Drop rows with missing price/original_price ---
df = df.dropna(subset=["price", "original_price"])

# --- Step 7: Save the Cleaned Data ---
df.to_csv("cleaned_ebay_deals.csv", index=False)
print("Data cleaning complete. Cleaned data saved as 'cleaned_ebay_deals.csv'.")
