import pandas as pd

# Read the CSV file
wine_file = 'wine.csv'  # Replace with the path to your CSV file
df = pd.read_csv(wine_file)

# Replace 'white' with 1 and 'red' with 0 in the 'type' column
df['type'] = df['type'].replace({'white': 1, 'red': 0})

# Print the modified DataFrame
print(df)

# Save the modified DataFrame back to a CSV file if needed
df.to_csv('wine_binary.csv', index=False)
