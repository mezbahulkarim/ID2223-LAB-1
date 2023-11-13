import pandas as pd

# Read the CSV file
df = pd.read_csv('original_wine.csv')

# Separate data into two dataframes based on 'type'
white_df = df[df['type'] == 'white']
red_df = df[df['type'] == 'red']

# Save the separated dataframes to new CSV files
white_df.to_csv('white.csv', index=False)
red_df.to_csv('red.csv', index=False)

print("Separation completed. Check 'white.csv' and 'red.csv'.")
