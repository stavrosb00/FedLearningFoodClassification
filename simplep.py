import pandas as pd

# First dictionary
data1 = {
    'ID': [1, 2, 3],
    'SR': [3, 5, 7],
    'Acc1': [25, 30, 35],
}

# Second dictionary
data2 = {
    'ID': [1, 2, 3],
    'SR': [3, 5, 7],
    'Acc2': [35, 40, 45],
}

# Convert dictionaries to DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Save first DataFrame to CSV
df1.to_csv('data.csv', index=False)
# Load the CSV file back into a DataFrame
df_csv = pd.read_csv('data.csv')

# Merge df2 with the DataFrame loaded from CSV based on common columns 'ID' and 'SR'
df_merged = pd.merge(df_csv, df2, on=['ID', 'SR'], how='inner')

# df2.join('data.csv')
print(df1)
# print(df2)
print(df_merged)
# Load the CSV file back into a DataFrame
# df_merged = pd.read_csv('data.csv')

# # Merge the second DataFrame with the first DataFrame based on common columns 'ID' and 'Name'
# df_merged = pd.merge(df_merged, df2, on=['ID', 'Name'], how='left')

# # Save the merged DataFrame to a new CSV file
# df_merged.to_csv('merged_data.csv', index=False)

# # Display the merged DataFrame
# print(df_merged)
import numpy as np

# Create a NumPy array and scalar values
array_data = np.array([1, 2, 3, 4, 5])
scalar1 = 10
scalar2 = 20
scalar3 = 30

# Save the array and scalar values into a .npz file
np.savez("data.npz", array=array_data, scalar1=scalar1, scalar2=scalar2, scalar3=scalar3)

# Load the saved data from the .npz file
loaded_data = np.load("data.npz")

# Access the array and scalar values
loaded_array = loaded_data["array"]
loaded_scalar1 = loaded_data["scalar1"]
loaded_scalar2 = loaded_data["scalar2"]
loaded_scalar3 = loaded_data["scalar3"]

print("Loaded array:", loaded_array)
print("Loaded scalar1:", loaded_scalar1)
print("Loaded scalar2:", loaded_scalar2)
print("Loaded scalar3:", loaded_scalar3)