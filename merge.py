import pandas as pd

# Load the test data and predictions
df1 = pd.read_csv('/content/mars-private_test-class.csv')
df2 = pd.read_csv('/content/mars-predictions_ens_of_3.csv')

# Merge the two DataFrames
mdf = pd.concat([df1, df2], axis=1)

# Save the merged DataFrame to a CSV file
mdf.to_csv('merged_class.csv', index=False)
