import pandas as pd

# Load the CSV files
age_data = pd.read_csv('age1.csv')
icu_data = pd.read_csv('icustays.csv')
main_data = pd.read_csv('status.csv')

# Standardize column names to ensure they match
age_data.columns = age_data.columns.str.strip().str.lower()
icu_data.columns = icu_data.columns.str.strip().str.lower()
main_data.columns = main_data.columns.str.strip().str.lower()

# Merge com_data with age_data using 'subject_id' as the key
merged_com_age = icu_data.merge(age_data, on='subject_id', how='inner')

# #
# # # Merge the result with main_data using 'stay_id' as the key
merged_final = icu_data.merge(merged_com_age , on='stay_id', how='inner')

# Display the merged dataset
print("Merged Final Data:")
print(len(merged_final))

# Save the merged dataset to a new file
merged_final.to_csv('merged_final.csv', index=False)
