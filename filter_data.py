import pandas as pd

# Load the merged dataset
merged_data = pd.read_csv('merged_final.csv')

# Standardize column names to ensure they match
merged_data.columns = merged_data.columns.str.strip().str.lower()

# Convert 'intime' and 'outtime' to datetime format
merged_data['intime'] = pd.to_datetime(merged_data['intime'])
merged_data['outtime'] = pd.to_datetime(merged_data['outtime'])

# Calculate the time difference between 'outtime' and 'intime' in hours
merged_data['icu_stay_hours'] = (merged_data['outtime'] - merged_data['intime']).dt.total_seconds() / 3600

# Round the 'icu_stay_hours' to the nearest integer
merged_data['icu_stay_hours'] = merged_data['icu_stay_hours'].round().astype(int)

# Update 'ventilation_status_flag' to set entries with 'ventilation_status' as None or starting with 'Non' to 0
merged_data['ventilation_status_flag'] = merged_data['ventilation_status'].apply(lambda x: 0 if pd.isna(x) or str(x).startswith('Non') else 1)

# Display the updated dataset
print("Updated Merged Dataset with Rounded ICU Stay Hours:")
# print(merged_data[:2000])

# Save the updated dataset to a new file
merged_data.to_csv('updated_merged_final_rounded.csv', index=False)
# Filter the dataset for age > 18 and icu_stay_hours > 100
filtered_data = merged_data[(merged_data['age'] > 18) & (merged_data['icu_stay_hours'] > 72)]

# Get the count of filtered data
filtered_count = len(filtered_data)

# Display the result
print(f"Number of records where age > 18 and ICU stay > 72 hours: {filtered_count}")


# Get the count of filtered data where ventilation_status_flag is 1
ventilation_flag_1_count = len(filtered_data[filtered_data['ventilation_status_flag'] == 1])

# Get the count of filtered data where ventilation_status_flag is 0
ventilation_flag_0_count = len(filtered_data[filtered_data['ventilation_status_flag'] == 0])

# Display the results
print(f"Number of records with ventilation_status_flag = 1: {ventilation_flag_1_count}")
print(f"Number of records with ventilation_status_flag = 0: {ventilation_flag_0_count}")