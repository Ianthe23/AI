import pandas as pd
import csv
import os

def load_data(filename):
    if not os.path.isfile(filename):
        print(f"File {filename} not found.")
        return None
    return pd.read_csv(filename)

# Load data
filename = 'surveyDataSience.csv'
df = load_data(filename)

if df is not None:
    # 1. Number of respondents
    num_respondents = len(df)
    print(f"Number of respondents: {num_respondents}")

    # 2. Number and type of information (columns)
    num_columns = len(df.columns)
    columns = df.columns.tolist()
    print(f"Number of informations: {num_columns}")
    print(f"Type of informations: {columns}")

    # 3. Number of respondents with complete data (no missing values)
    complete_data = df.dropna()
    num_complete_respondents = len(complete_data)
    print(f"Number of respondents who have complete data: {num_complete_respondents}")

    # 4. Calculate the average years of higher education for respondents with complete data
    # Assuming 'years_of_studies' is the column for the years of studies in the dataset
    # You may need to adjust the column name depending on your CSV file structure

    # Convert years of studies to integer if it's in string format
    complete_data['years_of_studies'] = complete_data['years_of_studies'].astype(int)

    avg_studies_years = complete_data['years_of_studies'].mean()
    print(f"Average years of superior studies (complete data): {avg_studies_years}")

    # 5. Average years of higher education for respondents from Romania
    # Assuming 'country' is the column for country and 'Romania' is a valid value
    romania_data = complete_data[complete_data['country'] == 'Romania']
    avg_studies_years_romania = romania_data['years_of_studies'].mean()
    print(f"Average years of superior studies (Romania): {avg_studies_years_romania}")

    # 6. Average years of higher education for female respondents from Romania
    # Assuming 'gender' is the column for gender and 'female' is a valid value
    romania_female_data = romania_data[romania_data['gender'] == 'female']
    avg_studies_years_romania_female = romania_female_data['years_of_studies'].mean()
    print(f"Average years of superior studies (Romania, Female): {avg_studies_years_romania_female}")

    # 7. Compare the results
    print("\nComparison of the results:")
    print(f"Complete Data: {avg_studies_years}")
    print(f"Romania: {avg_studies_years_romania}")
    print(f"Romania (Female): {avg_studies_years_romania_female}")
