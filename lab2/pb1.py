import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Adjust display settings to show full row
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Increase display width
    pd.set_option('display.max_colwidth', None)  # Prevent text truncation

    
    # Print row 1
    print(df.iloc[0].T)


    # 1. Number of respondents
    num_respondents = len(df)
    print(f"Number of respondents: {num_respondents}")

    # 2. Number and type of information (columns)
    num_columns = len(df.columns)
    column_types = df.dtypes
    print(f"Number of columns: {num_columns}")
    print("Column types:")
    print(column_types)


    # 3. Number of respondents with complete data (no missing values)
    complete_data = df.dropna()
    num_complete_respondents = len(complete_data)
    print(f"Number of respondents with complete data: {num_complete_respondents}")

    # 4. Calculate the average years of higher education for respondents with complete data
    # Assuming 'years_of_studies' is the column for the years of studies in the dataset
    # You may need to adjust the column name depending on your CSV file structure
    average_years = 0
    for index, row in df.iloc[1:].iterrows():
        sum = 0
        if row['Q4'] == "Bachelor's degree":
            sum += 3
        elif row['Q4'] == "Master's degree":
            sum += 5
        elif row['Q4'] == "Doctoral degree":
            sum += 8
        average_years += sum
    average_years = average_years / num_respondents
    print(f"Average years of higher education for respondents: {average_years}")

    average_years_Romania = 0
    for index, row in df.iloc[1:].iterrows():
        sum = 0
        if row['Q4'] == "Bachelor's degree" and row['Q3'] == "Romania":
            sum += 3
        elif row['Q4'] == "Master's degree" and row['Q3'] == "Romania":
            sum += 5
        elif row['Q4'] == "Doctoral degree" and row['Q3'] == "Romania":
            sum += 8
        average_years_Romania += sum
    average_years_Romania = average_years_Romania / num_respondents
    print(f"Average years of higher education for respondents from Romania: {average_years_Romania}")

    average_years_Romania_women = 0
    for index, row in df.iloc[1:].iterrows():
        sum = 0
        if row['Q4'] == "Bachelor's degree" and row['Q3'] == "Romania" and row['Q2'] == "Woman":
            sum += 3
        elif row['Q4'] == "Master's degree" and row['Q3'] == "Romania" and row['Q2'] == "Woman":
            sum += 5
        elif row['Q4'] == "Doctoral degree" and row['Q3'] == "Romania" and row['Q2'] == "Woman":
            sum += 8
        average_years_Romania_women += sum
    average_years_Romania_women = average_years_Romania_women / num_respondents
    print(f"Average years of higher education for Romanian women is: {average_years_Romania_women}")

    # 5. Number of women respondents from Romania who have complete data
    # Filter for Romanian women
    mask = (df['Q3'] == "Romania") & (df['Q2'] == "Woman")

    # Check for complete data (no NaN values in the row)
    complete_data_mask = df.notna().all(axis=1)

    # Count the number of complete rows that match the condition
    number_women_complete_data = df[mask & complete_data_mask].shape[0]

    print(f"Number of women respondents from Romania with complete data: {number_women_complete_data}")


    # 6. Number of woman respondents from Romania who code in Pyhton
    # Filter for Romanian women
    mask = (df['Q3'] == "Romania") & (df['Q2'] == "Woman") & (df['Q7_Part_1'] == "Python")
    print(f"Number of woman reposndents from Romania who code in Python: {df[mask].shape[0]}")

    mask_all = (df['Q2'] == "Woman") & (df['Q7_Part_1'] == "Python")

    # Interval of ages
    ages_intervals = df.loc[mask_all, 'Q1']

    if not ages_intervals.empty:
        most_common_age_interval = ages_intervals.value_counts().idxmax()
        print(f"The most common age interval for women who code in Python is: {most_common_age_interval}")
    else:
        print("No data with such women")

    # 7. domeniul de valori posibile si valorile extreme pentru fiecare 
    # atribut/proprietate (feature). In cazul proprietatilor nenumerice, 
    # cate valori posibile are fiecare astfel de proprietate
    print(df.columns)
    print(df.dtypes)
    print(df.nunique()[:10])  # Primele 10 coloane
    print(df.describe())
    print(df.min(numeric_only=True))
    print(df.max(numeric_only=True))

    # Pentru datele nenumerice
    print(df.select_dtypes(include=['object']).nunique())

    # 8. info about years in programming in number of years
    # get all the possible values for the Q6  which contains "years"
    df['years_in_programming'] = None

    # Iterate over the rows in the DataFrame
    for i in range(len(df)):
        if pd.isna(df['Q6'][i]):
            continue  # Skip NaN values
        
        # Check if the row contains "years"
        if 'years' in df['Q6'][i]:
            if '-' in df['Q6'][i]:
                start, end = df['Q6'][i].replace(' years', '').split('-')
                middle = (int(start) + int(end)) / 2
                df['years_in_programming'][i] = middle
            elif '+' in df['Q6'][i]:
                start = df['Q6'][i].replace(' years', '').replace('+', '')
                df['years_in_programming'][i] = int(start) + 5  # Adding 5 to represent an estimate
            elif '<' in df['Q6'][i]:
                df['years_in_programming'][i] = 0.5  # For < 1 year

    # Print the updated DataFrame with the new 'years_in_programming' column
    print(df['years_in_programming'].unique())

    # Calculate moments of order 1 and 2 (min, max, mean, std, median)
    min_value = df['years_in_programming'].min()
    max_value = df['years_in_programming'].max()
    mean_value = df['years_in_programming'].mean()
    std_dev = df['years_in_programming'].std()
    median_value = df['years_in_programming'].median()

    # Print the moments (statistics)
    print(f"Minimum years in programming: {min_value}")
    print(f"Maximum years in programming: {max_value}")
    print(f"Mean years in programming: {mean_value}")
    print(f"Standard deviation of years in programming: {std_dev}")
    print(f"Median years in programming: {median_value}")

    # Filter the data for respondents who program in Python
    python_programmers = df[df['Q7_Part_1'] == 'Python']

    # Plot the distribution of Python programmers by age categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=python_programmers, x='Q1', palette='viridis')
    plt.title('Distribution of Respondents Programming in Python by Age Categories')
    plt.xlabel('Age Category')
    plt.ylabel('Number of Respondents')
    plt.xticks(rotation=45)
    plt.show()

    # Filter the data for respondents from Romania and who program in Python
    python_programmers_romania = df[(df['Q3'] == 'Romania') & (df['Q7_Part_1'] == 'Python')]

    # Plot the distribution of Python programmers in Romania by age categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=python_programmers_romania, x='Q1', palette='viridis')
    plt.title('Distribution of Respondents from Romania Programming in Python by Age Categories')
    plt.xlabel('Age Category')
    plt.ylabel('Number of Respondents')
    plt.xticks(rotation=45)
    plt.show()

    # Filter the data for female respondents from Romania and who program in Python
    female_python_programmers_romania = df[(df['Q2'] == 'Woman') & (df['Q3'] == 'Romania') & (df['Q7_Part_1'] == 'Python')]

    # Plot the distribution of female Python programmers in Romania by age categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=female_python_programmers_romania, x='Q1', palette='viridis')
    plt.title('Distribution of Female Respondents from Romania Programming in Python by Age Categories')
    plt.xlabel('Age Category')
    plt.ylabel('Number of Respondents')
    plt.xticks(rotation=45)
    plt.show()

    # Create a boxplot to identify outliers in programming experience (years_in_programming)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='years_in_programming', palette='coolwarm')
    plt.title('Boxplot of Programming Experience (Years) to Identify Outliers')
    plt.xlabel('Years of Programming Experience')
    plt.show()





