import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

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

    print('-----------------------------------')


    # 1. Number of respondents
    # Eliminate the first row
    num_respondents = len(df) - 1
    print(f"Number of respondents: {num_respondents}")

    print("-----------------------------------")

    # 2. Number and type of information (columns)
    num_columns = len(df.columns)
    column_types = df.dtypes
    print(f"Number of columns: {num_columns}")
    print("Column types:")
    print(column_types)

    print("-----------------------------------")



    # 3. Number of respondents with complete data (no missing values)
    complete_data = df.dropna()
    num_complete_respondents = len(complete_data)
    print(f"Number of respondents with complete data: {num_complete_respondents}")

    print("-----------------------------------")


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

    print("-----------------------------------")

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

    print("-----------------------------------")

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

    print("-----------------------------------")

    # 5. Number of women respondents from Romania who have complete data
    # Filter for Romanian women
    mask = (df['Q3'] == "Romania") & (df['Q2'] == "Woman")

    # Check for complete data (no NaN values in the row)
    complete_data_mask = df.notna().all(axis=1)

    # Count the number of complete rows that match the condition
    number_women_complete_data = df[mask & complete_data_mask].shape[0]

    print(f"Number of women respondents from Romania with complete data: {number_women_complete_data}")

    print("-----------------------------------")


    # 6. Number of woman respondents from Romania who code in Pyhton
    # Filter for Romanian women
    mask = (df['Q3'] == "Romania") & (df['Q2'] == "Woman") & (df['Q7_Part_1'] == "Python")
    print(f"Number of woman reposndents from Romania who code in Python: {df[mask].shape[0]}")

    print("-----------------------------------")

    mask_all = (df['Q2'] == "Woman") & (df['Q7_Part_1'] == "Python")

    # Interval of ages
    ages_intervals = df.loc[mask_all, 'Q1']

    if not ages_intervals.empty:
        most_common_age_interval = ages_intervals.value_counts().idxmax()
        print(f"The most common age interval for women who code in Python is: {most_common_age_interval}")
    else:
        print("No data with such women")

    print("-----------------------------------")

    # 7. domeniul de valori posibile si valorile extreme pentru fiecare 
    # atribut/proprietate (feature). In cazul proprietatilor nenumerice, 
    # cate valori posibile are fiecare astfel de proprietate
    # Pentru datele numerice
    try:
        print(df.select_dtypes(include=['int64', 'float64']).describe())
    except Exception as e:
        print("Nu exista date numerice!!")
    print("-----------------------------------")

    # Pentru datele nenumerice
    print(df.select_dtypes(include=['object']).nunique())
    print("-----------------------------------")

    # 8. info about years in programming in number of years
    # get all the possible values for the Q6  which contains "years"
    def convert_years(value):
        if pd.isna(value):
            return None
        if 'years' in value:
            if '-' in value:
                start, end = map(int, value.replace(' years', '').split('-'))
                return (start + end) / 2
            elif '+' in value:
                return int(value.replace(' years', '').replace('+', '')) + 5
            elif '<' in value:
                return 0.5
        return None

    # Apply the function to the 'Q6' column to create a new 'years_in_programming' column
    df['years_in_programming'] = df['Q6'].apply(convert_years)

    # Print the updated DataFrame with the new 'years_in_programming' column
    print(df['years_in_programming'].unique())
    print("-----------------------------------")

    # Calculate moments of order 1 and 2 (min, max, mean, std, median)
    min_value = df['years_in_programming'].min()
    max_value = df['years_in_programming'].max()
    mean_value = df['years_in_programming'].mean()
    std_dev = df['years_in_programming'].std()
    median_value = df['years_in_programming'].median()

    # Print the moments (statistics)
    print(f"Minimum years in programming: {min_value}")
    print("-----------------------------------")
    print(f"Maximum years in programming: {max_value}")
    print("-----------------------------------")
    print(f"Mean years in programming: {mean_value}")
    print("-----------------------------------")
    print(f"Standard deviation of years in programming: {std_dev}")
    print("-----------------------------------")
    print(f"Median years in programming: {median_value}")
    print("-----------------------------------")

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





