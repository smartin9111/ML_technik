import pandas as pd

# Example 0: Creating a pandas Series one dimensional array with int and string
panadasSeries = pd.Series([10, 30, 50, 70, 90, 'bana', 'narancs', 'barack'])
print(panadasSeries)

### Pandas DataFrame Creation Examples

# Example 1: Creating a DataFrame from a dictionary of lists
# Keys become column names. Lists must be of the same length.
data_dict = {
    'A': [10, 20, 30],
    'B': [42, 54, 60],
    'C': [71, 82, 93]
}
df_from_dict = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df_from_dict)

pdS = pd.Series([1, 5, 3])
print('This pds: ')
print(pdS)

# Example 2: Creating a DataFrame from a list of dictionaries
# Each dictionary in the list represents a row. Keys become column names.
data_list_of_dicts = [
    {'A': 5, 'B': 6, 'C': 7},
    {'A': 8, 'B': 9, 'C': 10},
    {'A': 20, 'B': 21, 'C': 13}
]
df_from_list_of_dicts = pd.DataFrame(data_list_of_dicts)
print("\nDataFrame from list of dictionaries:")
print(df_from_list_of_dicts)


# Example 3: Creating a DataFrame using a MultiIndex
# A MultiIndex allows you to have multiple levels of indexes in rows or columns
index = pd.MultiIndex.from_tuples([('a', 1), ('a', 6), ('b', 1), ('b', 2)], names=['first index', 'second index'])
df_multi_index = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index = index)
print("\nDataFrame with MultiIndex:")
print(df_multi_index)


index = pd.MultiIndex.from_tuples([
    ('Észak', 'Q1'),
    ('Észak', 'Q2'),
    ('Dél', 'Q1'),
    ('Dél', 'Q2')
], names=['Régió', 'Negyedév'])

data = pd.DataFrame({
    'Értékesítés': [100, 150, 200, 250],
    'Profit': [30, 45, 60, 75]
}, index=index)

print(data)



# Example 4: Creating a DataFrame from a list of lists with column names specified
# Each inner list is a row. Column names are provided separately.
data_list_of_lists = [
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9]
]
columns = ['A', 'B', 'C']
df_from_list_of_lists = pd.DataFrame(data_list_of_lists, columns=columns)
print("\nDataFrame from list of lists with column names:")
print(df_from_list_of_lists)

# Example 5: Creating an empty DataFrame with column names
# Useful when you need to create a DataFrame structure to be filled later.
empty_df = pd.DataFrame(columns=['A', 'B', 'C'])
print("\nEmpty DataFrame:")
print(empty_df)

# Load a sample dataset from Scikit-learn library
from sklearn import datasets
iris = datasets.load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)


print("\nIris DataFrame:")
print(df_iris.head())



# Elérhető adatkészletek listája
available_datasets = {
    'iris': datasets.load_iris
}

def get_dataset_names():
    return list(available_datasets.keys())

def load_dataset(name):
    if name in available_datasets:
        data = available_datasets[name]()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df
    else:
        raise ValueError("Dataset not found!")

dataset_names = get_dataset_names()
print("Available datasets:", dataset_names)

# load and column print
for name in dataset_names:
    df = load_dataset(name)
    print(f"\n{name.capitalize()} DataFrame:")
    print(df.columns)
    print(df.head())
df.describe() # Descriptive statistics for each column

df.head()

# @title petal length (cm)


# 1. loc használata
df_subset_loc = df.loc[:, ['sepal length (cm)', 'petal length (cm)']]
print("\nSubset using loc:")
print(df_subset_loc.head())

# 2. iloc használata
column_indices = [0, 2]  # 'sepal length (cm)' és 'petal length (cm)' pozíciói
df_subset_iloc = df.iloc[:, column_indices]
print("\nSubset using iloc:")
print(df_subset_iloc.head())


# 3. Oszlopok listájának változón keresztüli megadása
columns_to_select = ['sepal length (cm)', 'petal length (cm)']
df_subset_list = df[columns_to_select]
print("\nSubset using list of columns:")
print(df_subset_list.head())


df['species'] = iris.target

# Az adatok két részre osztása
df_part1 = df.iloc[:100]
df_part2 = df.iloc[100:]

# A két rész egyesítése a pd.concat segítségével
df_concatenated = pd.concat([df_part2, df_part1], ignore_index=True)

print("\nConcatenated DataFrame:")
print(df_concatenated.head())
print(df_concatenated.tail())

# reset index - dropping indexes from columns
concated = pd.concat([df.iloc[100:], df.iloc[:100]]).reset_index(drop=True)
concated


# Subsets - filtering: with particular logic
concated.loc[concated["sepal length (cm)"] < 5]

# Subsets - filtering: More than one logics are in brackets
filtered_df = concated.loc[(concated["sepal length (cm)"] < 6) & (concated["sepal width (cm)"] == 3.0)]
print("\nFiltered DataFrame:")
print(filtered_df)


df.iloc[50:100]

df.sort_values("sepal length (cm)").head(50)
print(df["sepal width (cm)"].median())
print(df["sepal width (cm)"].mean())
print(df["sepal width (cm)"].min())
print(df["sepal width (cm)"].max())
print(df["sepal width (cm)"].sum())

