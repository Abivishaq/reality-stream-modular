import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp

def plot_fips_histogram(df: pd.DataFrame, fips_column: str = "Fips", bins: int = 20):
    """
    Plots a histogram of FIPS codes from the given DataFrame.

    Args:
        df (pd.DataFrame): The dataset containing FIPS codes.
        fips_column (str): The name of the column with FIPS codes.
        bins (int): Number of histogram bins.
    """
    if fips_column not in df.columns:
        raise ValueError(f"Column '{fips_column}' not found in DataFrame.")

    # Convert to numeric and drop NaNs
    fips_data = pd.to_numeric(df[fips_column], errors='coerce').dropna()

    # Plot
    plt.figure(figsize=(8, 5))
    fips_data.plot(kind='hist', bins=bins, title='FIPS Code Distribution')
    plt.xlabel('FIPS Code')
    plt.ylabel('Frequency')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def basic_info(df):
    print("\nData Overview")
    print(df.head())
    print("\nShape of the dataset:", df.shape)
    print("\nColumn Information:")
    print(df.info())
    print("\nDescriptive Statistics:")

    if isinstance(df, cudf.DataFrame):
        print(df.describe())  # no transpose for cudf
    else:
        print(df.describe().T)  # transpose for pandas

    print("\nNull Values:")
    print(df.isnull().sum())
    print("\nNumber of duplicate rows:", df.duplicated().sum())


def find_duplicates(self, X_gpu, aligned_df):
    """
    Identify and display duplicate rows in the GPU feature dataframe,
    and return the corresponding rows from the original merged dataframe.

    Args:
        X_gpu (cudf.DataFrame): GPU feature dataframe.
        aligned_df (pd.DataFrame): Merged CPU dataframe containing features and target.

    Returns:
        pd.DataFrame: Subset of aligned_df corresponding to duplicate feature rows.
    """
    # Find duplicates in GPU dataframe
    duplicates = X_gpu.duplicated(keep="first")
    duplicates_cpu = duplicates.to_pandas()

    # Filter aligned_df using duplicate mask
    aligned_df_duplicates = aligned_df[duplicates_cpu]

    # Display results
    print(f"Number of duplicate rows found: {aligned_df_duplicates.shape[0]}")
    if not aligned_df_duplicates.empty:
        print(aligned_df_duplicates.head())

    return aligned_df_duplicates

def missing_values_distribution(df):
    """
    Plots distribution of missing values across features.
    Works for both pandas and cuDF DataFrames.
    """
    missing_ratios = df.isnull().mean() * 100

    # If GPU (cuDF), convert to pandas Series
    if str(type(missing_ratios)).startswith("<class 'cudf"):
        missing_ratios = missing_ratios.to_pandas()

    # Now plotting
    plt.figure(figsize=(10, 6))
    missing_ratios.hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Missing Value Percentages Across All Features')
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Number of Features')
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.boxplot(missing_ratios, vert=False, patch_artist=True,
                flierprops={'marker': 'o', 'color': 'red', 'markersize': 5})
    plt.title('Boxplot of Missing Value Percentages')
    plt.xlabel('Percentage of Missing Values')
    plt.yticks([])
    plt.show()




def plot_histograms_and_test_normality(df, column_indices):
    results = pd.DataFrame(columns=['Column', 'Shapiro_Statistic', 'Shapiro_p-value'])

    for column in df.columns[column_indices]:
        data = df[column].dropna()

        # If cuDF, convert to pandas
        if str(type(data)).startswith("<class 'cudf"):
            data = data.to_pandas()

        # Force conversion to numeric (important)
        data = pd.to_numeric(data, errors='coerce')
        data = data.dropna()  # Final cleaning

        if len(data) < 3:
            print(f"Skipping column {column} due to insufficient valid data.")
            continue

        # Create histogram plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, alpha=0.75, color='blue')
        plt.title(f'Histogram of {column}')
        plt.xlabel('Data Points')
        plt.ylabel('Frequency')

        # Perform Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # QQ plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'QQ Plot of {column}')
        plt.tight_layout()
        plt.show()

        results = pd.concat([results, pd.DataFrame({
            'Column': [column],
            'Shapiro_Statistic': [shapiro_stat],
            'Shapiro_p-value': [shapiro_p]
        })], ignore_index=True)

    return results

def plot_correlation_heatmap(dataframe, column_prefix):
    columns_to_analyze = [col for col in dataframe.columns if not col.startswith(column_prefix)]

    # Ensure the correlation matrix is computed using pandas
    corr_matrix = dataframe[columns_to_analyze].to_pandas().corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_correlation_heatmap(dataframe, column_prefix):
    columns_to_analyze = [col for col in dataframe.columns if not col.startswith(column_prefix)]

    # Ensure the correlation matrix is computed using pandas
    corr_matrix = dataframe[columns_to_analyze].to_pandas().corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_correlation_heatmap_v2(dataframe, column_prefix, target_series=None, target_name='target'):
    columns_to_analyze = [col for col in dataframe.columns if not col.startswith(column_prefix)]

    if target_series is not None:
        if len(target_series) == len(dataframe):
            dataframe = dataframe.copy()
            dataframe[target_name] = target_series
            columns_to_analyze.append(target_name)
        else:
            raise ValueError("The length of target_series and dataframe must match.")

    corr_matrix = dataframe[columns_to_analyze].to_pandas().corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Heatmap')
    plt.show()

def target_variable_analysis(df):
    if isinstance(df, cp.ndarray):
        df = pd.Series(cp.asnumpy(df))

    print("\nTarget Variable Analysis")
    print("Data Type:", df.dtype)
    print("Unique Values:", df.nunique())
    print("Value Counts:")
    print(df.value_counts())

    if df.nunique() < 20:
        df.value_counts().plot(kind='bar', color='orange', figsize=(10, 6))
        plt.title('Target Variable Distribution (Categorical)')
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.show()

def plot_class_distribution_before_after_smote(y_before, y_after):
    before_counts = pd.Series(cp.asnumpy(y_before)).value_counts().sort_index()
    after_counts = pd.Series(cp.asnumpy(y_after)).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(before_counts.index.astype(str), before_counts.values, color='salmon')
    axes[0].set_title("Class Distribution Before SMOTE")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(before_counts.values):
        axes[0].text(i, v + 2, str(v), ha='center')

    axes[1].bar(after_counts.index.astype(str), after_counts.values, color='seagreen')
    axes[1].set_title("Class Distribution After SMOTE")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(after_counts.values):
        axes[1].text(i, v + 2, str(v), ha='center')

    plt.tight_layout()
    plt.show()