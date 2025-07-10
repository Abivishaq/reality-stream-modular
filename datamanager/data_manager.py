# datamanager/data_manager.py

import os
import pandas as pd
import cudf
import cupy as cp
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import SMOTE

class DataManager:
    def __init__(self, dataset_name, target_column, save_dir, save_training=True):
        """
        Initializes the DataManager.

        Args:
            dataset_name (str): Name of the dataset.
            target_column (str): Name of the target column.
            save_dir (str): Directory where outputs should be saved.
            save_training (bool): Whether to save the processed data.
        """
        self.dataset_name = dataset_name
        self.target_column = target_column
        self.save_dir = save_dir
        self.save_training = save_training


    def check_directory(self, path):
        """Create the directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        return path

    def copy_integrated_csv(self):
        """
        Reads the processed CSV and copies it to the save directory.

        Returns:
            Optional[str]: Path to saved CSV file or None if not saved.
        """
        if not self.save_training:
            return None

        self.check_directory(self.save_dir)

        csv_file = f"../process/{self.dataset_name}/{self.target_column}-{self.dataset_name}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"Read file from: {csv_file}")

            output_path = os.path.join(self.save_dir, f"{self.target_column}-{self.dataset_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved file at: {output_path}")
            return output_path
        else:
            print(f"Warning: CSV file not found at {csv_file}. Please check the path.")
            return None

    def build_gpu_dataset(self, param, states=None, naics=None):
        """
        Builds and returns GPU-compatible datasets by downloading and merging features and targets.

        Args:
            param (dict): Configuration dictionary containing feature & target paths.
            states (list[str], optional): List of U.S. state abbreviations.
            naics (int, optional): NAICS industry code to format the feature path.

        Returns:
            tuple:
                - X_total (cudf.DataFrame): Feature data on GPU
                - y_total (cupy.ndarray): Target labels on GPU
        """
        # Extract config
        target_url = param["targets"]["path"]
        features_template = param["features"]["path"]

        # Defaults (if not provided)
        if naics is None:
            naics = 6  # fallback default
        if states is None:
            states = ["ME", "NY"]  # fallback default

        years = range(param["features"]["startyear"], param["features"]["endyear"] + 1)

        # Construct feature file paths
        feature_files = []
        for state in states:
            for year in years:
                feature_files.append(features_template.format(naics=naics, year=year, state=state))

        print("Constructed Feature File Paths:")
        for feature_file in feature_files:
            print(feature_file)

        # Load all feature files
        feature_dfs = []
        for feature_file in feature_files:
            try:
                feature_dfs.append(pd.read_csv(feature_file))
                print(f"Loaded feature file: {feature_file}")
            except Exception as e:
                print(f"Error loading feature file {feature_file}: {e}")

        if not feature_dfs:
            raise FileNotFoundError("No feature files could be loaded. Please check the paths and try again.")

        features_df = pd.concat(feature_dfs, ignore_index=True)

        # Load target CSV
        try:
            target_df = pd.read_csv(target_url)
            print("Targets loaded successfully.")
        except Exception as e:
            raise FileNotFoundError(f"Error loading target file {target_url}: {e}")

        # Align by FIPS
        features_df["Fips"] = features_df["Fips"].astype(str)
        target_df["Fips"] = target_df["Fips"].astype(str)

        # Filter and merge
        features_df = features_df[features_df["Fips"].isin(target_df["Fips"])]
        features_df = features_df.sort_values(by="Fips")
        target_df = target_df.sort_values(by="Fips")

        aligned_df = pd.merge(features_df, target_df, on="Fips", how="inner")
        print("\nMerged aligned_df shape:", aligned_df.shape)

        # Split into X and y
        X_total_cpu = aligned_df.drop(columns=["Target"])
        y_total_cpu = aligned_df["Target"]

        print("X_total_cpu shape:", X_total_cpu.shape)
        print("y_total_cpu shape:", y_total_cpu.shape)

        # Convert to GPU
        X_total = cudf.DataFrame.from_pandas(X_total_cpu)
        y_total = cp.asarray(y_total_cpu)

        print("Data converted to GPU format successfully.")
        print("X_total (GPU) rows:", len(X_total))
        print("y_total (GPU) rows:", len(y_total))

        return X_total, y_total
    
    def fill_na(self, df): ## looks redundant delete # just keeping here to ensure filling is not forgotten.
        """
        Fill missing values in the DataFrame with 0.

        Args:
            df (pd.DataFrame or cudf.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame or cudf.DataFrame: DataFrame with NA values filled with 0.
        """
        return df.fillna(0)
    
    def select_columns(self, dataframe, prefixes_to_exclude=None, name_to_exclude=None):
        # Filter columns based on exclusion prefixes
        columns_to_exclude = [col for col in dataframe.columns if any(col.startswith(prefix) for prefix in prefixes_to_exclude)]

        # Remove the specific column name if provided
        if name_to_exclude and name_to_exclude in dataframe.columns:
            columns_to_exclude.append(name_to_exclude)

        # Final columns to keep
        columns_to_keep = [col for col in dataframe.columns if col not in columns_to_exclude]

        return dataframe[columns_to_keep]
    
    def apply_log_transform(df, exclude_columns=None):
        transformed_df = df.copy()
        if exclude_columns is None:
            exclude_columns = []

        for column in transformed_df.columns:
            if pd.api.types.is_numeric_dtype(transformed_df[column]) and column not in exclude_columns:
                transformed_df[column] = np.log1p(transformed_df[column])
        return transformed_df


    def preprocess_data(dataframe, scale_type='standardize', include_target=False, target=None):
        if scale_type == 'standardize':
            scaler = StandardScaler()
        elif scale_type == 'normalize':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling type. Choose 'standardize' or 'normalize'.")

        # Convert to pandas for sklearn scalers
        if isinstance(dataframe, cudf.DataFrame):
            dataframe_pd = dataframe.to_pandas()
        else:
            dataframe_pd = dataframe

        if include_target and target in dataframe_pd.columns:
            features = dataframe_pd.drop(columns=[target])
            scaled_features = scaler.fit_transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
            scaled_df[target] = dataframe_pd[target].values
        else:
            scaled_features = scaler.fit_transform(dataframe_pd)
            scaled_df = pd.DataFrame(scaled_features, columns=dataframe_pd.columns)

        # Convert back to cuDF
        return cudf.DataFrame.from_pandas(scaled_df)


    def split_and_save_data(self, X, y, test_size=0.2, random_state=42, save=True):
        """
        Split data into training and test sets and optionally save to disk.

        Args:
            X (pd.DataFrame or cudf.DataFrame): Features.
            y (cp.ndarray or pd.Series): Targets.
            test_size (float): Fraction of data to use for testing.
            random_state (int): Seed for reproducibility.
            save (bool): Whether to save the split datasets as CSV files.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if save:
            X_train.to_pandas().to_csv(os.path.join(self.save_dir, "X_train.csv"), index=False)
            X_test.to_pandas().to_csv(os.path.join(self.save_dir, "X_test.csv"), index=False)
            pd.Series(cp.asnumpy(y_train)).to_csv(os.path.join(self.save_dir, "y_train.csv"), index=False)
            pd.Series(cp.asnumpy(y_test)).to_csv(os.path.join(self.save_dir, "y_test.csv"), index=False)
            print("Train-test split files saved successfully.")

        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train, fillna_value=0, random_state=42):
        """
        Apply SMOTE to oversample the minority class.

        Args:
            X_train (cudf.DataFrame): Feature training data.
            y_train (cp.ndarray): Target training data.
            fillna_value (int or float): Value to fill NaNs before SMOTE.
            random_state (int): Seed for SMOTE reproducibility.

        Returns:
            tuple: (X_train_resampled, y_train_resampled)
                X_train_resampled: cudf.DataFrame
                y_train_resampled: cp.ndarray
        """
        # Fill NaNs
        X_train_filled = X_train.fillna(fillna_value)

        # Convert to pandas/numpy
        X_train_pd = X_train_filled.to_pandas()
        y_train_np = cp.asnumpy(y_train)

        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled_pd, y_resampled_np = smote.fit_resample(X_train_pd, y_train_np)

        # Convert back to GPU format
        X_resampled = cudf.DataFrame.from_pandas(X_resampled_pd)
        y_resampled = cp.asarray(y_resampled_np)

        print("SMOTE applied successfully. Shapes after resampling:")
        print(X_resampled.shape, y_resampled.shape)

        return X_resampled, y_resampled