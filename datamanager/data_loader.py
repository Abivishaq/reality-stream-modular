import os
import pandas as pd
def get_data():
    
    # Parameters - TO DO: to be deleted
    param = {
        "folder": "naics6-bees-counties",
        "features": {
            "data": "industries",
            "startyear": 2017,
            "endyear": 2021,
            "path": "https://raw.githubusercontent.com/ModelEarth/community-timelines/main/training/naics{naics}/US/counties/{year}/US-{state}-training-naics{naics}-counties-{year}.csv",
        },
        "targets": {
            "data": "bees",
            "path": "https://github.com/ModelEarth/bee-data/raw/main/targets/bees-targets-top-20-percent.csv",
        },
        "models": ["xgboost"],
    }
    # Was: "models": ["lr", "svc", "rfc", "rbf", "xgboost"]

    # TO DO
    # To be the default. All other params are from incoming yaml, then are edited in textbox inside this colab.
    paramDefaults = {
        "folder": "myfolder",
        "models": ["xgboost"]
    }
    # Paths and settings
    target_url = param["targets"]["path"]
    features_template = param["features"]["path"]

    # TO DO: Get from param instead - this value is optional
    naics = 6

    # TO DO: This value is optional
    years = range(param["features"]["startyear"], param["features"]["endyear"] + 1)

    # TO DO: Get states from param instead - this value is optional
    states = ["ME", "NY"]

    full_save_dir = "output/training"

    os.makedirs(full_save_dir, exist_ok=True)

    # Build feature file paths
    feature_files = []
    for state in states:
        for year in years:
            feature_files.append(features_template.format(naics=naics, year=year, state=state))

    print("Constructed Feature File Paths:")
    for feature_file in feature_files:
        print(feature_file)

    # Load feature datasets
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

    # Load target dataset
    try:
        target_df = pd.read_csv(target_url)
        print("Targets loaded successfully.")
    except Exception as e:
        raise FileNotFoundError(f"Error loading target file {target_url}: {e}")

    # Make Fips columns consistent
    features_df["Fips"] = features_df["Fips"].astype(str)
    target_df["Fips"] = target_df["Fips"].astype(str)

    # Filter features_df to only Fips present in target_df
    features_df = features_df[features_df["Fips"].isin(target_df["Fips"])]

    # Sort and merge
    features_df = features_df.sort_values(by="Fips")
    target_df = target_df.sort_values(by="Fips")

    aligned_df = pd.merge(features_df, target_df, on="Fips", how="inner")

    # Verify merged data
    print("\nMerged aligned_df shape:", aligned_df.shape)

    # Separate features and target
    X_total_cpu = aligned_df.drop(columns=["Target"])
    y_total_cpu = aligned_df["Target"]

    return X_total_cpu, y_total_cpu