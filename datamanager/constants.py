# STEP: Get Dictionaries for states and industries

# TO DO: Try including DC and US Territories
STATE_DICT = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
    # US Territories
    "AS": "American Samoa", "GU": "Guam", "MP": "Northern Mariana Islands", "PR": "Puerto Rico", "VI": "U.S. Virgin Islands"
}

STATE_DICT_DELETE = {
    "AL": "ALABAMA","AK": "ALASKA","AZ": "ARIZONA","AR": "ARKANSAS","CA": "CALIFORNIA","CO": "COLORADO","CT": "CONNECTICUT","DE": "DELAWARE","FL": "FLORIDA","GA": "GEORGIA","HI": "HAWAII","ID": "IDAHO","IL": "ILLINOIS","IN": "INDIANA","IA": "IOWA","KS": "KANSAS","KY": "KENTUCKY","LA": "LOUISIANA","ME": "MAINE","MD": "MARYLAND","MA": "MASSACHUSETTS","MI": "MICHIGAN","MN": "MINNESOTA","MS": "MISSISSIPPI","MO": "MISSOURI","MT": "MONTANA","NE": "NEBRASKA","NV": "NEVADA","NH": "NEW HAMPSHIRE","NJ": "NEW JERSEY","NM": "NEW MEXICO","NY": "NEW YORK","NC": "NORTH CAROLINA","ND": "NORTH DAKOTA","OH": "OHIO","OK": "OKLAHOMA","OR": "OREGON","PA": "PENNSYLVANIA","RI": "RHODE ISLAND","SC": "SOUTH CAROLINA","SD": "SOUTH DAKOTA","TN": "TENNESSEE","TX": "TEXAS","UT": "UTAH","VT": "VERMONT","VA": "VIRGINIA","WA": "WASHINGTON","WV": "WEST VIRGINIA","WI": "WISCONSIN","WY": "WYOMING"
}

def _load_industries_dict():
    country = "US"
    naics_level = 2
    industries_csv_file = f"https://raw.githubusercontent.com/ModelEarth/community-data/master/{country.lower()}/id_lists/naics{naics_level}.csv"
    # Attempt to load the industries DataFrame from URL
    try:
        industries_df = pd.read_csv(
            f"https://raw.githubusercontent.com/ModelEarth/community-data/master/{country.lower()}/id_lists/naics{naics_level}.csv",
            header=None
        )
        INDUSTRIES_DICT = industries_df.set_index(0).to_dict()[1]
        print("Successfully loaded industries_df from URL.")
        return(INDUSTRIES_DICT)
    except Exception as e:
        print(f"Failed to load industries_df from URL due to error: {e}")
        # Try loading from the local file path as a fallback
        try:
            industries_df = pd.read_csv(industries_csv_file, header=None, names=['Industry_Code', 'Industry_Name'])
            INDUSTRIES_DICT = industries_df.set_index('Industry_Code').to_dict()['Industry_Name']
            print("Successfully loaded industries_df from local file.")
            return INDUSTRIES_DICT
        except FileNotFoundError:
            print(f"Error: The file {industries_csv_file} does not exist.")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
        except pd.errors.ParserError:
            print("Error: There was a parsing error while reading the CSV file.")
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")

    return {}

            
INDUSTRIES_DICT = _load_industries_dict()