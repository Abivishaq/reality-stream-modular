import os 
def check_directory(directory_path): # Check whether the given directory exists, if not, then create it
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully by check_directory.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print("Current working directory:", os.getcwd())
        print("View under the folder icon which is followed by 2 dots..")
        print(f"check_directory '{directory_path}' already exists.")
    return directory_path