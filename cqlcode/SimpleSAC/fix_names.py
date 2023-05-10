import os # used to fix the names for the cql layers experiments, might not be useful later

def process_directory(path):
    # Iterate through the folders in the given path
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        # Check if the folder is a directory and if its name ends with "layer4", "layer6", or "layer8"
        if os.path.isdir(folder_path) and folder_name.endswith(("layer4", "layer6", "layer8")):
            for sub_folder_name in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder_name)
                if os.path.isdir(sub_folder_path):
                    tokens = sub_folder_name.split("_")
                    tokens = tokens[:-4] + tokens[-2:-1] + tokens[-4:-2] + tokens[-1:]
                    new_sub_folder_name = '_'.join(tokens)

                    new_sub_folder_path = os.path.join(folder_path, new_sub_folder_name)
                    os.rename(sub_folder_path, new_sub_folder_path)
                    print("rename:", sub_folder_name)
                    print("re  to:",new_sub_folder_name)

            # Divide the folder name into tokens by splitting it using "_"
            tokens = folder_name.split("_")
            tokens = tokens[:-3] + tokens[-1:] + tokens[-3:-1]
            new_folder_name = '_'.join(tokens)
            new_folder_path = os.path.join(path, new_folder_name)
            os.rename(folder_path, new_folder_path)
            print("REname:", folder_name)
            print("RE  to:", new_folder_name)

process_directory('/checkpoints')