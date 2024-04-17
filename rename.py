import os
import glob


folder_path = "test/"


new_name = "test"


files = glob.glob(folder_path + "*.jpeg")


for i, file_path in enumerate(files):
    file_name, file_ext = os.path.splitext(file_path)
    new_file_name = new_name + "." + str(i + 1) + file_ext
    os.rename(file_path, os.path.join(folder_path, new_file_name))
