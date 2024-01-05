import os
import zipfile

folder_path = 'static/artifacts'
zip_filename = 'artifacts_folder.zip'

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            arcname = os.path.relpath(file_path, folder_path)
            zipf.write(file_path, arcname)

zipf.close()
