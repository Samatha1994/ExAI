import gdown
import os

def download_file_from_google_drive(file_url, output_folder, filename):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Temporary output path
    temp_output_path = os.path.join(output_folder, 'tempfile')

    # Download the file using gdown to a temporary file
    gdown.download(url=file_url, output=temp_output_path, quiet=False, fuzzy=True)

    # Final output path
    final_output_path = os.path.join(output_folder, filename)
    
    # Rename the file
    os.rename(temp_output_path, final_output_path)

    return final_output_path

