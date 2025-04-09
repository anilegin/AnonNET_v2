import os
import shutil


def remove_contents(folder):
    """Remove all files and folders inside the given folder."""
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)  # Remove the file or link
            elif os.path.isdir(path):
                shutil.rmtree(path)  # Remove the directory and its contents
        except Exception as e:
            print(f"Failed to delete {path}. Reason: {e}")
            
            
            
if __name__ == "__main__":
    temp_driving_dir = os.path.join("/home/aegin/projects/anonymization/AnonNET/results", "temp_driving")
    temp_source_dir = os.path.join("/home/aegin/projects/anonymization/AnonNET/results", "temp_source")
    remove_contents("/home/aegin/projects/anonymization/AnonNET/source_images")
    remove_contents(temp_driving_dir)
    remove_contents(temp_source_dir)
    remove_contents("/home/aegin/projects/anonymization/AnonNET/anon")
    remove_contents("/home/aegin/projects/anonymization/AnonNET/.cache")