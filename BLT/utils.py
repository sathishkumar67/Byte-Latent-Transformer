from __future__ import annotations
import os
import shutil

def clear_directory(current_dir: str) -> None:
    """
    Clears all files and folders in the specified directory.

    Args:
        current_dir (str): Path to the directory to be cleared.
    """
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # remove folder and its contents
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")