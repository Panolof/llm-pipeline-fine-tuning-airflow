# scripts/data_versioning.py

import subprocess
import os

def version_data():
    """
    Adds the processed data to DVC, commits the changes to Git, and pushes to remote storage.
    """
    # Ensure DVC is initialized
    if not os.path.exists('.dvc'):
        subprocess.run(['dvc', 'init'], check=True)
    
    # Add processed data to DVC
    subprocess.run(['dvc', 'add', 'data/processed/'], check=True)
    
    # Commit changes to Git
    subprocess.run(['git', 'add', 'data/processed.dvc', '.gitignore'], check=True)
    subprocess.run(['git', 'commit', '-m', 'Add processed data to DVC'], check=True)
    
    # Push data to remote storage
    subprocess.run(['dvc', 'push'], check=True)