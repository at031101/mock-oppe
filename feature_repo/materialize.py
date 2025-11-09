#!/usr/bin/env python3
"""
Apply Feast definitions in feature_repo and print feature views.
This script uses Feast Python API so it works with local FileSource parquet files.
"""
import os
from feast import FeatureStore
import sys

def main():
    repo_dir = os.path.dirname(__file__)
    print("Feature repo dir:", repo_dir)
    fs = FeatureStore(repo_path=repo_dir)
    print("Applying feature definitions (this will write the registry)...")
    try:
        # Importing features will register objects on module import in Feast's apply
        # To ensure the python file is discoverable, we call apply()
        fs.apply()
    except Exception as e:
        print("Warning: fs.apply() raised:", e)
    print("Available feature views:")
    try:
        fvs = fs.list_feature_views()
        for fv in fvs:
            print(" -", fv.name)
    except Exception as e:
        print("Could not list feature views:", e)
    # Because FileSource points directly to a parquet, there's no separate "materialize" time-window step.
    print("Materialization for file-based offline stores is a no-op (source parquet is the canonical data).")

if __name__ == '__main__':
    main()
