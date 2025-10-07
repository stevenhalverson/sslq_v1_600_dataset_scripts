#!/usr/bin/env python3

"""
Utility script to renumber the 'id' column in a dataset CSV.

Used in the SSLQ (Synthetic Scenic Lore Image Quality) dataset pipeline
to maintain sequential IDs after row deletions or merges.

Usage:
  python id_iterate.py
"""
import pandas as pd

# Load CSV
df = pd.read_csv("[INSERT_PATH]")

# Reset ID column (assuming it's named 'id')
df['id'] = range(1, len(df) + 1)

# Save back
df.to_csv("metadata_fixed.csv", index=False)

print("✅ Renumbered IDs from 1 to", len(df))
