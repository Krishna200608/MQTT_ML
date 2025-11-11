#!/usr/bin/env python3
"""
zip_summaries.py
-----------------------------------
This script automatically finds all Slice_* folders
inside the current working directory (e.g., Bidirectional_Results/)
and creates a ZIP archive of only the 'Summary/' subfolder for each slice.
"""

import os
import zipfile

def zip_summary_folders(base_dir):
    # List all items in the base directory (e.g., Bidirectional_Results)
    items = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for item in items:
        if item.lower().startswith("slice_"):
            slice_path = os.path.join(base_dir, item)
            summary_path = os.path.join(slice_path, "Summary")

            if os.path.isdir(summary_path):
                zip_filename = f"{item}_Summary.zip"
                zip_path = os.path.join(base_dir, zip_filename)

                print(f"[+] Zipping {summary_path} → {zip_filename}")

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(summary_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            # store relative path inside zip
                            arcname = os.path.relpath(full_path, base_dir)
                            zipf.write(full_path, arcname)

                print(f"    ✔ Created {zip_filename} ({len(os.listdir(summary_path))} files)\n")
            else:
                print(f"[-] No Summary folder found in {item}, skipping.\n")

    print("✅ All slice summaries have been zipped successfully.\n")

if __name__ == "__main__":
    base_dir = os.getcwd()  # assume script is placed in Outputs/Bidirectional_Results/
    print(f"Working directory: {base_dir}")
    zip_summary_folders(base_dir)
