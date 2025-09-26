"""
Convert all notebooks (files ending with .ipynb) in this folder to Python scripts, do some reformatting and move to
examples folder (overwriting existing files). This way, notebooks are available in the gallery (even though not with
perfect formatting) and are included in the tests.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import re


def convert_notebooks(examples=True, blanks=True):
    # Use the folder of the script as source folder
    source_folder = Path(__file__).parent.resolve()

    # Define destination folder for examples and ensure it exists
    destination_folder = source_folder / "../examples/"
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over files in the source folder
    for source_filename in os.listdir(source_folder):

        # convert notebooks to examples
        if examples and source_filename.endswith(".ipynb") and not source_filename.endswith("_BLANKS.ipynb") and (source_filename.startswith("Lecture_") or source_filename.startswith("Practical_")):
            try:
                new_filename = notebook_to_example(source_filename, source_folder)
            except FileNotFoundError:
                print(f"Failed to find example file generated from '{source_filename}'")
            else:
                move_file(source_folder / new_filename, destination_folder / new_filename)

        # convert practical notebooks to ones with blanks
        if blanks and source_filename.endswith(".ipynb") and not source_filename.endswith("_BLANKS.ipynb") and source_filename.startswith("Practical_"):
            blank_filename = source_filename.replace(".ipynb", "_BLANKS.ipynb")
            try:
                notebook_to_blank(source_folder / source_filename, source_folder / blank_filename)
            except RuntimeError as msg:
                print(f"Filed to convert file '{source_filename}' to version with blanks: {msg}")


def notebook_to_example(source_filename, source_folder):
    """Convert notebook to example for sphinx gallery"""

    # Convert the notebook to a Python script
    subprocess.run(["jupyter", "nbconvert", "--to", "script", str(source_folder / source_filename)], check=True)

    # Construct the generated Python script filename
    new_filename = source_filename.replace(".ipynb", ".py")
    source_file_path = source_folder / new_filename

    # post-process and move file
    if source_file_path.exists():
        post_process_example(source_file_path)
        return new_filename
    else:
        raise FileNotFoundError


def post_process_example(
        file_path,
        fix_header=True,  # this is required for inclusion in gallery
        replace_input_cells=True,  # this breaks up the code by making input cells visible
        replace_bold_comments=True,  # this replaces bold comments (lines with # **XXX***) with headings
):
    """Post process Python file generated from notebook"""

    with open(file_path, "r+") as f:
        lines = f.readlines()

        if fix_header:
            if len(lines) >= 2 and lines[0].startswith("#!") and lines[1].startswith("# coding: utf-8"):
                # Look for the first markdown header
                header = "Example\n======="
                for line in lines[2:]:
                    if line.startswith("# # "):
                        header = line[3:].strip() + "\n" + "=" * len(line[3:].strip())
                        break
                lines = [f"\n\"\"\"\n{header}\n\"\"\"\n"] + lines[2:]

        if replace_input_cells:
            # Replace In[XXX] pattern with # %%
            lines = [re.sub(r"# In\[\d+\]:", "# %%", line) for line in lines]

        if replace_bold_comments:
            # Replace lines starting with # **XXX** with formatted block
            lines = [re.sub(r"# \*\*(.+?)\*\*",
                            lambda m: f"# %%\n# {m.group(1)}\n# {'-' * len(m.group(1))}\n# \n#", line) for
                     line in lines]

        # re-write file
        f.seek(0)
        f.writelines(lines)
        f.truncate()


def move_file(source_file_path, target_file_path):
    """Move file, overwriting any existing target files"""
    if target_file_path.exists():
        target_file_path.unlink()
    shutil.move(str(source_file_path), str(target_file_path))
    print(f"Moved: {source_file_path} to {target_file_path}")


def notebook_to_blank(source_file_path, target_file_path):
    """Convert a complete notebook into one with blanks"""

    # Regular expressions to detect the start and end markers.
    # They match a line starting with optional whitespace followed by a '#' and then at least three of the letter 'v' or '^'
    start_marker_re = re.compile(r'^(\s*"\s*)#.*v{3,}\\n",')
    end_marker_re = re.compile(r'^(\s*"\s*)#.*\^{3,}')

    with open(source_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        start_match = start_marker_re.match(line)
        if start_match:
            start_line = i
            # Output the start marker line as-is.
            new_lines.append(line)
            # Use the indentation from the marker line.
            indent = start_match.group(1)
            i += 1
            # Skip any lines until we find the end marker.
            # We replace the entire block with one line (if any code is present).
            replaced = False
            while i < len(lines):
                if end_marker_re.match(lines[i]):
                    # Insert our replacement line if we haven't already.
                    if not replaced:
                        new_lines.append(indent + '# Put your code here!\\n",\n')
                        replaced = True
                    # Output the end marker line.
                    new_lines.append(lines[i])
                    i += 1
                    break
                else:
                    # Skip this line (it is part of the code to be replaced)
                    i += 1
            else:
                # In case the file ends without an end marker, insert the replacement.
                raise RuntimeError(f"Could not find matching end marker for region starting in line {start_line}")
        else:
            new_lines.append(line)
            i += 1

    # Write the modified content
    with open(target_file_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Converted '{source_file_path}' to '{target_file_path}' with blanks")


if __name__ == '__main__':
    convert_notebooks()
