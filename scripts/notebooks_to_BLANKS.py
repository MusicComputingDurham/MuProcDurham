from pathlib import Path
import re

HEADING_RE = re.compile(r'^\s*#{1,6}\s+(.*\S)\s*$')


def notebook_to_blank(source_file_path, target_file_path, verbose=False):
    """Convert a complete notebook into one with blanks"""

    # Regular expressions to detect the start and end markers.
    # They match a line starting with optional whitespace followed by a '#' and then at least three of the letter 'v' or '^', respectively
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
    if verbose:
        print(f"Converted '{source_file_path}' to '{target_file_path}' with blanks")


def main():
    # root and notebook dir
    root_dir = Path(__file__).parent.parent.absolute()
    nb_dir = root_dir / "notebooks"

    #  where BLANKS versions are created
    # nb_blanks_dir = root_dir / "doc" / "extra" / "notebooks_ipynb"  # directly in doc (combine with BLANKS_IN_DOC_ONLY = True)
    nb_blanks_dir = nb_dir                                          # in NB folder (combine with BLANKS_IN_DOC_ONLY = False)
    nb_blanks_dir.mkdir(exist_ok=True)  # make sure dir exists

    # create BLANKS versions
    for nb in sorted(nb_dir.glob("*.ipynb")):  # only look at notebook files
        # only create BLANKS for practical notebooks and ignore existing BLANKS
        if nb.stem.startswith("Practical_") and not nb.stem.endswith("BLANKS"):
            print(f"Creating BLANKS version for: {nb.name}")
            notebook_to_blank(nb, nb_blanks_dir / f"{nb.stem}_BLANKS.ipynb", verbose=False)


if __name__ == "__main__":
    main()
