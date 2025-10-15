from pathlib import Path
import re

HEADING_RE = re.compile(r'^\s*#{1,6}\s+(.*\S)\s*$')


def print_line(case, i, lines, shorten=None):
    if shorten is not None:
        l = lines[i][:shorten]
    else:
        l = lines[i]
    print(f"{case} (l{i}): {l}")


def notebook_to_blank(source_file_path, target_file_path, verbose=0, shorten=None):
    """Convert a complete notebook into one with blanks"""

    # Regular expressions to detect the start and end markers.
    # They match a line starting with optional whitespace followed by a '#' and then at least three of the letter 'v' or '^', respectively
    start_marker_re = re.compile(r'^(\s*"\s*)#.*v{3,}.*\\n",')
    end_marker_re = re.compile(r'^(\s*"\s*)#.*\^{3,}.*')

    with open(source_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        start_match = start_marker_re.match(lines[i])
        if start_match:
            if verbose > 1:
                print_line("START", i, lines, shorten=shorten)
            start_line = i
            # Output the start marker line as-is.
            new_lines.append(lines[i])
            # Use the indentation from the marker line.
            indent = start_match.group(1)
            i += 1
            # Skip any lines until we find the end marker.
            # We replace the entire block with one line (if any code is present).
            replaced = False
            while i < len(lines):
                if end_marker_re.match(lines[i]):
                    if verbose > 1:
                        print_line("END", i, lines, shorten=shorten)
                    # Insert our replacement line if we haven't already.
                    if not replaced:
                        new_lines.append(indent + '# Put your code here!\\n",\n')
                        replaced = True
                    # Output the end marker line.
                    new_lines.append(lines[i])
                    i += 1
                    break
                else:
                    if verbose > 1:
                        print_line("SKIP", i, lines, shorten=shorten)
                    # Skip this line (it is part of the code to be replaced)
                    i += 1
            else:
                # In case the file ends without an end marker, insert the replacement.
                raise RuntimeError(f"Could not find matching end marker for region starting in line {start_line}")
        else:
            if verbose > 2:
                print_line("KEEP", i, lines, shorten=shorten)
            new_lines.append(lines[i])
            i += 1

    # Write the modified content
    with open(target_file_path, 'w') as f:
        f.writelines(new_lines)
    if verbose > 0:
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
        # if nb.name != "Practical_Working_with_Sampled_Audio.ipynb": continue  # pick single NB
        # only create BLANKS for practical notebooks and ignore existing BLANKS
        if nb.stem.startswith("Practical_") and not nb.stem.endswith("BLANKS"):
            print(f"Creating BLANKS version for: {nb.name}")
            notebook_to_blank(
                nb, nb_blanks_dir / f"{nb.stem}_BLANKS.ipynb",
                # verbose=3, shorten=100,  # for debugging
            )


if __name__ == "__main__":
    main()
