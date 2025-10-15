import os

from pathlib import Path
import re
import shutil

import nbformat
from nbconvert import PythonExporter, HTMLExporter

HEADING_RE = re.compile(r'^\s*#{1,6}\s+(.*\S)\s*$')
BLANKS_IN_DOC_ONLY = False


def first_heading(nb):
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            m = HEADING_RE.match(line)
            if m:
                return m.group(1).strip()
    return None


def convert(nb_path, py_out=None, html_out=None):
    """Convert a single notebook to .py and .html"""
    nb = nbformat.read(nb_path, as_version=4)
    # ---- .py ----
    if py_out is not None:
        # convert NB to script
        py_code, _ = PythonExporter().from_notebook_node(nb)
        # inject import to fix/fake IPython magic in scripts
        py_code = py_code.splitlines()
        py_code = py_code[:2] + ["from tests.notebooks.util import get_ipython"] + py_code[2:]
        py_code = "\n".join(py_code)
        # write
        py_out.write_text(py_code, encoding="utf-8")
    # ---- .html ----
    if html_out is not None:
        html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
        html_out.write_text(html, encoding="utf-8")
    # ---- title ----
    title = first_heading(nb)
    if title is None:
        title = nb.stem
    return title


def create_rst(rst_out, title, stem, html_dir, ipynb_dir):
    rst = rf"""{title}
{'=' * len(title)}

 * `{stem}.html <../{html_dir}/{stem}.html>`_ (view html)
 * `{stem}.ipynb <../{ipynb_dir}/{stem}.ipynb>`_ (notebook)"""

    if stem.startswith("Practical_") and BLANKS_IN_DOC_ONLY:
        rst += rf"""
 * `{stem}_BLANKS.ipynb <../{ipynb_dir}/{stem}_BLANKS.ipynb>`_ (notebook with BLANKS)"""

    rst += rf"""
 * You can find any additional files (for all notebooks) `here <../assets/index.html>`_ or as a zip file here:
   `assets.zip <../assets.zip>`_

--------------------

.. only:: html

   .. raw:: html

      <iframe src="../{html_dir}/{stem}.html"
              style="width:100%; height:80vh; border:0;"
              loading="lazy"
              referrerpolicy="no-referrer">
      </iframe>"""
    rst_out.write_text(rst, encoding="utf-8")


def write_asset_index(out, files):
    html = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>File Listing</title>
</head>
<body>
  <h1>Files</h1>
  <ul>"""
    for f in files:
        html += fr"""
    <li><a href="{f}">{f}</a></li>
"""
    html += r"""
  </ul>
</body>
</html>"""
    out.write_text(html)


def main():
    root_dir = Path(__file__).parent.parent.absolute()
    nb_dir = root_dir / "notebooks"
    doc_dir = root_dir / "doc"
    doc_extra_dir = doc_dir / "extra"
    doc_nb_html_sub_dir = "notebooks_html"
    doc_nb_rst_sub_dir = "notebooks_rst"
    doc_nb_ipynb_sub_dir = "notebooks_ipynb"
    doc_nb_html_dir = doc_extra_dir / doc_nb_html_sub_dir
    doc_nb_ipynb_dir = doc_extra_dir / doc_nb_ipynb_sub_dir
    doc_nb_rst_dir = doc_dir / doc_nb_rst_sub_dir
    doc_assets_dir = doc_extra_dir / "assets"
    test_nb_dir = root_dir / "tests" / "notebooks"

    # converted NBs are written into these directories, make sure they exist
    for dir in [doc_nb_html_dir, doc_nb_rst_dir, doc_nb_ipynb_dir, test_nb_dir, doc_assets_dir]:
        dir.mkdir(exist_ok=True)

    print("Using the following paths:")
    print(f"  root dir:          {root_dir}")
    print(f"  NB dir:            {nb_dir}")
    print(f"  doc dir:           {doc_dir}")
    print(f"  doc assets dir:    {doc_assets_dir}")
    print(f"  doc NB HTML dir:   {doc_nb_html_dir}")
    print(f"  doc NB RST dir:    {doc_nb_rst_dir}")
    print(f"  doc NB ipynb dir:  {doc_nb_ipynb_dir}")
    print(f"  test NB dir:       {test_nb_dir}")

    # get files
    all_files = sorted(nb_dir.iterdir())
    nb_files = [f for f in all_files if
                str(f).endswith(".ipynb") and  # notebooks
                (f.name.startswith("Practical_") or f.name.startswith("Lecture_"))]  # only practicals and lectures
    other_files = [f for f in all_files if
                   not str(f).endswith(".ipynb") and  # anything other than notebooks
                   not str(f.name).startswith(".")]  # ignore hidden files/directories

    # read gallery header
    with open(doc_nb_rst_dir/"README.rst", 'r') as file:
        nb_gallery = file.read()
    toc_tree = []

    # iterate through notebooks
    print("Start notebook conversion:")
    for nb in nb_files:
        stem = nb.stem
        # account for BLANKS version
        if stem.endswith("BLANKS") and BLANKS_IN_DOC_ONLY:
            print(f"WARNING: {nb.name} seems to be a BLANKS version. "
                  f"I am IGNORING this notebook! "
                  f"These are created automatically for notebooks whose name starts with 'Practical_'."
                  f"To manually create them, put the files in {doc_nb_ipynb_dir} "
                  f"(CAUTION: they may be automatically overwritten!)")
            continue
        # copy to doc
        shutil.copyfile(nb, doc_nb_ipynb_dir / f"{stem}.ipynb")
        # not BLANKS in tests
        if stem.endswith("BLANKS"):
            py_out = None
        else:
            py_out = test_nb_dir / f"{stem}.py"
        # get title and convert to .py and .html
        title = convert(
            nb_path=nb,
            py_out=py_out,
            html_out=doc_nb_html_dir / f"{stem}.html",
        )
        #
        if stem.endswith("BLANKS") and not BLANKS_IN_DOC_ONLY:
            title += " (BLANKS)"
        # create RST file
        create_rst(rst_out=doc_nb_rst_dir / f"{stem}.rst",
                   title=title,
                   stem=stem,
                   html_dir=doc_nb_html_sub_dir,
                   ipynb_dir=doc_nb_ipynb_sub_dir)
        # add to gallery
        nb_gallery += f"\n * `{title} <{doc_nb_rst_sub_dir}/{stem}.html>`_"
        toc_tree.append(f"/{doc_nb_rst_sub_dir}/{stem}")
        print(f"  ✓ {nb.name}")
    print("DONE")

    # add toc tree
    print("Adding TOC tree")
    nb_gallery += "\n\n.. toctree::\n   :hidden:\n"
    for t in toc_tree:
        nb_gallery += "\n   " + t
    print("DONE")
    nb_gallery += "\n"

    # write gallery file
    (doc_dir / "notebook_gallery.rst").write_text(nb_gallery, encoding="utf-8")

    # copy non-ipynb files
    print("Copying assets:")
    asset_files = []
    for f in other_files:
        print(f"    {f.name}")
        asset_files.append(f.name)
        for dest in [doc_assets_dir / f.name, test_nb_dir / f.name]:
            shutil.copyfile(f, dest)
    # remove olde index.html if it exists
    index_html = doc_assets_dir / 'index.html'
    if os.path.exists(index_html):
        os.remove(index_html)
    # zip assets
    print("Zipping assets")
    shutil.make_archive(doc_extra_dir / "assets", "zip", doc_assets_dir)
    # create new index.html
    print("Writing assets index.html")
    write_asset_index(out=index_html, files=asset_files)
    print("DONE")


if __name__ == "__main__":
    main()
