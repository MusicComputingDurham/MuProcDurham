from IPython.core.display import display, HTML


def no_linebreaks():
    display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))  # jupyter notebook
    display(HTML("<style>div.jp-OutputArea-output pre {white-space: pre;}</style>"))  # jupyter lab
