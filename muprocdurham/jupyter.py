def no_linebreaks():
    from IPython.core.display import display, HTML
    display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
