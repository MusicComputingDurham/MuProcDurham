#  Copyright (c) 2025 Robert Lieck.
from unittest import TestCase
import os
import importlib.util
import pathlib


class TestNotebooks(TestCase):

    cwd = None

    def setUp(self) -> None:
        self.cwd = os.getcwd()
        examples_dir = pathlib.Path(__file__).parent.resolve() / "notebooks"
        assert os.path.isdir(examples_dir), f"example directory '{examples_dir}' does not exist"
        os.chdir(examples_dir)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.cwd = None

    def test_examples(self):
        """Run all the files in notebooks folder"""
        for dir_path, dir_names, file_names in os.walk("."):
            for file in file_names:
                full_path = os.path.join(dir_path, file)
                if not file.endswith(".py"):
                    continue
                print(f"loading: {file}")
                # sourcing directly hides problematic lines in error traces
                # exec(open(os.path.join(dir_path, file)).read())
                # import as module instead
                spec = importlib.util.spec_from_file_location("", full_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
