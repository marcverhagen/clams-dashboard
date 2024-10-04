import os
import json
from pathlib import Path
from random import choice
from string import ascii_uppercase

# import pandas as pd
# import streamlit as st



class FileSystemNode:

    """Abstract class which provides some intialization, sorting and other common
    functionality for path-like classes from the annotation and evaluation modules."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.stem = path.stem

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return f'<{self.__class__.__name__} "{self.name}">'


def st_list_files(component, header: str, file_names: list, cutoff: int = 5):
    """Display a list of file names in a Streamlit component, returns a selectbox
    or a list of radio buttons, depending on how long the list is."""
    file_names = list(file_names)
    if len(file_names) > cutoff:
        return component.selectbox(
            header, file_names, label_visibility='collapsed')
    else:
        return component.radio(
            header, file_names, label_visibility='collapsed', format_func=identity)


def st_list_files2(component, task, cutoff: int = 5):
    """Display a list of file names in a Streamlit component, returns a selectbox
    or a list of radio buttons, depending on how long the list is. The difference
    with st_list_files() is that less assumptions are made on the golds directory
    structure."""
    # TODO: having these two functions is confusing, clean this up
    amended_fnames = []
    # using this to determine what part of the path is needed
    dir_parts = task.gold_directory.parts
    for fname in task.gold_files:
        amended_fnames.append(os.sep.join(fname.parts[len(dir_parts):]))
    if len(amended_fnames) > cutoff:
        return component.selectbox(
            'file-list', amended_fnames, label_visibility='collapsed')
    else:
        return component.radio(
            'file-list', amended_fnames, label_visibility='collapsed', format_func=identity)


def st_display_file(component, path: Path):
    """Display the content of a file path to a Streamlit component."""
    content = read_file(path)
    if path.name.endswith('.mmif') or path.name.endswith('.json'):
        content = json.dumps(json.loads(content), indent=2)
    component.text(content)


def st_display_branch(component, ANNOTATIONS):
    """Display all available branches in a selectbox. Return the selectbox and the
    branches."""
    branch_names = ANNOTATIONS.branch_names
    index = get_index(branch_names, 'main')
    return component.selectbox('Branch in repository:', branch_names, index=index)


def read_file(filepath: Path):
    if filepath.is_file():
        with filepath.open() as fh:
            return fh.read()
    return ''


def identity(text: str):
    # TODO: I am not happy with this name
    return text.replace('@', ' ⟹ ')


def random_string(length: int = 5):
    return ''.join(choice(ascii_uppercase) for i in range(length))


def get_index(haystack: list, needle: str):
    try:
        return haystack.index(needle)
    except ValueError:
        return 0



# Style to supress printing the first column of a table
style = """
<style>
thead tr th:first-child {display:none}
tbody th {display:none}
</style>
"""
