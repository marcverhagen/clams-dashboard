import os
import json
from io import StringIO
from pathlib import Path
from random import choice
from string import ascii_uppercase

from git import Repo


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


## Streamlit utilities

def st_list_files(component, header: str, file_names: list, cutoff: int = 5):
    """Display a list of file names in a Streamlit component, returns a selectbox
    or a list of radio buttons, depending on how long the list is."""
    file_names = list(file_names)
    if len(file_names) > cutoff:
        return component.selectbox(
            header, file_names, label_visibility='collapsed')
    else:
        return component.radio(
            header, file_names, label_visibility='collapsed', format_func=remove_at)


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
            'file-list', amended_fnames, label_visibility='collapsed', format_func=remove_at)


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


# Style to supress printing the first column of a table
style = """
<style>
thead tr th:first-child {display:none}
tbody th {display:none}
</style>
"""


## General utilities

def read_file(filepath: Path):
    """Read a text file and return the content if it exists, otherwise return an
    empty string."""
    if filepath.is_file():
        with filepath.open() as fh:
            return fh.read()
    return ''


def remove_at(text: str):
    """Replace the @ symbol in a string with an arrow."""
    return text.replace('@', ' ⟹ ')


def random_string(length: int = 5):
    return ''.join(choice(ascii_uppercase) for i in range(length))


def get_index(haystack: list, needle: str):
    try:
        return haystack.index(needle)
    except ValueError:
        return 0


## Repository utilities

class DirtyWorkingTreeWarning:

    def __init__(self, repo: Repo, diffs: list):
        self.commit = str(repo.head.commit)[:8]
        dirname = Path(repo.working_tree_dir).name
        self.message = f'some tracked files in repo "{dirname}"" were changed or deleted'
        self.diffs = diffs
        self.fatal = True

    def __str__(self):
        s = StringIO()
        s.write(f'WARNING:{self.message}')
        for diff in self.diffs:
            s.write(f'\n{diff.change_type} {diff_path_name(diff)}')
        return f'{s.getvalue()}'


class UntrackedFilesWarning:

    def __init__(self, repo: Repo):
        directory = Path(repo.working_tree_dir).name
        self.message = f'there are untracked files in repository "{directory}"'
        self.fnames = repo.untracked_files
        self.fatal = False

    def __str__(self):
        s = StringIO()
        s.write(f'WARNING:{self.message}')
        for fname in self.fnames:
            s.write(f'\n{fname}')
        return s.getvalue()


def check_repository(repo: Repo) -> list:
    """Check the repository to see whether tracked files were editted and whether
    there are untracked files. Return a list of warning instances."""
    # NOTE: could also use repo.is_dirty, but I want to make the distinction
    warnings = []
    diffs = repo.head.commit.diff(None)
    if diffs:
        warnings.append(DirtyWorkingTreeWarning(repo, diffs))
    if repo.untracked_files:
        warnings.append(UntrackedFilesWarning(repo))
    return warnings


def print_diff(diff):
    print(f'{diff.change_type} {diff_path_name(diff)}')


def diff_path_name(diff):
    if diff.a_blob:
        return diff.a_blob.path
    elif diff.b_blob:
        return diff.b_blob.path
    else:
        return None


