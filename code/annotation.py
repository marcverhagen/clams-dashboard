
import re
import json
from io import StringIO
from pathlib import Path
from collections import namedtuple

from git import Repo

import utils
import config


Comparison = namedtuple('Comparison', ['in_both', 'in_first', 'in_second'])


class Repository(utils.FileSystemNode):

    """Class to give access to data in the annotation repository."""

    def __init__(self, directory: str):
        super().__init__(Path(directory))
        self.repo = Repo(directory)
        self.load()

    def __str__(self):
        return f'<{self.__class__.__name__} "{self.path.name}">'

    def load(self):
        """Load repository data."""
        self.readme = Path(self.path / 'README.md').open().read()
        batch_files = [p for p in (self.path / 'batches').iterdir()]
        self.batches_idx = { p.stem: Batch(p) for p in batch_files }
        self.batches = sorted(self.batches_idx.values())
        self.batch_names = sorted(self.batches_idx.keys())
        self.tasks_idx = { p.stem: Task(self, p) for p in self.task_directories() }
        self.tasks = sorted(self.tasks_idx.values())
        self.task_names = list([task.name for task in self.tasks])
        self.branches = { str(branch): branch for branch in self.repo.branches }
        self.branch_names = [ str(branch) for branch in self.repo.branches ]
   
    def task_directories(self):
        # TODO: now depends on there being a golds sub directory, should perhaps
        # instead check presence of readme and process.py files.
        return [ p for p in self.path.iterdir() if Path(p / 'golds').is_dir()]

    def task(self, task: str):
        return self.tasks_idx[task]

    def batch(self, name: str):
        return self.batches_idx[name]

    def checkout(self, branch: str):
        self.branches[branch].checkout()
        self.load()

    def pp(self):
        print(f'\n{self}')
        print(f'\nActive branch:\n    {self.repo.active_branch}')
        print('\nBatches:')
        for batch in self.batches:
            print('   ', batch)
        print('\nTasks:')
        for task in self.tasks:
            print('   ', task)
        print()


class Batch(utils.FileSystemNode):

    """A Batch is created from a single file in the batches subdirectory of the
    annotations repository. It includes a list of files referred to in the batch
    as well as the full batch file content. It also seprates out the batch-level
    comment at the top of the batch file."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.files = []
        with open(path) as fh:
            lines = fh.readlines()
            self.files = [l.strip() for l in lines if not l.strip().startswith('#')]
            self.content = ''.join(lines)
        self._comment = None

    def __len__(self):
        return len(self.files)

    @property
    def comment(self):
        if self._comment is None:
            comment = StringIO()
            separator_count = 0
            for line in self.content.split('\n'):
                if '-' * 50 in line:
                    separator_count += 1
                    if separator_count == 2:
                        break
                    else:
                        continue
                if not line.startswith('#'):
                    break
                line = line.lstrip('#').strip()
                line = '\n' if not line else line
                comment.write(f'{line}\n')
            self._comment = comment.getvalue()
            #print(f'COMMENT:\n{self._comment}<<<')
        return self._comment

    def pp(self):
        print(f'\n{b}\n\n{b.content[:500]}\n')


class Task(utils.FileSystemNode):

    def __init__ (self, rep: Repository, path: Path):
        super().__init__(path)
        self.path = path
        self._gold_directory = Path(path / 'golds')
        self._gold_files = None
        self.readme_file = Path(path / 'readme.md')
        self.readme = utils.read_file(self.readme_file)
        self.process_file = Path(path / 'process.py')
        self.process = self.process_content()
        self.data_drops = {}
        for subdir in self.path.iterdir():
            if subdir.is_dir() and re.match(r'\d{6}', subdir.name):
                self.data_drops[subdir.name] = DataDrop(subdir)

    def __str__(self):
        return f'<Task "{self.path}">'

    def __len__(self):
        return len(self.gold_files)

    @property
    def gold_directory(self):
        return self._gold_directory
    
    @property
    def gold_files(self):
        if self._gold_files is None:
            self._gold_files = []
            if self._gold_directory.is_dir():
                for path in sorted(self._gold_directory.iterdir()):
                    if path.is_file():
                        self._gold_files.append(path)
                    else:
                        for subpath in sorted(path.iterdir()):
                            self._gold_files.append(subpath)
        return self._gold_files
    
    def process_content(self):
        if self.process_file.is_file():
            with self.process_file.open() as fh:
                return fh.read()
        return ''

    def gold_file_ids(self):
        return [f.stem for f in self.gold_files]

    def data_drop(self, data_drop: str):
        return self.data_drops.get(data_drop)

    def gold_content(self, gold_file):
        if gold_file is None:
            return ''
        gold_path = Path(self._gold_directory / gold_file)
        with open(gold_path) as fh:
            return fh.read()

    def compare_to_batch(self, batch: Batch):
        gold_files = set(self.gold_file_ids())
        batch_files = set(batch.files)
        return Comparison(
            len(gold_files.intersection(batch_files)),
            len(gold_files.difference(batch_files)),
            len(batch_files.difference(gold_files)))


class DataDrop(utils.FileSystemNode):

    def __init__(self, path: Path):
        super().__init__(path)
        self.files = list([f for f in self.path.iterdir()])
        self.file_names = [f.name for f in self.files]

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return f'<{self.__class__.__name__} {self.name} files={len(self)}>'

    def file_content(self, filename: str):
        path = Path(self.path / filename)
        if path.suffix == '.json':
            return json.dumps(json.load(path.open()), indent=2)
        with path.open() as fh:
            return fh.read()


def test_print_gold_files():
    task_names =  ('scene-recognition', 'newshour-chyron')
    for task in [repo.task(name) for name in task_names]:
        print(f'\n{task}\n')
        for p in task.gold_files:
            print(p)



if __name__ == '__main__':

    repo = Repository(config.ANNOTATIONS)
    repo.pp()

    task = repo.task('scene-recognition')
    print(task)
