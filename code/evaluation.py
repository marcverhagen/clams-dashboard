
from pathlib import Path
from git import Repo
import config
import utils


class Repository(utils.FileSystemNode):

    """Class to give access to data in the evaluation repository."""

    def __init__(self, directory: str):
        super().__init__(Path(directory))
        self.repo = Repo(directory)
        self.load()

    def __getitem__(self, index: int):
        return self.evaluations[index]

    def load(self):
        self.readme = Path(self.path / 'README.md').open().read()
        self.evaluations_idx = { p.stem: Evaluation(p) for p in self.eval_directories() }
        self.evaluations = sorted(self.evaluations_idx.values())
        self.evaluation_names = sorted(self.evaluations_idx.keys())
        self.branches = { str(branch): branch for branch in self.repo.branches }
        self.branch_names = [ str(branch) for branch in self.repo.branches ]

    def eval_directories(self) -> list:
        """Returns the Paths of all evaluation directories."""
        return [ p for p in self.path.iterdir() if p.name.endswith('eval') and p.is_dir()]

    def evaluation(self, name: str) -> 'Evaluation':
        return self.evaluations_idx.get(name)

    def checkout(self, branch: str):
        self.branches[branch].checkout()
        self.load()



class Evaluation(utils.FileSystemNode):

    def __init__(self, path: Path):
        super().__init__(path)
        self.readme_file = Path(path / 'README.md')
        self.readme = utils.read_file(self.readme_file)
        self.scripts = list(path.glob('*.py'))
        self.predictions_idx ={}
        self.reports_idx = {}
        for p in self.path.iterdir():
            if p.is_dir() and p.name.startswith('preds@'):
                self.predictions_idx[p.name] = PredictionBatch(p)
            elif p.is_file() and p.name.startswith('report-'):
                self.reports_idx[p.name] = Report(p)
        self.predictions = sorted(self.predictions_idx.values())
        self.prediction_names = sorted(self.predictions_idx.keys())
        self.reports = sorted(self.reports_idx.values())

    def __str__(self):
        return f'<{self.__class__.__name__} "{self.path.stem}">'

    def __lt__(self, other):
        return self.path.stem < other.path.stem

    def __eq__(self, other):
        return self.path.stem == other.path.stem

    def prediction(self, name: str):
        return self.predictions_idx.get(name)

    def report(self,name: str):
        return self.reports_idx.get(name)

    def info(self):
        return (
            f'{self}\n\n'
            + f'code files {[f.stem for f in self.scripts]}')


class PredictionBatch(utils.FileSystemNode):

    """Class to wrap a directory with predictions. Each prediction is a MMIF file
    with processing results."""

    def __init__(self, path: Path):
        super().__init__(path)
        if len(self.name.split('@')) < 3:
            print("WARNING - missing component in name:", self.path)
        self.prediction_name = '@'.join(self.name.split('@')[1:-1])
        self.prediction_batch = self.name.split('@')[-1]
        self.readme = None
        self.files = {}
        readme_path = self.path / 'README.md'
        if readme_path.is_file():
            self.readme = utils.read_file(readme_path)
        for p in self.path.iterdir():
            if p.is_file() and p.name.endswith('.mmif'):
                self.files[p.name] = p
        
    def __str__(self):
        return f'<{self.__class__.__name__} {self.prediction_name} {self.prediction_batch}>'

    def __len__(self):
        return len(self.files)

    def file_names(self):
        return sorted(self.files)


class Report(utils.FileSystemNode):

    def __init__(self, path: Path):
        super().__init__(path)
        self.report_tool = self.name.split('@')[-2]
        self.report_batch = self.path.stem.split('@')[-1]
        self.content = utils.read_file(self.path)



if __name__ == '__main__':

    repo = Repository(config.EVALUATIONS)
    print(repo)

    for evaluation in repo.evaluations:
        print(f'\n -- {evaluation}')
        for prediction in evaluation.predictions:
            print('   ', prediction)
