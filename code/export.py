"""

Export the annotation and evaluation repositories as a set of markdown pages.

The pages are intended to be uploaded as a GitHub pages site (which is now done
by hand, but it may in the future be done via GitHub actions).

$ python export.py 
$ python export.py --debug

The mardown pages are written to ../docs/www.

The code uses the local clones of the annotations and evaluations repositories.

Make sure that you check out the branches (typically the main branch) that you
want to be displayed. Running the script will tell you want branch is being used
and give warnings in two cases:

1- files in the committed branch differ from files in the working directory
2- there are untracked files

Never publish the site if you get the first warning. The second may be all right
if the untracked files do not contain any relevant information about annotations 
or evaluations.

"""

import io
import os
import sys
import pathlib

import config
import model
from utils import remove_at, read_file, check_repository


DEBUG = False


MODEL = model.Data(config.ANNOTATIONS, config.EVALUATIONS)
ANNOTATIONS = MODEL.annotations
EVALUATIONS = MODEL.evaluations

ANNOTATIONS_WARNINGS = check_repository(ANNOTATIONS.repo)
EVALUATIONS_WARNINGS = check_repository(EVALUATIONS.repo)

ANNOTATIONS_COMMIT = ANNOTATIONS.repo.head.commit.hexsha
EVALUATIONS_COMMIT = EVALUATIONS.repo.head.commit.hexsha
ANNOTATIONS_COMMIT_SHORT = ANNOTATIONS.repo.git.rev_parse(ANNOTATIONS_COMMIT, short=8)
EVALUATIONS_COMMIT_SHORT = EVALUATIONS.repo.git.rev_parse(EVALUATIONS_COMMIT, short=8)
ANNOTATIONS_TREE = f'{config.ANNOTATIONS_REPO_URL}/tree/{ANNOTATIONS_COMMIT}'
EVALUATIONS_TREE = f'{config.EVALUATIONS_REPO_URL}/tree/{EVALUATIONS_COMMIT}'
ANNOTATIONS_BRANCH = ANNOTATIONS.repo.head.reference
EVALUATIONS_BRANCH = EVALUATIONS.repo.head.reference

# Top-level directory structure
TASKS_DIR = 'tasks'
BATCHES_DIR = 'batches'
EVALUATIONS_DIR = 'evaluations'

# Mappings from pages in directories to pages higher up in the hierarchy.
BREADCRUMBS = {
    'batches':
        [('Dashboard', '../index.md'),
         ('Batches', 'index.md')],
    'batches/batch':
        [('Dashboard', '../../index.md'),
         ('Batches', '../index.md'),
         ('Batch', 'index.md')],
    'tasks':
        [('Dashboard', '../index.md'),
         ('Tasks', 'index.md')],
    'tasks/task':
        [('Dashboard', '../../index.md'),
         ('Tasks', '../index.md'),
         ('Task', 'index.md')],
    'tasks/task/drops':
        [('Dashboard', '../../../index.md'),
         ('Tasks', '../../index.md'),
         ('Task', '../index.md'),
         ('Drops', 'index.md')],
    'tasks/task/drops/drop':
        [('Dashboard', '../../../index.md'),
         ('Tasks', '../../index.md'),
         ('Task', '../index.md'),
         ('Drops', 'index.md'),
         ('Drop', None)],
    'evaluations':
        [('Dashboard', '../index.md'),
         ('Evaluations', 'index.md')],
    'evaluations/evaluation':
        [('Dashboard', '../../index.md'),
         ('Evaluations', '../index.md'),
         ('Evaluation', 'index.md')],
    'evaluations/evaluation/predictions':
        [('Dashboard', '../../../index.md'),
         ('Evaluations', '../../index.md'),
         ('Evaluation', '../index.md'),
         ('Predictions', 'index.md')],
    'evaluations/evaluation/reports':
        [('Dashboard', '../../../index.md'),
         ('Evaluations', '../../index.md'),
         ('Evaluation', '../index.md'),
         ('Reports', 'index.md')],
}

# Mappings from directories to subpages
SUBPAGES = {
    'batches/batch':
        [('batch', 'index.md'),
         ('files', 'files.md'),
         ('content', 'content.md'),
         ('tasks', 'tasks.md'),
         ('use in evaluation', 'evaluation.md')],
    'tasks/task':
        [('task', 'index.md'),
         ('readme', 'readme_file.md'),
         ('gold files', 'golds.md'),
         ('data drops', 'drops/index.md'),
         ('batches', 'batches.md'),
         ('script', 'script.md')],
    'evaluations/evaluation':
        [('evaluation', 'index.md'),
         ('readme', 'readme_file.md'),
         ('code', 'code.md'),
         ('predictions', 'predictions/index.md'),
         ('reports', 'reports/index.md')],
    'evaluations/evaluation/predictions':
        [('evaluation', '../index.md'),
         ('readme', '../readme_file.md'),
         ('code', '../code.md'),
         ('predictions', '../predictions/index.md'),
         ('reports', '../reports/index.md')],
}


def breadcrumbs(path: str, current_page: str = ''):
    crumbs = BREADCRUMBS.get(path, [])
    # TODO: could probably change this so that the last crumb is never a link
    # either in the code or in the BREADCRUMBS dictionary.
    return adjust_navigation_elements(crumbs, current_page)

def subpages(path: str, current_page: str = ''):
    pages = SUBPAGES.get(path, [])
    return adjust_navigation_elements(pages, current_page)

def adjust_navigation_elements(elements: list, current: str):
    # Removes the link if it is the current page
    return [((page, None) if page == current else (page, link))
            for page, link in elements]


def debug(text: str):
    if DEBUG:
        print(f'DEBUG: {text}')


class SiteBuilder():

    def __init__(self, directory: str):
        self.path = pathlib.Path(directory)
        self.info = [
            (ANNOTATIONS_COMMIT_SHORT, ANNOTATIONS_BRANCH, config.ANNOTATIONS),
            (EVALUATIONS_COMMIT_SHORT, EVALUATIONS_BRANCH, config.EVALUATIONS)]
        self.warnings = ANNOTATIONS_WARNINGS + EVALUATIONS_WARNINGS

    def build(self):
        self.print_info_and_warnings()
        self.create_directories()
        self.index()
        self.batches()
        self.tasks()
        self.evaluations()

    def print_info_and_warnings(self):
        """Write information and warnings to standard output."""
        for commit, branch, dirname in self.info:
            print(f'INFO: using commit {commit} of branch "{branch} '
                  f'of repository "{pathlib.Path(dirname).name}"')
        for warning in self.warnings:
            print(f'WARNING: {warning.message}')

    def create_directories(self):
        dirs = [TASKS_DIR, BATCHES_DIR, EVALUATIONS_DIR]
        for batch in ANNOTATIONS.batches:
            dirs.append(os.path.join(BATCHES_DIR, batch.stem))
        for task in ANNOTATIONS.tasks:
            dirs.append(os.path.join(TASKS_DIR, task.name))
            dirs.append(os.path.join(TASKS_DIR, task.name, 'drops'))
        for directory in dirs:
            full_path = self.path / directory
            full_path.mkdir(parents=True, exist_ok=True)
        for evaluation in EVALUATIONS:
            paths = [
                self.path / 'evaluations' / evaluation.name,
                self.path / 'evaluations' / evaluation.name / 'predictions',
                self.path / 'evaluations' / evaluation.name / 'reports']
            for path in paths:
                path.mkdir(parents=True, exist_ok=True)

    def index(self):
        # Building /index.md
        path = self.path / 'index.md'
        with PageBuilder(path) as pb:
            pb.header('CLAMS Dashboard')
            pb.p('CLAMS Dashboard for viewing annotations and evaluations. It is '
                 'a user-friendly interface to annotation tasks, annotation batches, '
                 'and evaluations, as well as the relations between them.')
            pb.p('Available viewers:')
            pb.write('- 🕵️‍♀️ [Annotation Batches](batches/index.md)\n')
            pb.write('- 🕵️‍♀️ [Annotation Tasks](tasks/index.md)\n')
            pb.write('- 🕵️‍♀️ [Evaluations](evaluations/index.md)\n\n')
            pb.p('The homepages for the repositories are available at:')
            pb.write(f'- 🏠 [{config.ANNOTATIONS_REPO_URL}]({config.ANNOTATIONS_REPO_URL})\n')
            pb.write(f'- 🏠 [{config.EVALUATIONS_REPO_URL}]({config.EVALUATIONS_REPO_URL})\n\n')
            pb.p('It is a good idea to look at the README.md files in those '
                 'repositories.')
            pb.p('Repository versions that were used for this dashboard:\n')
            pb.table_header('repository', 'commit', 'branch')
            for commit, branch, dirname in self.info:
                pb.table_row([pathlib.Path(dirname).name, commit, branch])
            pb.p('\nThe code for this dashboard is maintained at'
                 f' [{config.DASHBOARD_REPO_URL}]({config.DASHBOARD_REPO_URL}).')

    def batches(self):
        # Building /batches
        # Building /batches/{batch.stem}
        batches_path = self.path / 'batches'
        batches_index_path = batches_path / 'index.md'
        with PageBuilder(batches_index_path) as pb:
            pb.debug(f'batches()')
            pb.navigation_breadcrumbs(breadcrumbs('batches', 'Batches'))
            pb.header('Annotation Batches')
            pb.p(f'Annotation batches with number of files in each batch:')
            pb.table_header('batch', 'size', align='lr')
            for batch in ANNOTATIONS.batches:
                pb.table_row([f'[{batch.stem}]({batch.stem}/index.md)', len(batch)])
                self.batch(batch)

    def batch(self, batch):
        # Building /batches/{batch.stem}
        path = self.path / 'batches' / batch.stem
        self.batch_index(path / 'index.md', batch)
        self.batch_files(path / 'files.md', batch)
        self.batch_content(path / 'content.md', batch)
        self.batch_tasks(path / 'tasks.md', batch)
        self.batch_evaluation(path / 'evaluation.md', batch)

    def batch_index(self, path, batch):
        # Building /batches/{batch.stem}/index.md
        with PageBuilder(path) as pb:
            pb.debug(f'batch_index()')
            pb.navigation_breadcrumbs(breadcrumbs('batches/batch', 'Batch'))
            pb.header('Annotation Batch', batch.stem)
            pb.navigation_subpages(subpages('batches/batch', 'batch'))
            pb.p(f'Batch **{batch.stem}** has {len(batch.files)} files.')
            pb.subheader(f'Batch comment')
            pb.write(batch.comment)

    def batch_files(self, path, batch):
        # Building /batches/{batch.stem}/files.md
        with PageBuilder(path) as pb:
            pb.debug(f'batch_files()')
            pb.navigation_breadcrumbs(breadcrumbs('batches/batch', 'Batch'))
            pb.header('Annotation Batch', batch.stem)
            pb.navigation_subpages(subpages('batches/batch', 'files'))
            pb.subheader('File identifiers in batch')
            for fileid in batch.files:
                pb.write(f'1. {fileid}\n')

    def batch_content(self, path, batch):
        # Building /batches/{batch.stem}/content.md
        with PageBuilder(path) as pb:
            pb.debug(f'batch_content()')
            pb.navigation_breadcrumbs(breadcrumbs('batches/batch', 'Batch'))
            pb.header('Annotation Batch', batch.stem)
            pb.navigation_subpages(subpages('batches/batch', 'files'))
            pb.subheader('Batch file content')
            pb.pre(batch.content.strip())

    def batch_tasks(self, path, batch):
        # Building /batches/{batch.stem}/tasks.md
        with PageBuilder(path) as pb:
            pb.debug(f'batch_tasks()')
            pb.navigation_breadcrumbs(breadcrumbs('batches/batch', 'Batch'))
            pb.header('Annotation Batch', batch.stem)
            pb.navigation_subpages(subpages('batches/batch', 'tasks'))
            pb.subheader('Batch usage by annotation task')
            pb.p('This shows how many GUIDs from this batch were used in '
                 'annotation tasks.')
            pb.table_header('task name', 'overlap', 'task size', align='lrr')
            for task in ANNOTATIONS.tasks:
                # TODO: create links to tasks
                comparison = task.compare_to_batch(batch)
                url = f'../../tasks/{task.name}/index.md'
                pb.table_row([f'[{task.name}]({url})', comparison.in_both, len(task)])

    def batch_evaluation(self, path, batch):
        # Building /batches/{batch.stem}/evaluation.md
        with PageBuilder(path) as pb:
            pb.debug(f'batch_evaluation()')
            pb.navigation_breadcrumbs(breadcrumbs('batches/batch', 'Batch'))
            pb.header('Annotation Batch', batch.stem)
            pb.navigation_subpages(subpages('batches/batch', 'use in evaluation'))
            pb.subheader('Batch usage in evaluation repository')
            batch_usage1 = MODEL.batch_usage_in_system_predictions(batch.stem)
            batch_usage2 = MODEL.batch_usage_in_system_reports(batch.stem)
            pb.p('Usage in system predictions:')
            if batch_usage1:
                pb.table_header('evaluation', 'system prediction')
                pb.table_rows(batch_usage1)
            else:
                pb.write('*None*\n\n')
            pb.p('Usage in system reports:')
            if batch_usage2:
                pb.table_header('evaluation', 'system report')
                pb.table_rows(batch_usage2)
            else:
                pb.write('*None*\n\n')

    def tasks(self):
        # Building /tasks/index.md
        # Building /tasks/{task.name}/
        path = self.path / 'tasks' / 'index.md'
        with PageBuilder(path) as pb:
            pb.debug('tasks()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks', 'Tasks'))
            pb.header('Annotation Tasks')
            pb.p(f'Annotation tasks with number of gold files in each task:')
            pb.table_header('task', 'size', align="lr")
            for task in ANNOTATIONS.tasks:
                pb.table_row([f'[{task.name}]({task.name}/index.md)', len(task)])
                self.task(task)

    def task(self, task):
        # Building /tasks/{task.name}/
        path = self.path / 'tasks' / task.name
        self.task_index(path / 'index.md', task)
        self.task_readme(path / 'readme_file.md', task)
        self.task_gold(path / 'golds.md', task)
        self.task_drops(path / 'drops', task, task.data_drops)
        self.task_batches(path / 'batches.md', task)
        self.task_script(path / 'script.md', task)

    def task_index(self, path, task):
        # Building /tasks/{task.name}/index.md
        with PageBuilder(path) as pb:
            pb.debug(f'annotations_task_index()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task', 'Task'))
            pb.header('Annotation Task', task.name)
            pb.navigation_subpages(subpages('tasks/task', 'task'))
            pb.p(
                f'Task {task.name} has {len(task.data_drops)} data drops'
                f' and {len(task)} gold files.')

    def task_readme(self, path, task):
        # Building /tasks/{task.name}/readme_file.md
        with PageBuilder(path) as pb:
            pb.debug(f'task_readme()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task', 'Task'))
            pb.header('Annotation Task', task.name, 'Readme')
            pb.navigation_subpages(subpages('tasks/task', 'readme'))
            # NOTE: if we just give a link to the README at GitHub we can use
            # /tree/{self.commit_sha}/{task.name}/readme.md
            pb.write(task.readme)
    
    def task_gold(self, path , task):
        # Building /tasks/{task.name}/golds.md
        with PageBuilder(path) as pb:
            pb.debug(f'task_gold()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task', 'Task'))
            pb.header('Annotation Task', task.name, 'Gold files')
            pb.navigation_subpages(subpages('tasks/task', 'gold files'))
            pb.p(f'This page lists the gold standard files in this task. '
                 f'The total number of files is {len(task.gold_files)}, '
                 f'this includes files from all data drops. Clicking '
                 f'a link takes you to the data file in the repository.')
            url = f'{ANNOTATIONS_TREE}/{task.name}/golds'
            for gf in task.gold_files:
                pb.write(f'1. [{gf.name}]({url}/{gf.name})\n')
    
    def task_drops(self, path, task, data_drops):
        # Building /tasks/{task.name}/drops/index.md
        # Building /tasks/{task.name}/drops/{datadrop.name}.md
        index_path = path / 'index.md'
        with PageBuilder(index_path) as pb:
            pb.debug(f'task_drops()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task/drops', 'Drops'))
            pb.header('Annotation Task', task.name, 'Data drops')
            pb.p(f'Data drops for task **{task.name}** with size in number of files.')
            pb.table_header('data drop name', 'size', align='lr')
            for dd in task.data_drops.values():
                pb.table_row([f'[{dd.name}]({dd.name}.md)', len(dd)])
                self.task_drop(path, task, dd)

    def task_drop(self, path, task, datadrop):
        # Building /tasks/{task.name}/drops/{datadrop.name}.md
        index_path = path / f'{datadrop.stem}.md'
        with PageBuilder(index_path) as pb:
            pb.debug(f'task_drop()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task/drops/drop', 'Drop'))
            pb.header('Data Drop', datadrop.name)
            pb.p(f'Files in data drop **{datadrop.name}**,'
                 + f' with links to sources on the GitHub repository.')
            for fname in sorted(datadrop.file_names):
                file_url = f'{ANNOTATIONS_TREE}/{task.name}/{datadrop.name}/{fname}'
                pb.write(f'1. [{fname}]({file_url})\n')

    def task_batches(self, path, task):
        # Building /tasts/{task.name}/batches.md
        with PageBuilder(path) as pb:
            pb.debug(f'task_batches()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task', 'Task'))
            pb.header('Annotation Task', task.name, 'Batches')
            pb.navigation_subpages(subpages('tasks/task', 'batches'))
            pb.p('GUIDs from annotation batches that were used in this task:')
            pb.table_header('batch', 'guids', 'batch size', align='lrr')
            for batch in ANNOTATIONS.batches:
                comp = task.compare_to_batch(batch)
                batch_link = f'[{batch.stem}](../../batches/{batch.stem}/index.md)'
                pb.table_row([batch_link, comp.in_both, len(batch)])
    
    def task_script(self, path, task):
        # Building /tasts/{task.name}/script.md
        with PageBuilder(path) as pb:
            pb.debug(f'task_script()')
            pb.navigation_breadcrumbs(breadcrumbs('tasks/task', 'Task'))
            pb.header('Annotation Task', task.name, 'Script')
            pb.navigation_subpages(subpages('tasks/task', 'script'))
            pb.code(task.process_content())

    def evaluations(self):
       # Building /evaluations/index.md
       # Building /evaluations/{evaluation.name}/
        with PageBuilder(self.path / 'evaluations' / 'index.md') as pb:
            pb.navigation_breadcrumbs(breadcrumbs('evaluations', 'Evaluations'))
            pb.header('Evaluations')
            pb.table_header('Evaluation', 'Predictions', 'Reports', align='lrr')
            for evaluation in EVALUATIONS:
                pb.table_row([
                    f'[{evaluation.name}]({evaluation.name}/index.md)',
                    len(evaluation.predictions),
                    len(evaluation.reports)])
                self.evaluation(evaluation)

    def evaluation(self, evaluation):
        # Building /evaluations/{evaluation.name}/
        path = self.path / 'evaluations' / evaluation.name
        self.evaluation_index(path / 'index.md', evaluation)
        self.evaluation_readme(evaluation)
        self.evaluation_code(evaluation)
        self.evaluation_predictions(evaluation)
        self.evaluation_reports(evaluation)

    def evaluation_index(self, path, evaluation):
        # Building /evaluations/{evaluation.name}/index.md
        with PageBuilder(path) as pb:
            pb.debug('evaluation_index()')
            pb.navigation_breadcrumbs(
                breadcrumbs('evaluations/evaluation', 'Evaluation'))
            pb.header('Evaluation', evaluation.name)
            pb.navigation_subpages(
                subpages('evaluations/evaluation', 'evaluation'))
            pb.p(f'Evaluation **{evaluation.name}** '
                 f'with {len(evaluation.predictions)} predictions '
                 f'and {len(evaluation.reports)} reports.')

    def evaluation_readme(self, evaluation):
        # Building /evaluations/{evaluation.name}/readme_file.md
        path = self.path / 'evaluations' / evaluation.name / 'readme_file.md'
        with PageBuilder(path) as pb:
            pb.debug('evaluation_readme()')
            pb.navigation_breadcrumbs(
                breadcrumbs('evaluations/evaluation', 'Evaluation'))
            pb.header('Evaluation', evaluation.name, 'readme')
            pb.navigation_subpages(
                subpages('evaluations/evaluation', 'readme'))
            pb.write(f'{evaluation.readme}')

    def evaluation_code(self, evaluation):
        # Building /evaluations/{evaluation.name}/code.md
        path = self.path / 'evaluations' / evaluation.name / 'code.md'
        with PageBuilder(path) as pb:
            pb.debug('evaluation_code()')
            pb.navigation_breadcrumbs(
                breadcrumbs('evaluations/evaluation', 'Evaluation'))
            pb.header('Evaluation', evaluation.name, 'code')
            pb.navigation_subpages(
                subpages('evaluations/evaluation', 'code'))
            if len(evaluation.scripts) > 1:
                script_names = [f'**{s.name}**' for s in evaluation.scripts]
                pb.p(f'There are {len(evaluation.scripts)} code files: '
                     f'{" and ".join(script_names)}.')
            for script in evaluation.scripts:
                pb.subheader(f'{script.name}')
                url = f'{EVALUATIONS_TREE}/{evaluation.name}/{script.name}'
                pb.p(f'View [{script.name}]({url}) in the Evaluation repository on GitHub')
                code = read_file(evaluation.path / script.name)
                pb.code(code)

    def evaluation_predictions(self, evaluation):
        # Building /evaluations/{evaluation.name}/predictions/index.md
        path = self.path / 'evaluations' / evaluation.name / 'predictions' / 'index.md'
        with PageBuilder(path) as pb:
            pb.debug('evaluation_predictions()')
            pb.navigation_breadcrumbs(
                breadcrumbs('evaluations/evaluation/predictions', 'Predictions'))
            pb.header('Evaluation', evaluation.name, 'predictions')
            pb.navigation_subpages(
                subpages('evaluations/evaluation/predictions', 'predictions'))
            for n, prediction in enumerate(evaluation.predictions):
                batch_name = prediction.prediction_batch
                if batch_name not in ANNOTATIONS.batch_names:
                    pb.warning(f'batch {batch_name} referenced in item {n+1} below '
                               f'does not exist')
            pb.p('List of system prediction batches with number of files.')
            pb.p('Click the link to see the readme file and system output on GitHub.')
            pb.table_header('n', 'prediction batch', 'files', align='rlr')
            for n, prediction in enumerate(evaluation.predictions):
                url = f'{EVALUATIONS_TREE}/{evaluation.name}/{prediction.name}'
                pb.table_row([n+1, f'[{prediction.name}]({url})', len(prediction)])

    def evaluation_reports(self, evaluation):
        # Building /evaluations/{evaluation.name}/reports/index.md
        path = self.path / 'evaluations' / evaluation.name / 'reports' / 'index.md'
        with PageBuilder(path) as pb:
            pb.debug('evaluation_reports()')
            pb.navigation_breadcrumbs(
                breadcrumbs('evaluations/evaluation/reports', 'Reports'))
            pb.header('Evaluation', evaluation.name, 'reports')
            # subpage structure is same as for predictions
            pb.navigation_subpages(
                subpages('evaluations/evaluation/predictions', 'reports'))
            for report in evaluation.reports:
                url = f'{EVALUATIONS_TREE}/{evaluation.name}/{report.name}'
                pb.p(f'**{remove_at(report.name)}** ([view on GitHub]({url}))')
                batch_name = report.report_batch
                if batch_name not in ANNOTATIONS.batch_names:
                    pb.warning(f'batch {batch_name} does not exist')



class PageBuilder:

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._debug = DEBUG

    def __enter__(self):
        self.fh = self.path.open('w')
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            print(exc_type, exc_value, exc_tb)
        self.fh.close()

    def __str__(self):
        return f'<PageBuilder debug={self._debug}>'

    def debug(self, text: str):
        if self._debug:
            self.fh.write(f'<pre>{text}</pre>\n\n')

    def write(self, text: str):
        self.fh.write(text)

    def header(self, *components: list):
        sep = ' &nbsp; ⎯ &nbsp; '
        self.fh.write(f'\n# {sep.join(components)}\n\n')

    def subheader(self, text: str):
        self.fh.write(f'#### {text}\n\n')

    def navigation_breadcrumbs(self, breadcrumbs: list):
        """Adds the markdown for a navigation bar, including the breadcrumbs, the
        sub pages if any, and a horizontal rule as a separator."""
        for n, (name, link) in enumerate(breadcrumbs):
            prefix = '' if n == 0 else ' &nbsp; > &nbsp; '
            if link is None:
                self.fh.write(f'{prefix}**{name}** ')
            else:
                self.fh.write(f'{prefix}[{name}]({link}) ')
        self.fh.write('\n')

    def navigation_subpages(self, pages: list):
        for n, (name, link) in enumerate(pages):
            prefix = '' if n == 0 else '| '
            if link is None:
                self.fh.write(f'{prefix}**{name}** ')
            else:
                self.fh.write(f'{prefix}[{name}]({link}) ')
        self.fh.write('\n\n')

    def p(self, text: str):
        self.fh.write(f'{text}\n\n')

    def nl(self):
        self.fh.write('\n')

    def hr(self):
        self.fh.write(f'---\n\n')

    def table_header(self, *headers, align=None):
        if align is None:
            align = 'l' * len(headers)
        # TODO: now just allows 'l' and 'r', may need 'c' as well
        header_strings = []
        header_aligns = []
        for (h, a) in zip(headers, align):
            header_strings.append(h)
            header_align = ':------' if a == 'l' else '------:'
            header_aligns.append(header_align)
        self.fh.write(f'| {" | ".join(header_strings)} |\n')
        self.fh.write(f'| {" | ".join(header_aligns)} |\n')

    def table_rows(self, rows: list):
        for row in rows:
            self.table_row(row)
        self.nl()

    def table_row(self, row: list):
        row = [str(element) for element in row]
        self.fh.write(f'| {" | ". join(row)} |\n')

    def code(self, text: str, language='python'):
        self.fh.write(f'```{language}\n')
        self.fh.write(text)
        self.fh.write('\n```\n')

    def pre(self, text: str):
        self.fh.write(f'<pre>\n{text.strip()}\n</pre>\n\n')

    def warning(self, text):
        self.fh.write(f'🟠 *Warning: {text}*\n\n')



if __name__ == '__main__':

    if '--debug' in sys.argv:
        DEBUG = True
    builder = SiteBuilder('../docs/www')
    builder.build()


'''

🕵️‍♀️ 🌍 🌎 🌏 👁

🔴🟠🟡🟢🔵🟣🟤⚫⚪🔘🛑⭕

🟥🟧🟨🟩🟦🟪🟫⬛⬜🔲🔳⏹☑✅❎

🔺🔻🔷🔶🔹🔸♦💠💎💧🧊

🏴🏳🚩🏁

◻️◼️◾️◽️▪️▫️

'''
