import config
import annotation
import evaluation


class Data:

    """Class used for when we need access to information from both the annotations
    and the evaluations."""

    def __init__(self, annotations_repo: str, evaluations_repo: str):
        self.annotations = annotation.Repository(annotations_repo)
        self.evaluations = evaluation.Repository(evaluations_repo)

    def __str__(self):
        return f'<Data>\n    {self.annotations}\n    {self.evaluations}'

    def batch_usage_in_system_predictions(self, batch_name: str):
        """For a batch name from the annotation repository, return a list of pairs with
        evaluation name and system predictions name."""
        data = []
        for evaluation in self.evaluations:
            for prediction in evaluation.predictions:
                if prediction.prediction_batch == batch_name:
                    data.append([evaluation.name, prediction.name])
        return data

    def batch_usage_in_system_reports(self, batch_name: str):
        """For a batch name from the annotation repository, return a list of pairs with
        evaluation name and system report name."""
        data = []
        for evaluation in self.evaluations:
            for report in evaluation.reports:
                if report.report_batch == batch_name:
                    data.append([evaluation.name, report.name])
        return data



if __name__ == '__main__':

    data = Data(config.ANNOTATIONS, config.EVALUATIONS)
    print(data)
    for batch_name in data.annotations.batch_names:
        print(f'\n{data.annotations.batch(batch_name)}')
        batch_usage = data.batch_usage_in_system_predictions(batch_name)
        for evaluation, prediction in batch_usage:
            print('   ', evaluation, prediction)
        for evaluation, report in data.batch_usage_in_system_reports(batch_name):
            print('   ', evaluation, report)