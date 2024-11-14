import streamlit as st
import utils
import config


def viewer(MODEL):

    ANNOTATIONS = MODEL.annotations
    EVALUATIONS = MODEL.evaluations

    st.title('CLAMS Evaluation Viewer')
    navigation_col, eval_col = st.columns([0.2, 0.8])

    eval_name = navigation_col.radio(
        'eval-category', EVALUATIONS.evaluation_names, label_visibility='collapsed')
    evaluation = EVALUATIONS.evaluation(eval_name)

    eval_col.info(
        f'Evaluation "{evaluation.name}"'
        + f' with {len(evaluation.predictions)} predictions'
        + f' and {len(evaluation.reports)} reports')

    readme_tab, code_tab, predictions_tab, reports_tab = eval_col.tabs(
        [ 'Readme', 'Code', 'Predictions', 'Reports'])

    with readme_tab:
        readme_tab.markdown(evaluation.readme)
    
    with code_tab:
        code_file_names = [f.name for f in evaluation.scripts]
        code_file = code_tab.radio(
            'code_file', code_file_names, label_visibility='collapsed')
        code_tab.code(utils.read_file(evaluation.path / code_file))

    with predictions_tab:
        prediction = utils.st_list_files(
            predictions_tab, 'prediction', evaluation.prediction_names)
        if prediction is not None:
            prediction_obj = evaluation.prediction(prediction)
            prediction_files = prediction_obj.file_names()
            p_batch = prediction_obj.prediction_batch
            predictions_tab.info(f'Predictions batch **{prediction}** with {len(prediction_files)} prediction files')
            if prediction_obj.prediction_batch not in ANNOTATIONS.batch_names:
                predictions_tab.warning(f'Warning: there is no batch named "{p_batch}" in the annotations repository')
            predictions_readme_tab, predictions_files_tab = predictions_tab.tabs(
                ['Readme file', 'Prediction files'])
            with predictions_readme_tab:
                predictions_readme_tab.markdown(prediction_obj.readme)
            with predictions_files_tab:
                prediction_file = utils.st_list_files(
                    predictions_files_tab, 'prediction_files', prediction_files)
                path = evaluation.path / prediction / prediction_file
                file_size = path.stat().st_size
                predictions_files_tab.markdown(f'*File size: {file_size:,}*')
                if file_size < config.MAX_FILESIZE:
                    utils.st_display_file(predictions_files_tab, path)
                else:
                    predictions_files_tab.markdown(f'*This file is rather big, print it anyway?*')
                    print_file = predictions_files_tab.button('Print File')
                    if print_file:
                        utils.st_display_file(predictions_tab, path)

    with reports_tab:
        report = utils.st_list_files(
            reports_tab, 'report', evaluation.reports_idx.keys())
        if report is not None:
            # TODO: could now use the Report object for this
            report_obj = evaluation.report(report)
            report_name = utils.remove_at(report_obj.name)
            report_batch = report_obj.report_batch
            reports_tab.info(f'Report **{report_name}**')
            if report_batch not in ANNOTATIONS.batch_names:
                reports_tab.warning(f'Warning: there is no batch named "{report_batch}" in the annotations repository')
            reports_tab.markdown(utils.read_file(evaluation.path / report))
