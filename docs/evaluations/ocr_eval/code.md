[Dashboard](../../index.md)  &nbsp; > &nbsp; [Evaluations](../index.md)  &nbsp; > &nbsp; **Evaluation** 

# Evaluation &nbsp; ⎯ &nbsp; ocr_eval &nbsp; ⎯ &nbsp; code

[evaluation](index.md) | [readme](readme.md) | **code** | [predictions](predictions/index.md) | [reports](reports/index.md) 

There are 3 code files: **__init__.py** and **evaluate.py** and **goldretriever.py**.

#### __init__.py

View [__init__.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/ocr_eval/__init__.py) in the Evaluation repository on GitHub

```python

```
#### evaluate.py

View [evaluate.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/ocr_eval/evaluate.py) in the Evaluation repository on GitHub

```python
import argparse
import csv
import json
import os
import pathlib
import re
from collections import defaultdict
from typing import Dict, Union, Tuple, Iterable

import numpy
from mmif import Mmif, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from torchmetrics.text import CharErrorRate

from clams_utils.aapb import goldretriever as gr

# Constants ==|
GOLD_URL = 'https://github.com/clamsproject/aapb-annotations/tree/f96f857ef83acf85f64d9a10ac8fe919e06ce51e/newshour-chyron/golds/batch2'


def filename_to_guid(filename) -> str:
    guid = pathlib.Path(filename).stem
    if "." in guid:
        guid = guid.split('.')[0]
    return guid


def load_reference(ref_fname) -> Dict[Tuple[float, float], str]:
    gold_annotations = {}
    with open(ref_fname, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            start, end, text = row
            # text normalization MUST write about this in the report!
            text = text.strip().lower()
            text = text.replace(r'\n', ' ')
            gold_annotations.update({(start, end): text})
    return gold_annotations


def load_references(ref_dir: Union[str, pathlib.Path]) -> Dict[str, Dict[Tuple[float, float], str]]:
    """
    Loads OCR data from the gold directory, 
    text transcripts are keyed by guid, then by (start, end) tuple ,
    where start and end are in seconds (floats)
    """
    ref_dir = pathlib.Path(ref_dir)
    refs = {}
    for ref_src_fname in ref_dir.glob("*.?sv"):
        guid = filename_to_guid(ref_src_fname)
        refs[guid] = load_reference(ref_src_fname)
    return refs


def load_hypotheses(mmif_files: Iterable[pathlib.Path]) -> Dict[str, Dict[float, str]]:
    """
    Load in hypothesis mmifs and retrieve annotations,
    text transcripts are keyed by guid, then by timepoint (float)
    the unit of the timepoints is seconds (converted if necessary, assuming 29.97 fps)
    """
    hyps = defaultdict(dict)
    for mmif_file in mmif_files:
        guid = filename_to_guid(mmif_file)
        mmif = Mmif(open(mmif_file).read())
        vd = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        try:
            fps = vd.get('fps')
        except KeyError:
            # meaning there were no video document or no file found for the VD
            continue
        ocr_view = mmif.get_view_contains(AnnotationTypes.Alignment)
        bb_view = mmif.get_view_contains(AnnotationTypes.BoundingBox)
        tp_view = mmif.get_view_contains(AnnotationTypes.TimePoint)
        if ocr_view is not None:
            anno_id_to_annotation = {annotation.id: annotation
                                     for annotation in ocr_view.annotations}
            doc_id_to_doc = {doc.id: doc
                             for doc in mmif.get_documents_by_type(DocumentTypes.TextDocument)}
        if bb_view is not None:
            anno_id_to_annotation.update({annotation.id: annotation
                                          for annotation in bb_view.annotations})
        ocr_results = hyps[guid]
        if ocr_view is not None:
            if "doctr" in ocr_view.metadata.app.lower():
                for alignment in ocr_view.get_annotations(AnnotationTypes.Alignment):
                    source_id = alignment.properties['source']
                    target_id = alignment.properties['target']
                    if source_id.startswith("tp"):
                        source_anno = tp_view[source_id]
                        target_doc = doc_id_to_doc[target_id]
                        time_point = vdh.convert(source_anno.properties['timePoint'], 'milliseconds', 'seconds', fps)
                        time_text = target_doc.text_value
                        time_text = time_text.strip().lower()
                        # Remove newlines and double newlines
                        time_text = re.sub(r'\n+', ' ', time_text)
                        ocr_results[time_point] = time_text
            elif "paddle" in ocr_view.metadata.app.lower():
                for alignment in ocr_view.get_annotations(AnnotationTypes.Alignment):
                    if "td" in alignment.properties['target']:
                        source_id = alignment.properties['source']
                        target_id = alignment.properties['target']
                        source_anno = tp_view[source_id.split(":")[1]]
                        target_doc = doc_id_to_doc[target_id.split(":")[1]]
                        time_point = vdh.convert(source_anno.properties['timePoint'], 'milliseconds', 'seconds', fps)
                        time_text = target_doc.text_value
                        time_text = time_text.strip().lower()
                        # Remove newlines and double newlines
                        time_text = re.sub(r'\n+', ' ', time_text)
                        ocr_results[time_point] = time_text
            else:
                for annotation in ocr_view.annotations:
                    if annotation.at_type == AnnotationTypes.Alignment:
                        source_id = annotation.properties['source']
                        target_id = annotation.properties['target']
                        vid_anno, bbox_anno = source_id.split(":")
                        source_anno = anno_id_to_annotation[bbox_anno]
                        target_anno = anno_id_to_annotation[target_id]
                        time_point = vdh.convert(source_anno.properties['timePoint'], 'frames', 'seconds', fps)
                        time_text = target_anno.properties['text'].value
                        time_text = time_text.strip().lower()
                        if time_point not in ocr_results:
                            ocr_results[time_point] = time_text
                        else:
                            ocr_results[time_point] = ocr_results[time_point] + ' ' + time_text
    return hyps


def cer_by_timeframe(ref: Dict[tuple, str], hyp: Dict[float, str]):
    vals = {}
    doc_level_cer = CharErrorRate()
    for timepoint, text in hyp.items():
        for timespan, gold_text in ref.items():
            start = float(timespan[0])
            end = float(timespan[1])
            if start <= timepoint and end > timepoint:
                if start not in vals:
                    vals[start] = {'ref_text': gold_text, 'hyp_text': text}
                else:
                    if len(vals[start]['hyp_text']) < len(text):
                        vals[start]['hyp_text'] = text
        for comp in vals.values():
            comp['cer'] = doc_level_cer(comp['hyp_text'], comp['ref_text']).item()
    return vals


def evaluate(gold_data: Dict[str, Dict[tuple, str]], test_data: Dict[str, Dict[float, str]],
             outdir: pathlib.Path) -> None:
    output = {}
    for guid, annotations in test_data.items():
        if guid in gold_data:
            results = cer_by_timeframe(gold_data[guid], annotations)
            output[guid] = results
            cers = [comp['cer'] for comp in results.values()]
            output[guid]['mean_cer'] = numpy.mean(cers)
            with open(outdir/f'{guid}.json', 'w') as b:
                json.dump(results, b, indent=2)
    cers = []
    with open(outdir/'results.txt', 'w') as f:
        for guid, results in output.items():
            cers.append(results['mean_cer'])
            f.write(f'{guid}:\t{results["mean_cer"]}\n')
        f.write(f"Total Mean CER:\t{numpy.mean(cers)}\n")


# Main Block
if __name__ == "__main__":
    APPNAME = "app-doctr-wrapper"  # default app
    APPVERSION = 1.0  # default version 

    parser = argparse.ArgumentParser(description="compare the results of CLAMS OCR apps to a gold standard")
    parser.add_argument('-t', '--test-dir',
                        type=str,
                        default=None,
                        help="Directory of non-gold/hypothesis mmif files")
    parser.add_argument('-g', '--gold-dir',
                        type=str,
                        default=None,
                        help="Directory of gold annotations")
    parser.add_argument('-o', '--output-dir',
                        type=str,
                        default=None,
                        help="Directory to store results")
    args = parser.parse_args()

    ref_dir = gr.download_golds(GOLD_URL) if args.gold_dir is None else args.gold_dir
    hyp_dir = f"preds@{APPNAME}{APPVERSION}@batch2" if args.test_dir is None else args.test_dir
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(f"results@{APPNAME}{APPVERSION}@batch2")
    if not out_dir.exists():
        out_dir.mkdir()

    references = load_references(pathlib.Path(ref_dir))
    hypotheses = load_hypotheses(pathlib.Path(hyp_dir).glob("*.mmif*"))
    evaluate(references, hypotheses, out_dir)

```
#### goldretriever.py

View [goldretriever.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/ocr_eval/goldretriever.py) in the Evaluation repository on GitHub

```python
import json
from pathlib import Path
from urllib.parse import urljoin

import requests


def download_golds(gold_dir_url, folder_name=None):
    import tempfile
    # code adapt from Angela Lam's

    if folder_name is None:
        folder_name = tempfile.TemporaryDirectory().name
    # Create a new directory to store the downloaded files on local computer
    target_dir = Path(folder_name)
    if not target_dir.exists():
        target_dir.mkdir()

    # Check if the directory is empty
    try:
        next(target_dir.glob('*'))
        raise Exception("The folder '" + folder_name + "' already exists and is not empty")
    except StopIteration:
        pass

    # Send a GET request to the repository URL and extract the HTML content
    response = requests.get(gold_dir_url)

    # github responses with JSON? wow
    payload = json.loads(response.text)['payload']
    links = [i['path'] for i in payload['tree']['items']]

    # Download each file in the links list into the created folder
    for link in links:
        raw_url = urljoin('https://raw.githubusercontent.com/',
                          '/'.join((payload['repo']['ownerLogin'],
                                    payload['repo']['name'],
                                    payload['refInfo']['name'],
                                    link)))
        file_path = target_dir / link.split('/')[-1]
        with open(file_path, 'wb') as file:
            response = requests.get(raw_url)
            file.write(response.content)
    return folder_name

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download gold files from a github repository')
    parser.add_argument('-d', '--download_dir', default=None, 
                        help='The name of the folder to store the downloaded files. '
                             'If not provided, a system temporary directory will be created')
    parser.add_argument('gold_url', help='The URL of the gold directory')
    args = parser.parse_args()
    download_golds(args.gold_url, args.download_dir)
    
```
