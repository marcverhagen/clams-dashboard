[Dashboard](../../index.md)  &nbsp; > &nbsp; [Evaluations](../index.md)  &nbsp; > &nbsp; **Evaluation** 

# Evaluation &nbsp; ⎯ &nbsp; nel_eval &nbsp; ⎯ &nbsp; code

[evaluation](index.md) | [readme](readme_file.md) | **code** | [predictions](predictions/index.md) | [reports](reports/index.md) 

There are 4 code files: **test_nel.py** and **nel.py** and **evaluate.py** and **goldretriever.py**.

#### test_nel.py

View [test_nel.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/nel_eval/test_nel.py) in the Evaluation repository on GitHub

```python
"""
Unit test for NamedEntityLink class
"""

from unittest import TestCase
from nel import NamedEntityLink

# define some toy NEL data
united_nations = {"doc_id": "d1", "begin_offset": 50688, "end_offset": 50702, "entity_type": "location",
                  "surface_form": "United Nations", "uris": ["http://www.wikidata.org/entity/Q1065"]}
un = {"doc_id": "d1", "begin_offset": 50786, "end_offset": 50788, "entity_type": "location",
      "surface_form": "UN", "uris": ["http://www.wikidata.org/entity/Q1065"]}
terence_smith = {"doc_id": "d1", "begin_offset": 52886, "end_offset": 52899, "entity_type": "person",
                 "surface_form": "TERENCE SMITH", "uris": ["http://www.wikidata.org/entity/Q7702012"]}
jim_lehrer = {"doc_id": "d1", "begin_offset": 52966, "end_offset": 52976, "entity_type": "person",
              "surface_form": "JIM LEHRER", "uris": ["https://www.wikidata.org/wiki/Q931148"]}
# the first is constructed from system output data, the second is constructed from gold labeled data
cambodia_1 = {"doc_id": "d1", "begin_offset": 52511, "end_offset": 52519, "entity_type": "location",
              "surface_form": "Cambodia", "uris": ["http://www.wikidata.org/entity/Q1054184",
                                                   "http://www.wikidata.org/entity/Q2387250",
                                                   "http://www.wikidata.org/entity/Q424",
                                                   "http://www.wikidata.org/entity/Q867778"]}
cambodia_2 = {"doc_id": "d1", "begin_offset": 52511, "end_offset": 52519, "entity_type": "location",
              "surface_form": "Cambodia", "uris": "https://www.wikidata.org/wiki/Q424"}


class TestNEL(TestCase):
    def test_init(self):
        """Test constructor."""
        nel_terence_smith = NamedEntityLink(**terence_smith)
        self.assertEqual(nel_terence_smith.span, "d1: 52886 - 52899")
        self.assertEqual(nel_terence_smith.surface_form, "TERENCE SMITH")
        self.assertEqual(nel_terence_smith.entity_type, "person")
        self.assertEqual(nel_terence_smith.kbid, frozenset(["Q7702012"]))

    def test_equal(self):
        """Test whether two distinct NEL instances are equal.
        Span, entity_type, and kbid are compared.
        kbid's are treated as sets and equivalence is based on whether the intersection is not empty.
        """
        nel_cambodia_1 = NamedEntityLink(**cambodia_1)
        nel_cambodia_2 = NamedEntityLink(**cambodia_2)
        self.assertEqual(nel_cambodia_1, nel_cambodia_2)

    def test_not_equal(self):
        """Test whether two distinct NEL instances are not equal."""
        nel_united_nations = NamedEntityLink(**united_nations)
        nel_un = NamedEntityLink(**un)
        self.assertNotEqual(nel_united_nations, nel_un)

    def test_intersection(self):
        """Test whether set intersection captures distinct NEL instances that are equivalent."""
        nel_jim_lehrer = NamedEntityLink(**jim_lehrer)
        nel_cambodia_1 = NamedEntityLink(**cambodia_1)
        sys_instances = frozenset([nel_jim_lehrer, nel_cambodia_1])

        nel_cambodia_2 = NamedEntityLink(**cambodia_2)
        nel_un = NamedEntityLink(**un)
        gold_instances = frozenset([nel_cambodia_2, nel_un])

        intersection = gold_instances.intersection(sys_instances)
        self.assertIn(nel_cambodia_2, intersection)
        self.assertTrue(len(intersection) == 1)

```
#### nel.py

View [nel.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/nel_eval/nel.py) in the Evaluation repository on GitHub

```python
"""
Represents an instance of a Named Entity Link (NEL)
"""

from typing import List, Union


class NamedEntityLink:
    def __init__(self, doc_id: str, begin_offset: int, end_offset: int, entity_type: str,
                 surface_form: str, uris: Union[List[str], str]) -> None:
        """
        Initializes NEL instance. Gold data provides the URI as a string, whereas system output MMIFs provide
        the URI(s) in a list format.
        :param doc_id: the ID of the context document.
        :param begin_offset: position of the first character in the span.
        :param end_offset: position of the last character in the span.
        :param entity_type: the category of the entity.
        :param surface_form: the text corresponding to the entity.
        :param uris: the Wikidata URI(s) grounding the entity.
        """
        self.doc_id = doc_id
        self.begin_offset = begin_offset
        self.end_offset = end_offset
        self.entity_type = entity_type
        self.span = f"{self.doc_id}: {self.begin_offset} - {self.end_offset}"
        self.surface_form = surface_form
        self.kbid = frozenset() # knowledge base ID
        if isinstance(uris, str):
            if uris != '':
                self.kbid = frozenset([uris.rsplit("/", 1)[1]]) # get QID from URI
        elif isinstance(uris, list):
            # multiple wikidata URIs
            self.kbid = frozenset([uri.rsplit("/", 1)[1] for uri in uris])
        else:
            raise TypeError(f"Argument uris must be of type List[str] or str, not {type(uris)}")

    def __str__(self) -> str:
        """Returns a printable string representation of the NEL instance."""
        return f"{self.span} (QID: {self.kbid})"

    def __eq__(self, other) -> bool:
        """Returns True if another object is an NEL instance with the same span (doc id and offsets), entity type,
        and a QID matching this one. Returns False otherwise."""
        if isinstance(other, NamedEntityLink):
            return self.span == other.span and self.entity_type == other.entity_type \
                   and self.kbid.intersection(other.kbid)
        return False

    def __ne__(self, other) -> bool:
        """Returns True if another object is not an NEL instance or if it is an NEL instance not equal to this one.
        Returns False if another object is an NEL instance equal to this one."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Returns the hash value for the NEL object."""
        return hash((self.span, self.entity_type))

```
#### evaluate.py

View [evaluate.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/nel_eval/evaluate.py) in the Evaluation repository on GitHub

```python
"""Named Entity Linking Evaluation

$ evaluate.py [-h] [-o [OUTPUT]] [test-dir] [gold-dir]

Evaluate .mmif files with named entity linking annotations produced as output
from the DBpedia Spotlight wrapper app. Compares system
generated data with the gold data found in the aapb-annotations
repository.

NOTE: gold annotation files and test output files that correspond
to the same AAPB catalog item must begin with the same GUID.
i.e. `gold-files/cpb-aacip-507-1v5bc3tf81.tsv` and `test-files/cpb-aacip-507-1v5bc3tf81-transcript.txt.dbps.mmif

"""

import argparse
from collections import defaultdict
import fnmatch
import json
from lapps.discriminators import Uri
from mmif.serialize import Mmif
import os
import pandas as pd

import goldretriever
from nel import NamedEntityLink


def match_files(test_dir, gold_dir) -> list:
    """Compare the files in the gold and test directories. Return pairs of matching files in a list.
    :param test_dir: Directory of test .mmif files
    :param gold_dir: Directory of gold .tsv files
    :return: list of tuples containing corresponding data file locations in (test, gold) format.
    """
    test_files, gold_files = os.listdir(test_dir), os.listdir(gold_dir)
    file_matches = []
    for gold_file in gold_files:
        pattern = gold_file[:24] + "*"
        for test_file in test_files:
            if fnmatch.fnmatch(test_file, pattern):
                gold_match = os.path.join(gold_dir, gold_file)
                test_match = os.path.join(test_dir, test_file)
                file_matches.append((test_match, gold_match))
                test_files.remove(test_file)
                break

    return file_matches


def filter_nil_entities(gold_tsv) -> list:
    """Returns list of gold entities excluding the ones without grounding."""
    gold_entities = file_to_ne(gold_tsv)
    # remove gold NEL instances whose QIDs are empty strings
    gold_entities = [gold_ent for gold_ent in gold_entities if gold_ent.kbid]

    return gold_entities


def file_to_ne(file_path: str) -> list:
    """Checks whether the file is in .mmif or .tsv format and calls the appropriate function
    to get a list of NEL objects"""
    if file_path.endswith('.mmif'):
        return mmif_to_ne(file_path)
    elif file_path.endswith('.tsv'):
        return tsv_to_ne(file_path)
    else:
        raise Exception("Unsupported file type.")


def mmif_to_ne(mmif_path) -> list:
    """Fetch named entities from the input mmif.
    Returns a list of NEL objects.
    """
    with open(mmif_path) as fh_in:
        mmif_serialized = fh_in.read()
    mmif = Mmif(mmif_serialized)
    ne_views = mmif.get_all_views_contain(at_types=Uri.NE)
    view = ne_views[-1]  # read only the first view (from last) with Uri.NE
    annotations = view.get_annotations(at_type=Uri.NE)
    ne_list = []
    guid = os.path.basename(mmif_path)[:24]
    for anno in annotations:
        entity = anno.properties
        ne = NamedEntityLink(guid, entity['start'], entity['end'], entity['category'], entity['text'],
                             entity['grounding'][1:])
        ne_list.append(ne)

    return ne_list


def tsv_to_ne(gold_tsv_path) -> list:
    """Fetch named entities from the input tsv.
    Returns a list of NEL objects."""
    with open(gold_tsv_path) as fh_in:
        annotations_df = pd.read_csv(fh_in, sep='\t', encoding='utf-16')
        annotations_df.fillna('', inplace=True)
        annotations_df['guid'] = annotations_df['guid'].apply(lambda x: x[:24])
    ne_list = [NamedEntityLink(guid, begin_offset, end_offset, ent_type, text, uris) for
               guid, begin_offset, end_offset, ent_type, text, uris in
               zip(annotations_df['guid'], annotations_df['begin_offset'], annotations_df['end_offset'],
                   annotations_df['type'], annotations_df['text'], annotations_df['qid'])]

    return ne_list


def evaluate(test_dir, gold_dir=None):
    if gold_dir is None:
        gold_dir = goldretriever.download_golds('https://github.com/clamsproject/aapb-annotations/tree/feaf342477fc27e57dcdcbb74c067aba4a02e40d/newshour-namedentity-wikipedialink/golds/aapb-collaboration-21')
    results = defaultdict(dict)
    file_matches = match_files(test_dir, gold_dir)
    for sys_file, gold_file in file_matches:
        print(f'>>> Evaluating {os.path.basename(sys_file)}')
        guid = os.path.basename(sys_file)[:24]

        sys_instances = frozenset(file_to_ne(sys_file))
        gold_instances = frozenset(filter_nil_entities(gold_file))
        results[guid]['Gold Entities'] = {"count": len(gold_instances)}
        results[guid]['System Entities'] = {"count": len(sys_instances)}

        # calculate precision
        precision = len(gold_instances.intersection(sys_instances)) / len(sys_instances)

        # calculate recall
        recall = len(gold_instances.intersection(sys_instances)) / len(gold_instances)

        # calculate F1
        if precision + recall == 0:  # avoid ZeroDivisionError
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        results[guid]['Precision'] = "{:.2f}".format(precision)
        results[guid]['Recall'] = "{:.2f}".format(recall)
        results[guid]['F1'] = "{:.2f}".format(f1)
        print('>>> ... done')

    return results


def write_results(data: dict, result_path: str):
    """Write evaluation results to txt file."""
    with open(result_path, 'w') as fh_out:
        json.dump(data, fh_out, indent=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Evaluate accuracy of NEL mmif files against gold labeled data.')
    ap.add_argument('system_data_directory', metavar='sys-dir', nargs='?',
                    help='directory containing system output data in .mmif format.')
    ap.add_argument('gold_directory', metavar='gold-dir', nargs='?',
                    help='directory containing gold data in .tsv format.')
    ap.add_argument('-o', '--output', nargs='?', help='path to print out eval result.', default='results.txt')
    args = ap.parse_args()

    data = evaluate(args.system_data_directory, args.gold_directory)
    write_results(data, args.output)

```
#### goldretriever.py

View [goldretriever.py](https://github.com/clamsproject/aapb-evaluations/tree/854eeb362d3500232982eda53bda4eb47d76df51/nel_eval/goldretriever.py) in the Evaluation repository on GitHub

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
