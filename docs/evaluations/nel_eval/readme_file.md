[Dashboard](../../index.md)  &nbsp; > &nbsp; [Evaluations](../index.md)  &nbsp; > &nbsp; **Evaluation** 

# Evaluation &nbsp; ⎯ &nbsp; nel_eval &nbsp; ⎯ &nbsp; readme

[evaluation](index.md) | **readme** | [code](code.md) | [predictions](predictions/index.md) | [reports](reports/index.md) 

# NEL Evaluation

`evaluate.py` processes `.mmif` files with named entity linking annotations produced as output
from the [DBpedia Spotlight wrapper](https://github.com/clamsproject/app-dbpedia-spotlight-wrapper) app.
The system-generated data in the `.mmif` files are compared to the gold `.tsv` data in order to compute precision, recall, and F1 for each AAPB-GUID.
Gold annotations can be found in the [aapb-annotations](https://github.com/clamsproject/aapb-annotations) repository.


__Note__: gold annotation files and test output files that correspond
to the same aapb catalog item must share the same file name (with the exception of file extension).
i.e. `gold-files/cpb-aacip-507-1v5bc3tf81.tsv` and `test-files/cpb-aacip-507-1v5bc3tf81.mmif`.

## Metrics

Evaluation follows the description provided by Ji et al. (2017) regarding the Entity Discovery and Linking task of the Knowledge Base Population track at TAC 2017. 

Annotations are treated as tuples (Span, Type, KBID).
- **Span** consists of the document ID (for our purposes, the GUID) and the begin/end character offsets of the entity in the text.
- **Type** corresponds to Category (one of Person, Location, Organization, Product, Event and Title).
- **KBID** (Knowledge Base ID) for our use case is the Wikidata Q Identifier.

Metrics are calculated as follows:

- **Precision**: computed by taking the cardinality of the intersection between the set of Gold annotations and the set of System annotations, divided by the cardinality of the set of System annotations.
- **Recall**: computed by taking the cardinality of the intersection between the set of Gold annotations and the set of System annotations, divided by the cardinality of the set of Gold annotations.
- **F1**: The conventional F1 formula.

The dbpedia spotlight app does not seem to perform NIL recognition (marking entities as not having a node in a KB), so this will not be evaluated.


## Output file format

The output txt file is in json format, where each top level item is a dictionary corresponding
to the catalog item GUID. Within each dictionary is a field for the metrics and two nested dictionaries
describing how many gold/test entities were retained for the evaluation.
Gold labeled entities were omitted from the evaluation when there was no associated QID available.

Example output:

```json
{
    "cpb-aacip-507-1v5bc3tf81": {
        "Gold Entities": {
            "count": 528
        },
        "System Entities": {
            "count": 238
        },
        "Precision": "0.37",
        "Recall": "0.17",
        "F1": "0.23"
    }
}
```

## References

Heng Ji, Xiaoman Pan, Boliang Zhang, Joel Nothman,
James Mayfield, Paul McNamee, and Cash Costello.
2017. Overview of TAC-KBP2017 13 languages entity discovery and linking. In _Proceedings of the 2017 Text Analysis Conference, TAC 2017, Gaithersburg, Maryland, USA, November 13-14, 2017._