[Dashboard](../../index.md)  &nbsp; > &nbsp; [Tasks](../index.md)  &nbsp; > &nbsp; **Task** 

# Annotation Task &nbsp; ⎯ &nbsp; newshour-namedentity &nbsp; ⎯ &nbsp; Script

[task](index.md) | [readme](readme_file.md) | [gold files](golds.md) | [data drops](drops/index.md) | [batches](batches.md) | **script** 

```python
"""Processing NER uploads.

NER annotation for this project is done with Brat tool and the output format is 
Brat standalone `.ann` format. We will use the `.ann` format as the gold format as
well. Thus processing these "raws" into golds files is simply just copying files.
"""
import pathlib
import shutil

if __name__ == '__main__':
    root_dir = pathlib.Path(__file__).parent
    golds_dir = root_dir / 'golds'
    golds_dir.mkdir(exist_ok=True)
    for batch_dir in root_dir.glob('*'):
        if batch_dir.is_dir() and len(batch_dir.name) > 7 and batch_dir.name[6] == '-' and all([c.isdigit() for c in batch_dir.name[:6]]):
            for ann in batch_dir.glob('*.ann'):
                shutil.copy(ann, golds_dir)

```
