[Dashboard](../../index.md)  &nbsp; > &nbsp; [Tasks](../index.md)  &nbsp; > &nbsp; **Task** 

# Annotation Task &nbsp; ⎯ &nbsp; january-slates &nbsp; ⎯ &nbsp; Script

[task](index.md) | [readme](readme_file.md) | [gold files](golds.md) | [data drops](drops/index.md) | [batches](batches.md) | **script** 

```python
"""
Processes Slate annotation files. 
To read all the tabular files from in the YYMMDD-batchname directories and generate one file per GUID in golds
"""

import os
import pathlib

import pandas as pd


def process_csv(input_directory, output_directory):
    desired_columns = ["GUID", "collection", "start", "end", "type", "digital", "format-summary", "moving-elements"]
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_directory, filename)
            df = pd.read_csv(filepath,encoding='utf-8')
            df.replace(',', '', regex=True, inplace=True)
            df.columns = [col.replace(',', '') for col in df.columns]
            df.columns = [col.replace('"', '') for col in df.columns]
            df.columns = [col.replace(' ', '') for col in df.columns]
            df.rename(columns={
                'Series/Group': 'collection',
                'SlateStart': 'start',
                'SlateEnd': 'end',
                'WritingTypes': 'type',
                'Recorded/Digital': 'digital',
                'formatofmostoftheinformation': 'format-summary',
                'Anythingmovingonscreenduringslate?': 'moving-elements'
            }, inplace=True)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip().str.replace(r'\s+', ' ')
            for _, row in df.iterrows():
                guid = row['GUID']
                csv_filename = f"{guid}.csv"
                csv_filepath = os.path.join(output_directory, csv_filename)
                os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
                row_df = pd.DataFrame([row[desired_columns]])
                is_annotated = False
                has_slate = False
                for time_col in ['start', 'end']:
                    v = row_df[time_col].values[0]
                    if v.startswith('no'):
                        row_df[time_col] = 'NO'
                        is_annotated = True
                    elif ';' in v:
                        # standard format is 00:00:00.000, which is 12 characters
                        time, frnum = v.split(';')
                        millisecs = int(frnum[:2]) / 30 * 1000
                        row_df[time_col] = f'{time}.{millisecs:03.0f}'
                        is_annotated = True
                        has_slate = True
                    elif ':' in v:
                        row_df[time_col] = f'{v}.000'
                        is_annotated = True
                        has_slate = True
                if not is_annotated:
                    print(f"Start or end time not annotated for {row_df['GUID'].values[0]}")
                    continue
                v = row_df['type'].values[0]
                if not has_slate:
                    row_df['type'] = '-'
                elif v.lower().startswith('hand') or v.lower().startswith('fixed'):
                    row_df['type'] = 'h'
                elif v.lower().startswith('type'):
                    row_df['type'] = 't'
                else:
                    raise ValueError(f"Unknown type: {v}")
                v = row_df['digital'].values[0]
                if not has_slate:
                    row_df['digital'] = False
                elif v.lower().startswith('digit'):
                    row_df['digital'] = True
                elif v.lower().startswith('record'):
                    row_df['digital'] = False
                else:
                    raise ValueError(f"Unknown type: {v}")
                row_df.to_csv(csv_filepath, index=False, sep=',')


if __name__ == '__main__':
    root_dir = pathlib.Path(__file__).parent
    for batch_dir in root_dir.glob('*'):
        if batch_dir.is_dir() and len(batch_dir.name) > 7 and batch_dir.name[6] == '-' and all([c.isdigit() for c in batch_dir.name[:6]]):
            print(f'Processing {batch_dir.name}...')
            process_csv(batch_dir.name, root_dir / 'golds')



```
