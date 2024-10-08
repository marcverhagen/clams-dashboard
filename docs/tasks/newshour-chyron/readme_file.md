[Dashboard](../../index.md)  &nbsp; > &nbsp; [Tasks](../index.md)  &nbsp; > &nbsp; **Task** 

# Annotation Task &nbsp; ⎯ &nbsp; newshour-chyron &nbsp; ⎯ &nbsp; Readme

[task](index.md) | **readme** | [gold files](golds.md) | [data drops](drops/index.md) | [batches](batches.md) | [script](script.md) 

# Chyrons

## Project Overview
This project provides a dataset used for the detection of chyrons/banner-text that appears onscreen in video media.  
[Chyron](https://www.merriam-webster.com/dictionary/chyron) - a caption superimposed over *usually* the lower part of a video image (as during a news broadcast). Note, not neccesarily lower third. Sometimes they are temporary and will fade/blink in and out. 
Named after the company that made this technology. aka: the text that appears under people during broadcasts.  
### Specs
* Annotation project name - `newshour-chyron`
* Annotator Demographics
    * Number of annotators - 1 
    * Occupational description - College Student
    * Age range - 20s
    * Education - College
* Annotation Environment information
    * Name - VIA
    * Version - Unknown
    * Link/Tool Used/User Manual - (See below Tool Installation)
* Project changes
    * Number of Batches - 1, the batch is named "batch2", but there seems to be only one batch for this project.
    * Other Version Control Information - Unknown/1.0

## Tool Installation: VIA tool
_The following section is recalled from memory and may not be fully accurate._  
[VIA3](https://www.robots.ox.ac.uk/~vgg/software/via/). This project used [this version](https://github.com/kelleyl/clams-via3/blob/master/app.py#L16) and the Video Annotator tool.  
A video must be imported. Docker is likely used.  

## Annotation Guidelines
> [!Note]  
> This `readme.md` is the guideline for this project. 

### Preparation
Videos are loaded into the VIA tool for annotation. 
The annotator watches a video using the annotation tool, and labels chyrons in time.  

### What to Annotate
Usually, chyrons will fade in and out.  
* **START TIME** - When a chyron has fully appeared/solid and is no longer see through, pause the video and highlight the chyron via a bounding box. It's not too dissimilar from taking a screenshot.  
* **TEXT** - When the chyron is highlighted, double-click it and in the DESCRIPTION box, copy/write out what was said in the chyron exactly as it was displayed. When a chyron consists of more than one line, use the enter/return key (becomes \n in the gold output) to make a new line in the description.  
* **END TIME** - After that, submit how long the chyron was onscreen: annotating the end time as when it begins to become transparent.  
_Time format: (sec.ten-thousandths of sec)_   

### How to Annotate It
Please see VIA link for tool usage on step by step how to use the tool. 

### Decisions, Differentiation, and Precision Level during Annotation
Some decisions within the annotation process are described above.  

#### Precision Details

* **Boundaries** - This project was done with the annotation time as a subinterval. 
This means the annotation will be fully within the time/frame where the chyron is occurring and is annotated as lasting up to while the chyron is still onscreen. 
To repeat above, the choice was made to annotate the chyron as the first moment it stopped being transitory/fading-in, and the moment right before it begins to transition-out/fade-out. 
(This is currently unvalidated.)  
> [!Note]  
> This is opposite the decision made for the Slates project.  

* **Lack of Fine-Tuned Controls** - There is some lack of granularity in the annotator's creation. Even at slower speeds, precise accuracy beyond .1-.2sec is suspect. 
The project data here has divisions of seconds of ".5271, .7771, .2371". 
It is assumed this is some default in the tool, and therefore, precision of the annotation beyond xx.x is not recorded.  

* **Margin of Error** - As per [this](https://github.com/clamsproject/aapb-annotations/blob/main/repository_level_conventions.md), moving through the video with a video player can lead to some imprecision. 
Therefore, while the decision was followed above, it is possible that some data will not be entirely exact to the subinterval concept.

#### Other Notes  

* **Raws Time format** -
This project was done with an outside tool which outputted time as this format: `sec.ten-thousandths of sec`.

_The golds data now conforms to fieldname and time conventions._  

## Data Format and `process.py`
### `raw` data
`.json` file of the annotations from one batch as two files. It is unclear exactly what the division is for. 
See [batch2 comments](https://github.com/clamsproject/aapb-annotations/issues/24#issuecomment-1638870043) for more information. 
(See one of the [files](https://github.com/clamsproject/aapb-annotations/blob/feaf342477fc27e57dcdcbb74c067aba4a02e40d/newshour-chyron/220701-batch2/3a054b38_18Jul2022_16h18m12s.json) for format.)
 
### [`process.py`](process.py)
This script takes the raw data and converts it into a more usable format, by
1. reorganizing the chyron results into files based on video_filename/GUID.
2. removing extra columns; leaving only start, end, text, then adding entry index, sorted by start time.

### `golds` data
`.csv` file of which chyrons appear within one GUID/media. Row are sorted by values of `start` column. 
* Fields:
    * `index` - 1-indexed integer number for each chyron instances
    * `start` - renamed from z[0] or z[1]. Whichever was earlier. 
    * `end` - renamed from z[0] or z[1]. Whichever was later. 
    * `text` - from "text-boxes", what text was the chyron saying.

* Example:
```
index,start,end,text
1,00:04:35.777,00:04:39.777,JOHN BLOCK\nSecretary of Agriculture
2,00:05:56.777,00:05:59.527,REP. THOMAS P. O'NEILL\nSpeaker of the House
3,00:08:36.527,00:08:40.527,RITA LAVELLE\nFormer E.P.A. Official
4,00:12:35.527,00:12:40.027,WALTER WARD\nN.Y.C. Councilman
5,00:13:05.777,00:13:11.027,MAYOR ED KOCH
...
```
