[Dashboard](../../index.md)  &nbsp; > &nbsp; [Tasks](../index.md)  &nbsp; > &nbsp; **Task** 

# Annotation Task &nbsp; ⎯ &nbsp; january-slates &nbsp; ⎯ &nbsp; Readme

[task](index.md) | **readme** | [gold files](golds.md) | [data drops](drops/index.md) | [batches](batches.md) | [script](script.md) 

# Slates

## Project Overview

### What is a film slate? 
> "The term film slate is still used to reference the clapper board that appears on the screen when filming. Just like early clappers.  The slate is shown at the beginning of a take immediately prior to the commencement of action." - [Beverly Boy Productions](https://beverlyboy.com/filmmaking/what-does-slate-mean-in-film/#:~:text=The%20term%20film%20slate%20is,the%20term%20slate%20in%20film.).  

* [More information about slates](https://www.youtube.com/watch?v=Heg6kDxXZ8k&t=13).  
* [What goes on a slate](https://www.theblackandblue.com/2012/11/05/deciphering-film-slate-1/).  

This project creates a dataset that annotates where informational frames about the video are located timewise from many videos/GUIDs. 
This dataset is needed to power automatic detection of slate information in the video collection at the AAPB. 
Many of the collection pieces have incomplete or unverified information about what the video is. 
This detected information could be used to verify and update metadata about that collection piece.

### Specs
* Annotation project name - `january-slates`
* Annotator Demographics
    * Number of annotators - Two
    * Occupational description - Two Public Broadcasting Organization Media Librarians and Volunteers
    * Age range - 20s
    * Education - college or masters education
* Annotation Environment information
    * Name - Manual google sheets entry + AAPB video viewer + unknown video viewer with frame time 
    * Version - n/a
    * Link/Tool Used/User Manual - (See below Tool Installation)
* Project changes
    * Number of Batches - 1 
    * Other Version Control Information - None

## Tool Installation: None
This annotation was done manually by entering information into a Google Sheet. 
Multiple videos were prepared or downloaded or were accessed from an AAPB terminal. 
Videos were opened in the AAPB viewer or in some unknown tool that had time information up to division by 30 frames per second.  

## Annotation Guidelines
For a quick overview of [slate types](https://docs.google.com/document/d/1Xf43EpVzQbIOB-7KTadEyU3eam9xIvLlSGkjy4Ff2v4/edit) please see this.  
The verbal guidelines for this project were to annotate as a superinterval/superset times the slate appeared, and details about its appearance.  

### Preparation
A google sheet must be prepared to annotate the below columns.  
A set of video videos must be prepared to be opened for annotation, preferably openable in mass in a video viewer that has some fraction of a second denomination.

### What to Annotate
Per column:  
* `GUID` - the AAPB id for that video e.g. "cpb-aacip-81-881jx33t".
    > [!Note]
    > CPB is the [Corporation for Public Broadcasting](https://cpb.org/faq#1-1:~:text=Public%20Broadcasting%20(CPB)%3F-,CPB,-is%20a%20private).
    > "aacip" is likely a collection name (unverified).
    > The first number seems to be series number. Eg. 81 is "Woman", and "29" is both Prime Time Wisconsin" and "Wisconsin Week".
    > The final number which includes letters is the unique guid number.  
* `Series/Group` - what tv series or group this video belongs to.  
* `Slate Start` - When the slate starts appearing. (See Decisions). Format likely "hr:mn:se:fr" out of 30 fps.     
* `Slate End` - When the slate stops being shown on screen. (See Decisions)  
* `Writing Types` - Slates contain written information. Much of this material is from the early days of tv. eg. "handwritten", "typed" or "other"   
* `Recorded/Digital` - Whether the slate is recorded by camera or digitally encoded into the video. 
* `format of most of the information` - visually how is the textual information presented in the slate? e.g. "boxes to fill in", "key-value pairs", "free text"  
* `Anything moving on screen during slate?` - are there things like animations and other things that are moving during the slate information

### How to Annotate It 
Move along in the video until the start of the slate is seen. Annotate the time right before that, or 00:00:00.000 if it starts the video. 
Find the end of the slate, annotate the moment right after it fully disappears. 
Add the other information as needed. 

### Decisions, Differentiation, and Precision Level during Annotation
* **What denotes start and end** - This project was done with the annotation time as a superinterval.
This means the annotation will begin on a time/frame without the slate where possible (or 00:00:00.000) 
and is annotated as ending after the slate has disappeared.
_(This is currently unvalidated.)_
    > [!Warning]  
    > This is opposite to the decision made in the Chyrons project. 

* **Valid Annotation: No Slate** - there are valid instances where a video does not have slate information shown within the actual video. Annotate as "no slate" in both Slate Start and Slate End.

#### Notes about the `raws` data
* **Raws Data Entry Errors** - There a small typo in the raw data, in `CLAMS_slate_annotation_metadata.csv` line 203, there is a typo "typeed" instead of "typed" .
* **Raws time format** - Likely `hr:min:sc;fr` likely a semicolon followed by frame number (30 fps), with some possible typos (line 77, Slate End "00:00:27;110,", line 97 Slate Start "00:00:05;119,").
* **Time format change** - At around line203 is a note that `sammy started annotating here.`. 
Shift of time format to only 3 numbers: "xx:xx:xx". This is confirmed as hh:mm:ss (no milliseconds!) since the AAPB viewer used did not offer sub-seconds precision.
* **Time Precision** - Because of the time format change, it should be assumed that the numbers without frames is only precise down to the second.
The precision of the other annotations with frame precision is unverified. 

* **Skip to new material in raw** - The raw `.csv` file also has an area that is skipped and annotation moved onto new/different videos. 
It starts at line 624, and resumes at line 1409. This might be because all of those GUIDs are the same and more variety in the dataset was needed for better results.
Annotation by the 2nd annotator paused/ended with line 1672 being the last annotated row.

## Data format and `process.py`

### `raw` data
`.csv` file where each line is the time of when the slate frames appear in that video.
* Fields
    * `GUID`
    * `,`
    * `Series/Group              ,`
    * `Slate Start ,`
    * `Slate End   ,`
    * `Writing Types,`
    * `Recorded/Digital`
    * 
    * 
    * `,`
    * `format of most of the information`
    * `,`
    * `Anything moving on screen during slate?`
> [!Important]
> There are extra commas added in and extra blank columns and extra space within the column names.
> Also, in some columns, there is a _comma suffixed_ to the raw value of the column.
* Example:
    ```
    cpb-aacip-81-881jx33t,",","Woman                     ,","00:00:00;00 ,","00:00:05;04 ,","handwriting  ,",recorded,,,",",boxes to fill in,",",no,
    cpb-aacip-41-34fn32g7,",","Carolina Journal          ,","00:00:00;00 ,","00:00:14;28 ,","typed        ,",digital?,,,",",key-value pairs,",",countdown,
    ```

### [`process.py`](process.py)
This script takes the raw data and converts it into a more machine-ingestible format by:
1. removing the extra commas and spaces in the raw data,
2. taking out the extra comma suffixes in the raw data,
3. normalizing column names based on repository conventions,
4. normalizing some values, possibly fixing typos
    * normalizes `typeed` type equivalent to `typed`
    * normalizes timestamps with semicolons, converting only the first two digits after the semicolon to milliseconds, assuming 30 fps.
5. process.py and golds were updated on [23/12/14](https://github.com/clamsproject/aapb-annotations/pull/77) to follow golds field-naming conventions. 

### `golds` data
`.tsv` tabular format separated with tab (`U+0009`) characters. The gold files conform to the repository readme guideline that each gold must relate to only one GUID. Therefore, each of these gold files is only 1 video/GUID each.
* Fields:
    * `GUID` - the same as the raw data
    * `collection` - renamed from `Series/Group`
    * `start` - renamed from `Slate Start`
    * `end` -  renamed from `Slate End`
    * `type` - renamed from `Writing Types`, values are normalized to `h`, `t`, `o` for `handwritten`, `typed`, `other` respectively.
    * `digital` - renamed from `Recorded/Digital`, values are normalized to boolean `True` or `False` values, where `True` means the slate is digitally encoded.
    * `format-summary` - renamed from `format of most of the information`
    * `moving-elements` - renamed from `Anything moving on screen during slate?`
    * all other columns from the raw data are removed
* Example:
    ```
    $ cat golds/cpb-aacip-81-881jx33t.tsv
    GUID    collection      start   end     type    digital format-summary  moving-elements
    cpb-aacip-81-881jx33t   Woman   00:00:00.000    00:00:05.040    h       False   boxes to fill in no
    
    $ cat golds/cpb-aacip-41-34fn32g7.tsv
    GUID    collection      start   end     type    digital format-summary  moving-elements
    cpb-aacip-41-34fn32g7   Carolina Journal        00:00:00.000    00:00:14.280    t       True    key-value pairs countdown
    ```
    > [!Note] 
    > Each file has the column header in it.

This project's `golds` files conform to both conventions for field-name and time format.  
## See also 

### 2020 Evaluation Dataset   
> [Evaluation dataset](https://docs.google.com/spreadsheets/d/1VHEpYmAtBHkIHTzbYtUexRNqALEHLi-3rwzIXtfQG-E/edit#gid=0)  
 
In 2020, an evaluation of the performance of the slates app tool was done by GBH. This is the result of it, comparing the output 
prediction of the app to the judgment of a human annotator. The annotation generated for the evaluation process weren't used in this `january-slates` project. 
