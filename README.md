# The CLAMS Dashboard

Experiments with a CLAMS dashboard, starting with an annotation viewer and an evaluation viewer.


### Requirements

Install the necessary Python modules:

```shell
pip install streamlit==1.34.0
pip install directory-tree==1.0.0
```

Or use the requirements file:

```shell
pip install -f requirements.txt
```

The dashboard expects access to a couple of repositories, you should get a local copy of them:

<!--
<img src="docs/workflows/dashboard-annotations.png" width=500>
<img src="docs/workflows/dashboard-evaluation.png" width=500>
-->

- [https://github.com/clamsproject/aapb-annotations](https://github.com/clamsproject/aapb-annotations)
- [https://github.com/clamsproject/aapb-evaluations](https://github.com/clamsproject/aapb-evaluations)
- [https://github.com/clamsproject/aapb-collaboration](https://github.com/clamsproject/aapb-collaboration)

Once these repositories are available on your local machine, edit the following lines in `code/config.py` if needed:

```python
# The default is that source files, annotations and evaluations are in a sister repository.
SOURCES = '../../wgbh-collaboration/'
ANNOTATIONS = '../../aapb-annotations/'
EVALUATIONS = '../../aapb-evaluations/'
```


### Running the Dashboard

To start the Streamlit application:

```shell
streamlit run app.py
```


### Testing the static site locally

> This is for functionality not yet existing as of September 20, 2024, namely the option to export a static Markdown site.

For development you may also want to use a Markdown editor or use the grip package, which allows you to test a site locally.

```shell
pip install grip==4.6.2
```

Then go to the directory with the main readme.md file and do

```shell
$ grip -b
```

The site then runs at [http://localhost:6419](http://localhost:6419), which is opened automatically due to the -b option.

The documentation at [https://github.com/joeyespo/grip](https://github.com/joeyespo/grip) claims that it uses the [GitHub markdown API](https://developer.github.com/v3/markdown) and that it should render the site as it would appear on GitHub. However, in many cases that actually does not happen because of some installation issues. I got the problem on Mac OS 12.7.5 and it seems related to the Ubuntu issue at [https://github.com/joeyespo/grip/issues/333]. The solution offered there worked for me:

```shell
git clone https://github.com/joeyespo/grip.git
cd grip/grip
python3 __main__.py -b /path/to/your/README.md
```

I suspect that using `brew install grip` may also work.
