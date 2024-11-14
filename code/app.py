"""CLAMS Dashboard

Top-level application code.

To run this:

$ streamlit run app.py
$ streamlit run app.py -- OPTION+

The second invocation is meant to send options to the application. We have the
following options:

debug
    This for when the appliation runs in debug mode, that is, when you select
    "Always rerun" after a code update. These reruns do not impact the session
    state I think which causes some code updates to not be reflected in the new
    interface, with the debug option annotations and evaluations will be reloaded.

no-checkout
    Do not checkout a branch in the annotation/evaluation repository. This is
    useful when experimenting with the code on changes in the repositories that
    were not committed. Also, when your annotation/evaluation repository has 
    local changes then checkout will fail.

"""

import json
import sys

import pandas as pd
import streamlit as st

import config
import model
import utils
from viewers.annotation_viewer import viewer as annotation_viewer
from viewers.evaluation_viewer import viewer as evaluation_viewer


# No debugging by default, this can be overwritten by handing an argument to the
# script, as explained in the doc string.
DEBUG = True if 'debug' in sys.argv[1:] else False

# Checking out a branch by default, when False it will just access what's in the
# working directory
CHECKOUT = False if 'no-checkout' in sys.argv[1:] else True

if 'MODEL' not in st.session_state or DEBUG:
    # also regenerating data when in debug mode
    st.session_state['MODEL'] = model.Data(config.ANNOTATIONS, config.EVALUATIONS)
MODEL = st.session_state['MODEL']


def debug(text: str):
    if DEBUG:
        print(text)


st.set_page_config(
    page_title='CLAMS Dashboard',
    layout="wide")

st.markdown(utils.style, unsafe_allow_html=True)


dashboard = st.sidebar.radio(
    'dashboard',
    ['Overview', 'Annotation viewer', 'Evaluation viewer'],
    label_visibility='hidden')


if dashboard == 'Overview':

    st.title('CLAMS Dashboard')
    st.info(
        'These pages contain the CLAMS Viewers, which provide ways to browse ' +
        'the annotations and evaluations repositories. \n\n')
    st.markdown('##### The homepages for the repositories are available at')
    st.page_link(
        'https://github.com/clamsproject/aapb-annotations',
        icon="🏠",
        label='AAPB Annotations GitHub Repository')
    st.page_link(
        'https://github.com/clamsproject/aapb-evaluations',
        icon="🏠",
        label='AAPB Evaluations GitHub Repository')
    st.markdown('The code for this dashboard is maintained at '
                + '[https://github.com/clamsproject/dashboard]'
                + '(https://github.com/clamsproject/dashboard)')

elif dashboard == 'Annotation viewer':

    annotation_viewer(MODEL, CHECKOUT)

elif dashboard == 'Evaluation viewer':

    evaluation_viewer(MODEL)
