######################################################
# Author:   Steffen
#
# Creation: 20180215
#
# Quick Description:
# Helper tools for NLP analysis
#
######################################################

import re
import pandas as pd

def read_list(path):
    df = pd.read_csv(path, encoding='latin-1', index_col=0, delimiter=';')
    return df