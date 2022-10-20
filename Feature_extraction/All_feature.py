import sys

sys.path.append("..")
from dct_histogram import extract_all_component_feature
import numpy as np

if __name__ == '__main__':
    dct_histogram_feature = extract_all_component_feature(QF=90)
    print('finish save features.')
