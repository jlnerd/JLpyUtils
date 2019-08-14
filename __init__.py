__author__ = "John T. Leonard, Display Investigation, Apple Inc."
__version__ = "2019.04.28"
__maintainer__ = "John T. Leonard"
__email__ = "john_t_leonard@apple.com"
__date__ = "2019.04.28"


import sys, os

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0,  os.path.dirname(os.path.abspath(__file__)))
    
import plot
import strings
import summary_tables
import img
import JL_ML_models as ML_models

print('JLpy_utils_package mounted (repo: https://github.com/jlnerd/JLpy_utils_package.git)')