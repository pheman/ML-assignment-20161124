import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sympy as sm
from sympy import lambdify,solveset,Eq
from scipy.optimize import fmin_slsqp
import time

import sys
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QWidget, QTableWidget, QHBoxLayout, QApplication, QDesktopWidget, QTableWidgetItem, QHeaderView

qtCreatorFile = "taxPlan.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
pd.set_option( 'mode.chained_assignment', 'raise')
zhfont = FontProperties()
zhfont.set_family('SimHei')
