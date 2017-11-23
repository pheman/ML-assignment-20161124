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

def sumHub(dbcbudget, signratio, key, term):
    saleSum = 0.0
    for dbc in signratio.index:
        try:
            #print(dbc)
            saleSum = saleSum + dbcbudget.loc[dbc].loc[term,key] * signratio[dbc]
            #print(saleSum)
        except KeyError:
            print(dbc,'not in dbcbudget from sumHub')        
    return saleSum

pd.set_option('mode.chained_assignment','raise')
zhfont = FontProperties()
zhfont.set_family('SimHei')

def check_index(df1, df2):
    df1set = set(df1.index)
    df2set = set(df2.index)
    print('length =', len(df1), ' and ',len(df2))
    print('only in left ', df1set - df2set)
    print('only in right', df2set - df1set)
