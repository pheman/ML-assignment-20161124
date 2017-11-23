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



pd.set_option('mode.chained_assignment','raise')
zhfont = FontProperties()
zhfont.set_family('SimHei')

def core_code(round, year, gbcrat, wlbgrat):

      
    b = sm.Symbol('a')
    d = list(solveset( Eq(a, b ), c))[0]
    
    dfDeltaSum = dfDelta.sum()

    deltaTaxFun = lambdify([list(dfSOR) ], deltaTax.sum(), modules='numpy')

    jacopt = [sm.diff(func,x) for x in dfr_o_s]
    jacopt = [lambdify([list(dfr_o_s) ], y, modules='numpy') for y in jacopt]
    def jacfun(x):
        return np.array([f(x) for f in jacopt])
   
    cons = [a, b]
    jaccon = [[sm.diff(ratCoreWL,x) for x in dfSOR],[sm.diff(ratCoregbc,x) for x in dfSOR]]
    jaccon = [[lambdify([list(dfSOR) ],y, modules='numpy') for y in x] for x in jaccon  ]
    def jacconfun(x):
        return np.array( [ [fx(x) for fx in fc] for fc in jaccon] )

    eqcons = []
    for idx in dfSOR.index[:-5]:
        dbc = idx.split('_')[0]  #idx[:-4] 
        
        eqcons.append(lambdify([list(dfSOR) ], dfSOR[idx] - dbcinfo.loc[dbc,'median']))
            
    
    x0 = (lower/2 + upper/2)
    optiFunc = lambdify([list(dfSOR) ], dfDeltaSum, modules='numpy')

    bounds = list(zip( list(lower),list(upper)))
    
    if   round==1:
        res,funcvalue, iters, imode, smode = fmin_slsqp(optiFunc,x0=x0, bounds=bounds,fprime=jacfun, iter=1000,iprint=2,full_output=True)
    elif round==2:
        res,funcvalue, iters, imode, smode = fmin_slsqp(optiFunc, x0=x0, ieqcons=cons, eqcons=eqcons, bounds=bounds,fprime=jacfun, fprime_ieqcons=jacconfun, iter=1000,iprint=2,full_output=True)
    
   
    dfReportFunc = dfReport.apply(lambda x: x.apply(lambda y:lambdify([list(dfSOR) ], y)))
    dfReportNum  = dfReportFunc.apply(lambda x: x.apply(lambda y:y(dfRes)))
    

    for dbc in dfReport.index:
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda x: x.apply(lambda y:lambdify([list(dfSOR) ], y)))
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda x: x.apply(lambda y:y(dfRes)))
        
    for dbc in dfReportHub.index:
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda y:lambdify([list(dfSOR) ], y))
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda y:y(dfRes))
    

    dfReportHubFunc = dfReportHub.apply(lambda x: x.apply(lambda y:lambdify([list(dfSOR) ], y)))
    dfReportHubNum  = dfReportHubFunc.apply(lambda x: x.apply(lambda y:y(dfRes)))
    #%%
    
    return smode, gReport, consCoregbc(dfRes)+gbcrat, consCoreWL(dfRes)+ wlbgrat, touchDict, riskTax
#%%
class WorkThread(QtCore.QThread):  
    trigger = QtCore.pyqtSignal(str, pd.DataFrame, float, float, dict, float)
    round = 1
    gbcrat = 0.0
    wlbgrat = 0.0
    def __int__(self):  
        super(WorkThread,self).__init__()  
  
    def run(self):     
        smode,gdfres, gbcratRes, wlbgratRes, touchDict, riskTax = core_code(self.round, '2017', self.gbcrat, self.wlbgrat)
        print(self.round)
        self.trigger.emit(smode, gdfres, gbcratRes, wlbgratRes, touchDict, riskTax)         #完毕后发出信号
        
class taxPlan(QtWidgets.QMainWindow, Ui_MainWindow): 
    _signal = QtCore.pyqtSignal()
    gbcrat = 0.0
    wlbgrat = 0.0
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        self.round1.clicked.connect(self.func_round1)
        self.round2.clicked.connect(self.func_round2)
        self.round3.clicked.connect(self.func_round3)
        
    def func_round1(self):
        self.resultWindow.setText('Round1 Start' + '\n')
        self.thread  = WorkThread()
        self.thread.round = 1
        self.thread.gbcrat  = float( self.gbc_domes_rat.toPlainText() )
        self.thread.wlbgrat = float( self.wlbg_domes_rat.toPlainText())
        print(self.gbc_domes_rat.toPlainText())
        print(self.wlbg_domes_rat.toPlainText())
        
        self.thread.trigger.connect(self.finished)
        self.thread.start()
    def func_round2(self):
        self.resultWindow.setText('Round2 Start' + '\n')
        self.thread  = WorkThread()
        self.thread.round = 2
        self.thread.gbcrat  = float( self.gbc_domes_rat.toPlainText() )
        self.thread.wlbgrat = float( self.wlbg_domes_rat.toPlainText())
        print(self.gbc_domes_rat.toPlainText())
        print(self.wlbg_domes_rat.toPlainText())
        
        self.thread.trigger.connect(self.finished)
        self.thread.start()
    def finished(self, smode, gdfres, gbcratRes, wlbgratRes, touchDict, riskTax):
        print(smode)
        print(gdfres)
        print(type(gdfres))

        #self.resultWindow.append(smode)
        if smode != 'Optimization terminated successfully.':
           self.outputTable.setColumnCount(0)
           self.outputTable.setRowCount(0)
           self.resultWindow.append('Round 2 failed, please run round3' ) 
           
        else:
           self.resultWindow.append(smode)
           
           for key, item in touchDict.items():
               self.resultWindow.append(key + ' '*(20 - len(key)) +item)
           self.resultWindow.append('Total Tax Risk is'+ format(float(format(riskTax,'.2f')),',') )
           print(list(gdfres.columns))
           print(list(gdfres.index))
           
           self.outputTable.setColumnCount(len(gdfres.columns))
           self.outputTable.setRowCount(len(gdfres.index))
           self.outputTable.setHorizontalHeaderLabels([str(x) for x in gdfres.columns])
           self.outputTable.setVerticalHeaderLabels([str(x) for x in gdfres.index])
           
           for col in range(len(gdfres.columns)-1):
               for idx in range(len(gdfres.index)):
                   self.outputTable.setItem(idx,col , QTableWidgetItem(format(float(format(gdfres.iloc[idx,col],'.2f')),',') ))
           for idx in range(len(gdfres.index)):
               col = len(gdfres.columns) - 1
               self.outputTable.setItem(idx,col , QTableWidgetItem(format(gdfres.iloc[idx,col],'.2%')))

        print('Tables Written')
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    workTread = WorkThread()
    window = taxPlan()
    window.show()
    sys.exit(app.exec_())

