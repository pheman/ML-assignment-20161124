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
    
def core_code(round, year, cbgRatio, wlbgRatio):

      
        tranmatrix.loc['ROS','CBG'] = sm.Symbol('ros_cbg_'+dbc)
        lockCBGRos = list(solveset( Eq(tranmatrix.loc['EBIT','CBG'], tranmatrix.loc['净销售收入','CBG'] - tranmatrix.loc['销售成本','CBG'] - tranmatrix.loc['期间费用','CBG'] ), tranmatrix.loc['ROS','CBG']))[0]

    dfWLDelta  = (dfReport['WL_ROS']   - dbcinfo.loc[dbcList,'泛网络分销的ROS-median'])**2 * dfReport['泛网络收入']**2 *dfReport['CIT Rate']**2
    dfCBGDelta = (dfReport['CBG_ROS']  - dbcinfo.loc[dbcList,'CBG分销的ROS-median'])**2 * dfReport['终端收入'] **2 *dfReport['CIT Rate']**2
    dfHubDelta = (dfReportHub['ROS'] - [0.025,0.01,0.01,0.02,0.02])**2 * dfReportHub['收入']

    dfDelta = dfWLDelta.append(dfCBGDelta) #.append(dfHubDelta)
    dfDeltaSum = dfDelta.sum()

    deltaTaxFun = lambdify([list(dfROS) ], deltaTax.sum(), modules='numpy')

    jacopt = [sm.diff(dfDeltaSum,x) for x in dfROS]
    jacopt = [lambdify([list(dfROS) ], y, modules='numpy') for y in jacopt]
    def jacfun(x):
        return np.array([f(x) for f in jacopt])

    ## 用集团经营净利润## 用集团经营净利润## 用集团经营净利润## 用集团经营净利润## 用集团经营净利润## 用集团经营净利润
    cbgwlbgRatio = dbcbudget.swaplevel().loc['贡献利润']['消费者'].sum() / (dbcbudget.swaplevel().loc['贡献利润']['消费者'].sum() + dbcbudget.swaplevel().loc['贡献利润']['泛网络'].sum())
    consCoreWL  = lambdify([list(dfROS) ], ratioCoreWL  - wlbgRatio)
    consCoreCBG = lambdify([list(dfROS) ], ratioCoreCBG - cbgRatio)
   
    cons = [consCoreWL, consCoreCBG]
    jaccon = [[sm.diff(ratioCoreWL,x) for x in dfROS],[sm.diff(ratioCoreCBG,x) for x in dfROS]]
    jaccon = [[lambdify([list(dfROS) ],y, modules='numpy') for y in x] for x in jaccon  ]
    def jacconfun(x):
        return np.array( [ [fx(x) for fx in fc] for fc in jaccon] )

    eqcons = []
    for idx in dfROS.index[:-5]:
        dbc = idx.split('_')[0]  #idx[:-4] 
        if dbcinfo.loc[dbc,'代表处对应子公司类型'] == '服务型子公司':
            eqcons.append(lambdify([list(dfROS) ], dfROS[idx] - dbcinfo.loc[dbc,'泛网络分销的ROS-median']))
            
    lower = dbcinfo.loc[dbcList,'泛网络分销的ROS-Lower End'].append(
            dbcinfo.loc[dbcList,'CBG分销的ROS-Lower End']).append(
            dfReportHub['ROS']*0.0    + [0.024,0.009,0.009,0.019,0.019])

    upper = dbcinfo.loc[dbcList,'泛网络分销的ROS-Upper End'].append(
            dbcinfo.loc[dbcList,'CBG分销的ROS-Upper End']).append(
            dfReportHub['ROS']*0.0    + [0.026,0.011,0.011,0.021,0.021])
    initialBounds = pd.concat([lower, upper], axis=1)
    initialBounds.columns = ['lower', 'upper']
    initialBounds.index = dfROS.index

    x0 = (lower/2 + upper/2)
    optiFunc = lambdify([list(dfROS) ], dfDeltaSum, modules='numpy')

    bounds = list(zip( list(lower),list(upper)))
    
    if   round==1:
        res,funcvalue, iters, imode, smode = fmin_slsqp(optiFunc,x0=x0, bounds=bounds,fprime=jacfun, iter=1000,iprint=2,full_output=True)
    elif round==2:
        res,funcvalue, iters, imode, smode = fmin_slsqp(optiFunc, x0=x0, ieqcons=cons, eqcons=eqcons, bounds=bounds,fprime=jacfun, fprime_ieqcons=jacconfun, iter=1000,iprint=2,full_output=True)
    
   
    dfReportFunc = dfReport.apply(lambda x: x.apply(lambda y:lambdify([list(dfROS) ], y)))
    dfReportNum  = dfReportFunc.apply(lambda x: x.apply(lambda y:y(dfRes)))
    initial_col = list(dfReportNum.columns)
    dfReportNum['代表处类型'] = [dbcinfo.loc[dbc,'代表处对应子公司类型'] for dbc in dfReportNum.index]
    new_col = ['代表处类型'] + initial_col
    dfReportNum = dfReportNum.loc[:,new_col] 

    for dbc in dfReport.index:
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda x: x.apply(lambda y:lambdify([list(dfROS) ], y)))
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda x: x.apply(lambda y:y(dfRes)))
        
    for dbc in dfReportHub.index:
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda y:lambdify([list(dfROS) ], y))
        dbcallmatrix[dbc] = dbcallmatrix[dbc].apply(lambda y:y(dfRes))
    

    dfReportHubFunc = dfReportHub.apply(lambda x: x.apply(lambda y:lambdify([list(dfROS) ], y)))
    dfReportHubNum  = dfReportHubFunc.apply(lambda x: x.apply(lambda y:y(dfRes)))
    #%%
    
    return smode, gReport, consCoreCBG(dfRes)+cbgRatio, consCoreWL(dfRes)+ wlbgRatio, touchDict, riskTax
#%%
class WorkThread(QtCore.QThread):  
    trigger = QtCore.pyqtSignal(str, pd.DataFrame, float, float, dict, float)
    round = 1
    cbgRatio = 0.0
    wlbgRatio = 0.0
    def __int__(self):  
        super(WorkThread,self).__init__()  
  
    def run(self):     
        smode,gdfres, cbgRatioRes, wlbgRatioRes, touchDict, riskTax = core_code(self.round, '2017', self.cbgRatio, self.wlbgRatio)
        print(self.round)
        self.trigger.emit(smode, gdfres, cbgRatioRes, wlbgRatioRes, touchDict, riskTax)         #完毕后发出信号
        
class taxPlan(QtWidgets.QMainWindow, Ui_MainWindow): 
    _signal = QtCore.pyqtSignal()
    cbgRatio = 0.0
    wlbgRatio = 0.0
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
        self.thread.cbgRatio  = float( self.cbg_domes_ratio.toPlainText() )
        self.thread.wlbgRatio = float( self.wlbg_domes_ratio.toPlainText())
        print(self.cbg_domes_ratio.toPlainText())
        print(self.wlbg_domes_ratio.toPlainText())
        
        self.thread.trigger.connect(self.finished)
        self.thread.start()
    def func_round2(self):
        self.resultWindow.setText('Round2 Start' + '\n')
        self.thread  = WorkThread()
        self.thread.round = 2
        self.thread.cbgRatio  = float( self.cbg_domes_ratio.toPlainText() )
        self.thread.wlbgRatio = float( self.wlbg_domes_ratio.toPlainText())
        print(self.cbg_domes_ratio.toPlainText())
        print(self.wlbg_domes_ratio.toPlainText())
        
        self.thread.trigger.connect(self.finished)
        self.thread.start()
    def finished(self, smode, gdfres, cbgRatioRes, wlbgRatioRes, touchDict, riskTax):
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
           self.resultWindow.append('国内利润占比， CBG:%s, 泛网络:%s\n'%(format(cbgRatioRes,'.2%'),format(wlbgRatioRes,'.2%')) )
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

