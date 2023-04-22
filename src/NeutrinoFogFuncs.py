#===========================NeutrinoFogFuncs.py===================================#
# Created by Bing-Long Zhang, 2023

# Contains functions for performing calculations of the neutrino fog and floor, etc.
# All functions below are developed by modifying Ciaran's Python code.

#==============================================================================#
# import
import scipy as sc
import numpy as np
import time, multiprocessing
from numba import jit, float64
import matplotlib.pyplot as plt
from scipy import interpolate
#==============================================================================#
# Asymptotic-Analytic Method
@jit([(float64[:], float64[:,:], float64[:], float64)], nopython=True)
def phiGen1(s, b, sigmaTheta, exposure):
    n_nu = len(b)
    temp = s + np.sum(b, axis=0)
    derList = [[exposure*np.sum(b[i]*x/temp) for x in b[i+1:]] for i in range(n_nu-1)]
    G1 = exposure*np.sum(s*s/temp)
    G2 = exposure*np.array([np.sum(s*x/temp) for x in b])
    G3 = np.zeros((n_nu,n_nu))
    for i in range(0, n_nu-1):
        G3[i,i+1:] = derList[i]
        G3[i+1:, i] = derList[i]
    diagTerm = 1/(sigmaTheta*sigmaTheta)+exposure*np.array([np.sum(x*x/temp) for x in b])
    G3 = G3+np.diag(diagTerm)
    res = G1 - G2@np.linalg.inv(G3)@G2
    return res

# Quasi-Asimov Dataset Method
@jit([(float64[:], float64[:,:], float64[:], float64)], nopython=True)
def phiGen2(s, b, sigmaTheta, exposure):
    n = len(b) + 1
    sb = np.zeros((n,len(s)))
    sb[0] = s
    sb[1:] = b
    temp = np.sum(sb, axis=0)
    ijList = [[exposure*np.sum(sb[i]*sb[j]/temp) for j in range(i+1, n)] for i in range(n-1)]
    ldd = np.zeros((n,n))
    for i in range(0, n-1):
        ldd[i,i+1:] = ijList[i]
        ldd[i+1:, i] = ijList[i]
    diagTerm = np.append([0.],1/(sigmaTheta*sigmaTheta)) + exposure*np.array([np.sum(sb[i]*sb[i]/temp) for i in range(n)])
    ldd = ldd+np.diag(diagTerm)
    HMat = np.zeros((n,n))
    HMat[1:,1:] = np.linalg.inv(ldd[1:,1:])
    niuList = (-(np.append([1.],np.zeros(n-1)) - HMat@ldd@np.append([1.],np.zeros(n-1))))[1:] + 1.
    datObs = s + np.sum(b,axis=0)
    temp = np.sum(np.transpose(b)*niuList,axis=1)
    phi = 2*(-exposure*np.sum(datObs*(np.log(temp)-np.log(datObs))-(temp-datObs))+ 0.5*np.sum(((1-niuList)/sigmaTheta)**2))
    return phi

class Fog(object):
    
    def __init__(self, phiGen):
        self._phiGen = phiGen
        
    def _sigmaPhi(self, paraSet, sigma0Log):
        RWIMP, RNu, NuUncs, phi, exposure = paraSet
        temp = self._phiGen(RWIMP*10**(sigma0Log+45.), RNu, NuUncs, exposure)-phi
        if temp<0:
            return [0, temp]
        else:
            return [1, temp]

    def sigmaExpoListGen(self, exposureList, RWIMP, RNu, NuUncs, phi):
        step0 = 0.1*0.5
        [sigmaLog, temp, recordDat] = biSearch(1., 6, self._sigmaPhi, [RWIMP, RNu, NuUncs, phi, exposureList[0]], -48., 0)
        recordDatList = [recordDat]
        sigmaList = np.array([sigmaLog])
        for i in range(1, len(exposureList)):
            [sigmaLog, temp, recordDat] = biSearch(step0, 4, self._sigmaPhi, [RWIMP, RNu, NuUncs, phi, exposureList[i]], sigmaLog, 0)
            sigmaList = np.append(sigmaList, sigmaLog)
            recordDatList.append(recordDat)
        return [exposureList, 10**(sigmaList), recordDatList]

    def sigmaExpoListGen2(self, exposureInit, sigmaLogEnd, RWIMP, RNu, NuUncs, phi):
        step0 = 0.1*0.5
        [sigmaLog, temp, recordDat] = biSearch(1., 6, self._sigmaPhi, [RWIMP, RNu, NuUncs, phi, exposureInit], -48., 0)
        recordDatList = [recordDat]
        exposureList = np.array([exposureInit])
        sigmaList = np.array([sigmaLog])
        i = 0
        expoStep = 0.1
        while True:
            i = i + 1
            exposure = exposureInit*10**(i*expoStep)
            [sigmaLog, temp, recordDat] = biSearch(step0, 4, self._sigmaPhi, [RWIMP, RNu, NuUncs, phi, exposure], sigmaLog, 0)
            exposureList = np.append(exposureList, exposure)
            sigmaList = np.append(sigmaList, sigmaLog)
            recordDatList.append(recordDat)
            if sigmaLog < sigmaLogEnd:
                break
        return [exposureList, 10**(sigmaList), recordDatList]

#==============================================================================#
# Useful functions
def biSearch(step0, n, func, paraSet, para0, flag):
    [temp,para] = [step0,para0]
    tempRes = func(paraSet, para)
    recordPara = np.array([para])
    recordRes = np.array([tempRes[1]])
    if tempRes[0]==flag:
        [sign, label] = [1,0]
    else:
        [sign, label] = [-1,1]
    para = para+sign*temp;
    boundPara = para0+sign*100*step0
    
    for i in range(0,n):
        for j in range(0,100):
            if sign > 0 and (para>boundPara or abs(para-boundPara)<1e-5):
                para = para - 0.5*sign*temp
                break
            elif sign < 0 and (para<boundPara or abs(para-boundPara)<1e-5):
                para = para - 0.5*sign*temp
                break
            tempRes = func(paraSet, para)
            recordPara = np.append(recordPara, para)
            recordRes = np.append(recordRes, tempRes[1])
            if tempRes[0] != label:
                boundPara = para
                para = para - 0.5*sign*temp
                break
            para = para + sign*temp
        temp = 0.5*temp
    return [para, temp, np.array([recordPara,recordRes])]

def myFindRoot(dat):
    x1, x2, y1, y2 = dat[0,-2], dat[0,-1], dat[1,-2], dat[1,-1]
    a = (y2-y1)/(x2-x1)
    b = (y1*x2-y2*x1)/(x2-x1)
    if a==0:
        res = (x1+x2)/2
    else:
        res = -b/a
    return res

def findPoint(dat):
    x0 = dat[0]
    y0 = 10**(np.array(list(map(lambda i: myFindRoot(dat[2][i]), range(len(dat[2]))))))
    x1 = 1/2*(x0[1:]+x0[:-1])
    y1 = -(x1/(1/2*(y0[1:]+y0[:-1]))*(y0[1:]-y0[:-1])/(x0[1:]-x0[:-1]))**(-1)
    y0 = 1/2*(y0[1:]+y0[:-1])
    if y1[0]>2:
        print("The first element exceed 2.")
        return [0,0]
    label = 0
    for i in range(len(y1)):
        if y1[i]>2:
            label = i
            break
    if label > 0:
        temp = np.array([x1[label-1], x1[label], y1[label-1]-2., y1[label]-2., y0[label-1], y0[label]])
        return myFindPoint(temp)
    else:
        print("No elements can exceed 2.")
        return [0,0]

def myFindPoint(dat):
    x1, x2, y1, y2 = dat[0:4]
    a = (y2-y1)/(x2-x1)
    b = (y1*x2-y2*x1)/(x2-x1)
    x0 = -b/a
    x1, x2, y1, y2 = np.append(dat[0:2], dat[4:])
    a = (y2-y1)/(x2-x1)
    b = (y1*x2-y2*x1)/(x2-x1)
    return [x0, a*x0+b]

#==============================================================================#
# Functions for generating tha data for plots

def sigmaNDat(dat):
    x0 = dat[0]
    y0 = 10**(np.array(list(map(lambda i: myFindRoot(dat[2][i]), range(len(dat[2]))))))
    x1 = 1/2*(x0[1:]+x0[:-1])
    y1 = -(x1/(1/2*(y0[1:]+y0[:-1]))*(y0[1:]-y0[:-1])/(x0[1:]-x0[:-1]))**(-1)
    y0 = 1/2*(y0[1:]+y0[:-1])
    return [np.log10(y0), y1]

#To avoid some numerical fluctuations when exposures are too large
def intDatCut(dat):
    i = 30
    n = len(dat[0])
    x, y = dat[0][-i:], dat[1][-i:]
    temp = np.argwhere(y<2.)
    if len(temp)==0:
        return dat
    else:
        return [dat[0][:n-i+temp[0,0]],dat[1][:n-i+temp[0,0]]]

def plotDatGen(massList, dat):
    intDatList = list(map(sigmaNDat, dat))
    intDatList = list(map(intDatCut, intDatList))
    intDatBoundsList = list(map(lambda x: [x[0].min(),x[0].max()],intDatList))
    intFuncList = list(map(lambda x: interpolate.interp1d(x[0],x[1]),intDatList))
    sigRange = np.linspace(-50,-39,500)
    def intFuncBounds(intFunc, var, bounds):
        if var>bounds[1]:
            return 1.
        elif var<bounds[0]:
            return 2.
        else:
            return intFunc(var)
    def subFunc(i):
        nList = np.array(list(map(lambda x: intFuncBounds(intFuncList[i], x, intDatBoundsList[i]), sigRange)))
        return nList
    m, sig = np.meshgrid(massList, sigRange)
    n = np.transpose(list(map(subFunc, range(0,len(massList)))))
    return m, 10**sig, n

def plotDatGen2(massList, dat):
    intDatList = list(map(sigmaNDat, dat))
    intDatList = list(map(intDatCut, intDatList))
    intDatBoundsList = list(map(lambda x: [x[0].min(),x[0].max()],intDatList))
    intFuncList = list(map(lambda x: interpolate.interp1d(x[0],x[1]),intDatList))
    sigRange = np.linspace(-50.,-39,500)
    def intFuncBounds(intFunc, var, bounds):
        if var>bounds[1]:
            return 0.
        elif var<bounds[0]:
            return 2.
        else:
            temp = intFunc(var)
            if temp > 3.:
                temp2 = 3.2
            else:
                temp2 = temp
            return temp2
    def subFunc(i):
        nList = np.array(list(map(lambda x: intFuncBounds(intFuncList[i], x, intDatBoundsList[i]), sigRange)))
        return nList
    m, sig = np.meshgrid(massList, sigRange)
    n = np.transpose(list(map(subFunc, range(0,len(massList)))))
    return massList, 10**sigRange, n

#==============================================================================#
# Functions for evaluating impacts from different parameters
def matOneHalfGen(mat):
    va, ve = np.linalg.eigh(mat)
    return [ve@np.diag(1/va)@np.transpose(ve), ve@np.diag(np.sqrt(va))@np.transpose(ve)]

def symMatGen(diagTerms, ijTerms):
    n = len(diagTerms)
    myMat = np.zeros((n,n))
    for i in range(0, n-1):
        myMat[i, i+1:] = ijTerms[i]
        myMat[i+1:, i] = ijTerms[i]
    myMat = myMat + np.diag(diagTerms)
    return myMat

def phiGen3(s, b, sigmaTheta, exposure):
    sb = np.append([s],b,axis=0)
    sigmaTheta2 = sigmaTheta**2
    n = len(sb)
    
    vList = np.sum(sb, axis=0)
    vInvList = 1/vList
    deriList = sb
    deriNuListT = np.transpose(deriList[1:])
    
    ijList = [[np.sum(deriList[i]*deriList[j]*vInvList) for j in range(i+1,n)] for i in range(0,n-1)]
    diagTerms = np.array([np.sum(deriList[i]*deriList[i]*vInvList) for i in range(0,n)])
    lddTemp = symMatGen(diagTerms, ijList)
    ldd = exposure*lddTemp + np.diag(np.append([0.], 1/(sigmaTheta2)))
    
    nBin = len(vList)
    ijTerms = [[np.sum(deriNuListT[i]*deriNuListT[j]*sigmaTheta2) for j in range(i+1,nBin)] for i in range(0,nBin-1)]
    diagTerms = [np.sum(deriNuListT[i]*deriNuListT[i]*sigmaTheta2) for i in range(0,nBin)]
    varGauTermConst = np.outer(vInvList, vInvList)*symMatGen(diagTerms, ijTerms)
    varGauTermIJ = [[np.sum(np.outer(deriList[i], deriList[j])*varGauTermConst)\
                             for j in range(i+1,n)] for i in range(0,n-1)]
    varGauTermDiag = [np.sum(np.outer(deriList[i], deriList[i])*varGauTermConst)\
                             for i in range(0,n)]
    varGauTerm = symMatGen(varGauTermDiag, varGauTermIJ)
    varMat = exposure*lddTemp + exposure**2*varGauTerm
    
    G3 = ldd.copy()[1:,1:]
    HMat = np.block([[np.zeros((1,1)), np.zeros((1,len(b)))],[np.zeros((len(b),1)), np.linalg.inv(G3)]])
    lddInv, lddOneHalf = matOneHalfGen(ldd)
    varMatInv, varMatOneHalf = matOneHalfGen(varMat)
    varMatOneHalfInv = np.linalg.inv(varMatOneHalf)
    tmat = lddInv - HMat
    
    res1 = varMatOneHalf@(lddInv - HMat)@varMatOneHalf
    res2 = np.linalg.eigh(res1)
    orderList = np.flip(np.argsort(res2[0]))
    res2 = [res2[0][orderList], np.transpose(res2[1])[orderList]]
    res3 = varMatOneHalfInv@ldd@np.append([1.],np.zeros(len(b)))
    return [res2[0][0:2], (res2[1][0]@res3)**2, res2[1][0]]