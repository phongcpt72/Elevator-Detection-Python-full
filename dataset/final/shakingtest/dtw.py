import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtwalign import dtw
def Magnitude(tmp):
    return np.sqrt((np.square(tmp[0]))+(np.square(tmp[1]))+(np.square(tmp[2])))

def CalMagnitude(X,Y,Z):
    arrtmp = []
    for i in range(len(X)):
        a = X[i]
        b = Y[i]
        c = Z[i]
        tmp = [a,b,c]
        arrtmp.append(Magnitude(tmp))

    return arrtmp

def processdata(p, RMSAcc,value,i):
    valuearr = []
    IDarr =[]
    parr = []
    tmprmsraw = []
    while i < len(p):
        parr.append(p[i])
        tmprmsraw.append(RMSAcc[i])
        valuearr.append(value[i])
        i+=1
    j=0
    sums = 0
    while j < len(parr):
        IDarr.append(sums)
        sums = round(sums+0.04,2)
        j+=1
    return  IDarr, parr, tmprmsraw, valuearr

def calLPF(alpha, value, tmparr, info):
    if len(tmparr) == 0:
        va = alpha * value;
        tmparr.append(va)
    else:
        va = alpha * value + (1-alpha) * tmparr[info-1]
        tmparr.append(va)
    return va

def calalpha(cutoff, fs):
    dt = 1/fs
    T  = 1/cutoff
    return round(dt/(T+dt),2)

def LPF(alpha,rawarr):
    i = 0
    tmparr = []
    lpfarr = []
    while i < len(rawarr):
        va = calLPF(alpha,rawarr[i],tmparr, i)
        i+=1
        lpfarr.append(va)
    return lpfarr

def process(total):
    Ax = total.Ax
    Ay = total.Ay
    Az = total.Az
    MagnitudeAcc = CalMagnitude(Ax, Ay, Az)
    P = total.P
    Type = total.Type

    IDaa, paa, rmsaccraw, Type = processdata(P, MagnitudeAcc, Type, 0)
    fs = 25
    cutoff = 1.5
    alpha = calalpha(cutoff, fs)
    plpf = LPF(alpha, paa)
    plpf = LPF(alpha, plpf)
    rmsacclpf = LPF(alpha, rmsaccraw)
    rmsacclpf = LPF(alpha, rmsacclpf)

    IDaa, pa, acc, typearr = processdata(plpf, rmsacclpf, Type, 300)

    return acc


if __name__ == '__main__':
    totaldung  = pd.read_csv('Downstair2.csv') #dung
    totalsai = pd.read_csv('Downstair4.csv') #sai

    magdung = np.array(process(totaldung))
    magsai = np.array(process(totalsai))


    res = dtw(magdung, magsai)

    print("dtw distance: {}".format(res.distance))
    print("dtw normalized distance: {}".format(res.normalized_distance))

    plt.plot(magdung, label="query")
    plt.plot(magsai, label="reference")
    plt.legend()
    plt.show()