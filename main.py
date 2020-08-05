import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from dtwalign import dtw
def RMS(tmp):
    return np.sqrt((np.square(tmp[0]))+(np.square(tmp[1]))+(np.square(tmp[2])))

def CalRMS(X,Y,Z):
    arrtmp = []
    for i in range(len(X)):
        a = X[i]
        b = Y[i]
        c = Z[i]
        tmp = [a,b,c]
        arrtmp.append(RMS(tmp))

    return arrtmp

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

def xulytim(total):
    Ax = total.Ax
    Ay = total.Ay
    Az = total.Az
    RMSAcc = CalRMS(Ax, Ay, Az)
    P = total.P
    Type = total.Type

    IDaa, paa, rmsaccraw, Type = processdata(P, RMSAcc, Type, 0)
    fs = 25
    cutoff = 1.5
    alpha = calalpha(cutoff, fs)
    tmp = [None for _ in range(len(paa))]
    plpf = LPF(alpha, paa)
    plpf = LPF(alpha, plpf)
    tmp = [None for _ in range(len(rmsaccraw))]
    rmsacclpf = LPF(alpha, rmsaccraw)
    rmsacclpf = LPF(alpha, rmsacclpf)

    IDaa, pa, acc, typearr = processdata(plpf, rmsacclpf, Type, 300)

    return IDaa, pa, acc, typearr

def overlap(N, arr, arr2):
    overlapno = int(N*0.2)
    tmp = []
    tmp2 = []
    for i in range(overlapno):
        tmp.append(i)
    for h in range(overlapno):
        tmp2.append(h)
    for a in range(len(tmp2)):
        arr2.append(arr[tmp2[a]])
    tmp.sort(reverse=True)
    for j in range(len(tmp)):
        del arr[tmp[j]]

    return arr, arr2

def plotcheck(parr, rmsaaa, IDarr):

    pointarrnew = [None for _ in range(len(parr))]
    maxminpoint = [None for _ in range(len(rmsaaa))]
    Plen = len(parr)
    N = 50 #window time 2s
    i = 0
    tmp2s = []
    tmp2soverlap = []
    tmprms = []
    tmpid = []
    tmpidoverlap = []
    tmptimestamp = []
    pointarr = []
    th = 0.1
    checkfloor = False
    flag1st = False
    flagonetime = True
    checkcount = 0
    pointarrtmp = []
    tmpidnooverlap=[]
    while True:
        tmp2s.append(parr[i])
        tmpid.append(i)
        tmpidnooverlap.append(i)
        tmptimestamp.append(IDarr[i])
        tmprms.append(rmsaaa[i])
        if len(tmp2s) == N:
            aa = np.mean(tmp2s[:10])
            ab = np.mean(tmp2s[-10:])
            a = ab - aa
            if abs(a) > th:
                checkfloor = True
                pointarr.append(tmpid[0])
                pointarr.append(tmpid[-1])
                if a < 0 and flagonetime == True:
                    des = "Go up"
                    desno = 1
                    flagonetime = False
                elif a > 0 and flagonetime == True:
                    des = "Go down"
                    desno = 2
                    flagonetime = False
                elif flag1st == False:
                    checklocation1st = tmpid[0]
                    flag1st = True
            elif abs(a) < th and checkfloor == True:
                if checkcount > 3:
                    checkfloor = False
                    pointarrtmp = pointarr
                    tmp2soverlap = []
                    tmpidoverlap = []
                    flag1st = False
                    checkcount = 0
                checkcount+=1
            tmp2s, tmp2soverlap = overlap(N, tmp2s, tmp2soverlap)
            tmpid, tmpidoverlap = overlap(N, tmpid, tmpidoverlap)
            i+=1
            if i == Plen:
                break
        else:
            i+=1
            if i == Plen:
                break

    pointarrtmp.sort()

    begintime = pointarrtmp[0]
    endtime = pointarrtmp[-1]

    # print(len(pointarrnew))
    # print(len(parr))
    for j in range(len(tmpidnooverlap)):
        if tmpidnooverlap[j] == begintime:
            pointarrnew[j] = parr[j]
        elif tmpidnooverlap[j] == endtime:
            pointarrnew[j] = parr[j]

    startendtimeparr = []
    for i in range(len(tmptimestamp)):
        if begintime == i:
            startendtimeparr.append(tmptimestamp[i])
        elif endtime == i:
            startendtimeparr.append(tmptimestamp[i])

    beforebegintime = begintime - 38
    afterbegintime = begintime + 38

    beforeendtime = endtime - 38
    afterendtime = endtime + 38

    maxarr = []
    maxminarr = []
    minarr = []
    if desno == 1:
        for i in range(len(tmpidnooverlap)):
            if beforebegintime <= tmpidnooverlap[i] <= afterbegintime:
                maxarr.append(tmprms[i])
        for j in range(len(tmpidnooverlap)):
            if beforeendtime <= tmpidnooverlap[j] <= afterendtime:
                minarr.append(tmprms[j])
        maxminarr.append(np.max(maxarr))
        maxminarr.append(np.min(minarr))

    elif desno == 2:
        for j in range(len(tmpidnooverlap)):
            if beforeendtime <= tmpidnooverlap[j] <= afterendtime:
                minarr.append(tmprms[j])
        for i in range(len(tmpidnooverlap)):
            if beforebegintime <= tmpidnooverlap[i] <= afterbegintime:
                maxarr.append(tmprms[i])
        maxminarr.append(np.max(minarr))
        maxminarr.append(np.min(maxarr))

    timeaccarr = []
    for i in range(len(tmprms)):
        if tmprms[i] == maxminarr[0]:
            timeaccarr.append(tmptimestamp[i])
        elif tmprms[i] == maxminarr[-1]:
            timeaccarr.append(tmptimestamp[i])

    beforebegintimeaccarr = timeaccarr[0] - 1.0
    afterbegintimeaccarr = timeaccarr[0] + 1.0

    beforeendtimeaccarr = timeaccarr[-1] - 1.0
    afterendtimeaccarr = timeaccarr[-1] + 1.5

    dtwacc = []

    startmagacc = []
    endmagacc = []

    for i in range(len(tmptimestamp)):
        if beforebegintimeaccarr <= tmptimestamp[i]  <= afterbegintimeaccarr:
            dtwacc.append(tmprms[i])
            startmagacc.append(tmprms[i])
        elif beforeendtimeaccarr <= tmptimestamp[i]  <= afterendtimeaccarr:
            dtwacc.append(tmprms[i])
            endmagacc.append(tmprms[i])


    for i in range(len(dtwacc)):
        a = dtwacc[i]
        for j in range(len(rmsaaa)):
            if a == rmsaaa[j]:
                maxminpoint[j]= a

    changeparr = []
    idarr =[]
    for i in range(len(pointarrtmp)):
        for j in range(len(tmpidnooverlap)):
            if pointarrtmp[i] == tmpidnooverlap[j]:
                changeparr.append(parr[j])
                idarr.append(tmpidnooverlap[j])

    idarr = []
    countsss = 0
    for i in range(len(changeparr)):
        idarr.append(countsss)
        countsss+=1


    X = np.array([changeparr]).T
    y = np.array([idarr]).T


    # print(X)
    #
    # plt.plot(X, y, 'ro')
    # plt.show()

    end = len(X)

    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis = 1)

    # Calculating weights of the fitting line
    A = np.dot(Xbar.T, Xbar)
    b = np.dot(Xbar.T, y)
    w = np.dot(np.linalg.pinv(A), b)
    #print('w = ', w)
    # Preparing the fitting line
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(X[0], X[-1], 2)
    y0 = w_0 + w_1*x0

    if desno == 1:
        xB = changeparr[0]
        yB = idarr[0]
        xC = changeparr[-1]
        yC = idarr[-1]
        xA = changeparr[-1]
        yA = 0
        rangeAB = np.sqrt((xB-xA)**2 + (yB-yA)**2)
        rangeBC = np.sqrt((xB-xC)**2 + (yB-yC)**2)
        cosa = rangeAB/rangeBC
        Palpha= round(math.degrees(math.cos(cosa)),2)
    else:
        xB = changeparr[0]
        yB = idarr[0]
        xC = changeparr[-1]
        yC = idarr[-1]
        xA = changeparr[-1]
        yA = 0
        rangeAB = np.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
        rangeBC = np.sqrt((xB - xC) ** 2 + (yB - yC) ** 2)
        cosa = rangeAB / rangeBC
        Palpha= round(math.degrees(math.cos(cosa)),2)

    # Drawing the fitting line
    # plt.plot(X.T, y.T, 'ro')     # data
    # plt.plot(x0, y0)               # the fitting line
    # #plt.axis([140, 190, 45, 75])
    # plt.ylabel('Samples')
    # plt.xlabel('Pressure value')
    # plt.show()

    return startmagacc, endmagacc, timeaccarr,startendtimeparr, des, desno, Palpha

def dnf(n):
    a = 3
    b = 7
    count = 1
    if n <3:
        return count
    while(True):
        if(a <= n < b):
            break;
        else:
            count+=1
            a+=4
            b+=4
    return count

def getvalue(IDaa, value):
    i = 0
    tmpti = []
    while i < len(value):
        if value[i] == 3:
            tmpti.append(IDaa[i])
        i+=1

    movingtime = []
    movingtime.append(tmpti[0])
    movingtime.append(tmpti[-1])
    #print(movingtime)
    return movingtime

def changepo(arr):
    tmparr = []
    a = arr[0]
    b = arr[1]
    if a>b:
        tmparr.append(b)
        tmparr.append(a)
    else:
        tmparr.append(a)
        tmparr.append(b)
    return tmparr

if __name__ == '__main__':
    building = ["ofSWBuilding", "ofOfficialBuilding", "ofInternationalHall", "ofBiomedicalScience", "ofCampus",
                "ofLibrary"]
    title = ["Calling", "Pocket", "Texting", "Swinging"]
    profiledown = pd.read_csv(
        'processeddatatestbanlaylai/final/ofOfficialBuilding/Texting/2_1_Texting_1_Done.csv')  # dung lam profile down
    profileup = pd.read_csv(
        'processeddatatestbanlaylai/final/ofOfficialBuilding/Texting/1_2_Texting_1_Done.csv')  # dung lam profile up

    for j in range(len(building)):
        namebuilding = building[j]
        print(namebuilding)
        actualmovingtime = []
        predictmovingtime = []
        beforeerrorstarttime = []
        aftererrorstarttime = []
        beforeerrorendtime = []
        aftererrorendtime = []
        nametitle = 'Pocket'
        leftarr = [1, 1, 1]
        rightarr = [2, 3, 4]
        dem = 0
        flag = True
        countsoluong = 0
        while dem < 3:
            left = leftarr[dem]
            right = rightarr[dem]
            dem += 1
            totals = 11
            sub = left - right
            counterror = 0
            if sub < 0:
                truedes = "Go up"
                truelevel = abs(sub)
            else:
                truedes = "Go down"
                truelevel = abs(sub)

            countfile = 1
            while countfile < totals:
                total1 = pd.read_csv(
                    'processeddatatestbanlaylai/final/{buildingname}/{holding}/{LEFT}_{RIGHT}_{holding}_{counts}_Done.'
                    'csv'.format(buildingname=namebuilding, holding=nametitle, LEFT=left, RIGHT=right, counts=countfile))

                IDaa, pa, acc, typearr = xulytim(total1)
                startpredict, endpredict, timeaccarrpredict, startendtimeparrpredict, despredict, desnopredict, palpha = plotcheck(
                    pa, acc, IDaa)
                actualtimemovingarr = getvalue(IDaa, typearr)
                actualtimemovingarr = changepo(actualtimemovingarr)
                if 56 < palpha < 58:
                    if desnopredict == 1:
                        IDaa, pa, acc, typearr = xulytim(profileup)
                        startprofile, endprofile, timeaccarrprofile, startendtimeparrprofile, desprofile, desnoprofile, palpha = plotcheck(
                            pa, acc, IDaa)
                        #print(f'Up {startprofile} - {endprofile}')
                    else:
                        IDaa, pa, acc, typearr = xulytim(profiledown)
                        startprofile, endprofile, timeaccarrprofile, startendtimeparrprofile, desprofile, desnoprofile, palpha = plotcheck(
                            pa, acc, IDaa)
                        #print(f'Down {startprofile} - {endprofile}')

                    startpredict = np.array(startpredict)
                    startprofile = np.array(startprofile)
                    startres = dtw(startpredict, startprofile)
                    endpredict = np.array(endpredict)
                    endprofile = np.array(endprofile)
                    endres = dtw(endpredict, endprofile)
                    predicttimemovingarr = []
                    th = 25
                    if startres.distance < th and endres.distance < th:
                        movingtimepredict = abs(timeaccarrpredict[-1] - timeaccarrpredict[0])
                        predicttimemovingarr.append(timeaccarrpredict[0])
                        predicttimemovingarr.append(timeaccarrpredict[-1])
                        levelpredict = dnf(movingtimepredict)
                    elif startres.distance > th and endres.distance < th:
                        movingtimepredict = abs(timeaccarrpredict[-1] - startendtimeparrpredict[0]) - 1.5
                        predicttimemovingarr.append(startendtimeparrpredict[0])
                        predicttimemovingarr.append(timeaccarrpredict[-1])
                        levelpredict = dnf(movingtimepredict)
                    elif startres.distance < th and endres.distance > th:
                        movingtimepredict = abs(startendtimeparrpredict[-1] - timeaccarrpredict[0]) - 1.5
                        predicttimemovingarr.append(timeaccarrpredict[0])
                        predicttimemovingarr.append(startendtimeparrpredict[-1])
                        levelpredict = dnf(movingtimepredict)
                    else:
                        movingtimepredict = abs(startendtimeparrpredict[-1] - startendtimeparrpredict[0]) - 1.5
                        predicttimemovingarr.append(startendtimeparrpredict[0])
                        predicttimemovingarr.append(startendtimeparrpredict[-1])
                        levelpredict = dnf(movingtimepredict)
                else:
                    print("Elevator is not moving")

                a = actualtimemovingarr[1] - actualtimemovingarr[0]
                b = predicttimemovingarr[1] - predicttimemovingarr[0]
                actualmovingtime.append(abs(round(a, 2)))
                predictmovingtime.append(abs(round(b, 2)))


                if levelpredict == truelevel and desprofile == truedes:
                    countsoluong += 1
                    a = actualtimemovingarr[0] - predicttimemovingarr[0]
                    b = actualtimemovingarr[1] - predicttimemovingarr[1]
                    beforeerrorstarttime.append(abs(round(a,2)))
                    beforeerrorendtime.append(abs(round(b, 2)))
                countfile += 1
            if dem == 3 and flag == True:
                flag = False
                tmp = leftarr
                leftarr = rightarr
                rightarr = tmp
                dem = 0

        meanbeforeerrorstarttime = round(np.mean(beforeerrorstarttime),2)
        meanbeforeerrorendtime = round(np.mean(beforeerrorendtime),2)

        print(actualmovingtime)
        print(predictmovingtime)
