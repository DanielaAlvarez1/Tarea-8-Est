import scipy.stats as ss
from scipy import optimize
import numpy as np

def f(x):
    return (np.exp(-x))/(1 + np.exp(-x))**2

def finverse(x):
    return -np.log((1/x)-1)

#SAMPLES
def randomSamples(n):
    for n_i in n:
        Samples = []
        for j in range(0,1000):
            X = np.random.rand(n_i)
            X2 = []
            for y in X:
                X2.append(finverse(y))
            Samples.append(X2)
        nSamples[n_i] = Samples

#MVE
def sampleMeans(nSamples, nMeans):
    for n_i in n:
        nSamp = nSamples[n_i]
        nMn = []
        for X in nSamp:
            mean = ss.gmean(X)
            nMn.append(mean)
        nMeans[n_i] = nMn

def l(Sample,theta):
    n = len(Sample)
    sum = 0
    for x_i in Sample:
        value = np.log(1 + np.exp(theta - x_i))
        sum+=value
    return n*theta - n*(ss.gmean(Sample)) - 2*sum

def dldtheta(Sample,theta):
    n = len(Sample)
    sum = 0
    for x_i in Sample:
        value = (np.exp(theta - x_i))/(1 + np.exp(theta - x_i))
        sum+=value
    return n - 2*sum

def newtonRaphson(Sample, Mean):
    x_0 = Mean
    max_iter = 100
    i = 0  
    xi_1 = x_0
    while i > max_iter:
        i = i + 1
        xi = xi_1-l(Sample,xi_1)/dldtheta(Sample,xi_1) 
        xi_1 = xi
    return xi_1

def MVE(nSamples, nMeans, nMVE):
    for n_i in n:
        nSamp = nSamples[n_i]
        nMn = nMeans[n_i]
        nMV = []
        for j in range(0,1000):
            actualSample = nSamp[j]
            acSamplemean = nMn[j]
            mve = newtonRaphson(actualSample, acSamplemean)
            nMV.append(mve)
        nMVE[n_i] = nMV

#STATISTICS
def lnLambda(n,sample,theta,emv):
    sum=0
    for x in sample:
        sum += np.log((1+np.exp(emv-x))/(1+np.exp(theta-x)))
    return (-2)*((n*(theta-emv))+(2*sum))

def Wald(n,emv):
    return (n/3)*np.power(emv,2)

def Scores(n,sample,theta):
    sum=0
    for x in sample:
        sum += np.exp(theta-x)/(1+np.exp(theta-x))
    return np.power(((n-(2*sum))/np.sqrt(n/3)),2)

def getStats(n,nSamples,nMVE,nLambda,nWald,nScores):
    for n_i in n:
        nSamp = nSamples[n_i]
        nMv = nMVE[n_i]
        Lambd = []
        Wal = []
        Sco = []
        for j in range(0,1000):
            actualSample = nSamp[j]
            acSamplemv = nMv[j]
            print("Prueba")
            print(acSamplemv)
            lam = lnLambda(n,actualSample,0,acSamplemv)
            wa = Wald(n,acSamplemv)
            sc = Scores(n,actualSample,0)
            Lambd.append(lam)
            Wal.append(wa)
            Sco.append(sc)
        nLambda[n_i] = Lambd
        nWald[n_i] = Wal
        nScores[n_i] = Sco

#PEARSON
def quantils(Quantils):
    density = ss.chisquare(1)
    p = 1/7
    for i in range(0,7):
        quan = density.ppf(p)
        Quantils.append(quan)
        p+=1/7

def classify(n, nSamples, nFreq, Quantils):
    for n_i in n:
        nSamp = nSamples[n_i]
        listFreq = []
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
        p5 = 0
        p6 = 0
        p7 = 0
        for j in range(0,1000):
            actualSample = nSamp[j]
            for x_i in actualSample:
                if x_i <= Quantils[0]:
                    p1+=1
                elif (x_i > Quantils[0]) and (x_i <= Quantils[1]):
                    p2+=1
                elif (x_i > Quantils[1]) and (x_i <= Quantils[2]):
                    p3+=1
                elif (x_i > Quantils[2]) and (x_i <= Quantils[3]):
                    p4+=1
                elif (x_i > Quantils[4]) and (x_i <= Quantils[5]):
                    p5+=1
                elif (x_i > Quantils[5]) and (x_i <= Quantils[6]):
                    p6+=1
                elif (x_i > Quantils[6]) and (x_i <= Quantils[7]):
                    p7+=1
            dic = {1:p1,2:p2,3:p3,4:p4,5:p5,6:p6,7:p7}
            listFreq.append(dic)
        nFreq[n_i] = listFreq

def pearson(n,nFreq,Pearson):
    for n_i in n:
        nFre = nFreq[n_i]
        Per = []
        for j in range(0,1000):
            sum = 0
            for i in range(1,8):
                yi = nFre[j][i]
                value = ((yi - (n_i/7))**2)/((n_i/7))
                sum+=value
            K = sum
            Per.append(K)
        Pearson[n_i] = Per

#CHI2
def chisquare(n,Pearson, nLambda, nWald, nScores, nChi):
    for n_i in n:
        nPer = Pearson[n_i]
        nLam = nLambda[n_i]
        nWa = nWald[n_i]
        nSc = nScores[n_i]
        Chilam = []
        Chiwa = []
        Chisc = []
        for j in range(0,1000):
            per = nPer[j]
            lam = nLam[j]
            wa = nWa[j]
            sc = nSc[j]
            chiestlam = ss.chisquare(per,lam)
            chiestwa = ss.chisquare(per,wa)
            chiestsc = ss.chisquare(per,sc)
            Chilam.append(chiestlam)
            Chiwa.append(chiestwa)
            Chisc.append(chiestsc)
        nChi[n_i] = {1: Chilam,2:Chiwa,3:Chisc}

#TEST
def test(n, nSamples, nTest):
    for n_i in n:
        nSamp = nSamples[n_i]
        Test = []
        for j in range(0,1000):
            actualSample = nSamp[j]
            tst = ss.kstest(actualSample, ss.chi2(1).cdf)
            Test.append(tst)
        nTest[n_i] = Test

"""
START
"""
#PUNTO 1
n = [10, 20, 50, 100, 200]
nSamples = {}
nMeans ={}
nMVE = {}
randomSamples(n)

#PUNTO 2
sampleMeans(nSamples, nMeans)
MVE(nSamples, nMeans, nMVE)
nLambda = {}
nWald ={}
nScores = {}
getStats(n,nSamples,nMVE,nLambda,nWald,nScores)

#PUNTO 3
Quantils = []
nFreq = {}
quantils(Quantils)
classify(n, nSamples, nFreq, Quantils)
Pearson = {}
pearson(n,nFreq,Pearson)
nChi = {}
chisquare(n,Pearson, nLambda, nWald, nScores, nChi)

#PUNTO 4
nTest = {}
test(n, nSamples, nTest)

