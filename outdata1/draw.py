import numpy as np
import matplotlib.pyplot as plt

fFibre = open("FibreStress.csv","r")
Fibre = fFibre.readlines()

fxy = open("xy.csv","r")
Xy = fxy.readlines()
plt.ion()
fig = plt.figure("frame")

for i in range(0,250):
    X = Xy[i*30].strip().split(",")
    Xnum = np.array([float(i) for i in X])
    Y = Xy[i*30+1].strip().split(",")
    Ynum = np.array([float(i) for i in Y])

    fibre = Fibre[i*20].strip().split(",")
    # fibreNum = np.array([float(i) for i in fibre])

    # imax = np.argmax(fibreNum)

    plt.plot(Xnum,Ynum)
    # plt.scatter(Xnum[imax],Ynum[imax])
    plt.xlim([0,2e6])
    plt.ylim([-3e6,0.5e6])
    plt.gcf().gca().set_aspect("equal")
    # plt.show()
    # plt.pause(0.01)
    print(i)
    fig.savefig(f"{i}.png",dpi=600)
    fig.clf()

