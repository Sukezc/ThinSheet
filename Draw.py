from fileinput import filename
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

millionYearseconds = 31557384000000

def drawPic(deltaT_real,data_frame,folders,picfoldername,xyfilename, xylim,depthline,length):
    '''
    All the folder and data must be created before the function is called
    root/foldername/picture/temp.png
    root/foldername/motion.gif
    root/foldername/Conf.xml
    root/foldername/xy.csv
    root/foldername/moment.csv
    '''

    DeltaT = deltaT_real / millionYearseconds

    Path = os.walk(folders).__next__()
    root = Path[0]
    for folder in Path[1]:
        with open(os.path.join(root,folder,xyfilename)) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if not (i % data_frame * 3):
                    xline = lines[i]
                    yline = lines[i+1]
                    x = [np.float64(j) for j in xline.split(',')]
                    y = [np.float64(j) for j in yline.split(',')]
                    plt.gcf().clf()
                    plt.title("{:8.4f} Myr".format(DeltaT * i / 3))
                    plt.plot(x, y)
                    for depth in depthline:
                        plt.plot([xylim["xleft"]*length,xylim["xright"]*length],[-depth*1e3,-depth*1e3],linestyle="--")    
                    # plt.plot([xylim["xleft"]*length,xylim["xright"]*length],[-1e4,-1e4],linestyle="--")
                    # plt.plot([xylim["xleft"]*length,xylim["xright"]*length],[-1.3e5,-1.3e5],linestyle="--")
                    # plt.plot([xylim["xleft"]*length,xylim["xright"]*length],[-4.1e5,-4.1e5],linestyle="--")
                    # plt.plot([xylim["xleft"]*length,xylim["xright"]*length],[-6.6e5,-6.6e5],linestyle="--")
                    
                    plt.xlim([xylim["xleft"]*length,xylim["xright"]*length])
                    plt.ylim([xylim["yleft"]*length, xylim["yright"]*length])
                    plt.gcf().gca().set_aspect('equal')
                    plt.gcf().savefig(os.path.join(root,folder,picfoldername,"{}.png".format(int(i / 3))))
                    plt.gcf().clf()


def drawGif(fps,folders,picfoldername):
    '''
    All the folder and data must be created before the function is called
    root/foldername/picture/temp.png
    root/foldername/motion.gif
    root/foldername/Conf.xml
    root/foldername/xy.csv
    root/foldername/moment.csv
    '''
    duration = 1.0 / fps

    Path = os.walk(folders).__next__()
    root = Path[0]
    for folder in Path[1]:
        image_list = []
        for image_name in os.listdir(os.path.join(root,folder,picfoldername)):
            image_list.append(image_name)
        image_list.sort(key=lambda x: int(x.split('.')[0]))
        gif_name = os.path.join(root,folder,"motion.gif")
        frames = []
        image_list = [(os.path.join(root,folder,picfoldername,i)) for i in image_list]
        for im in image_list:
            frames.append(imageio.v2.imread(im))
        imageio.mimsave(gif_name, frames, "GIF", duration=duration, loop=0)

    
def drawTorque(deltaT_real,folders,torquefilename):
    DeltaT = deltaT_real / millionYearseconds
    Path = os.walk(folders).__next__()
    root = Path[0]
    for folder in Path[1]:
        with open(os.path.join(root,folder,torquefilename)) as f:
                lines = f.readlines()
                Gravityline = lines[0]
                Pforceline = lines[1]
                Gravity = [np.float64(j) for j in Gravityline.split(',')]
                Pforce = [np.float64(j) for j in Pforceline.split(',')]
                length = Gravity.__len__()
                temp = np.arange(0, length, 1)
                t = [i * DeltaT for i in temp]
                plt.gcf().clf()
                plt.title("Torque")
                plt.plot(t, Gravity, t, Pforce)
                plt.gcf().savefig(os.path.join(root,folder,"torque.png"))
                plt.gcf().clf()