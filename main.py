import Draw
import subprocess
import time
import shutil
import os
import numpy as np
import xml.etree.ElementTree as ET


def main():
    
    def HalfUp(value):
        return np.format_float_scientific(value, precision = 4, exp_digits=2,trim='-')

    MantleViscosity = np.array([5e19])
    SlabViscosity = np.array([2e24])
    CriticalAngle =  np.array([0.26])
    # np.arange(0.27, 0.21, -0.002)
    #UpperVelocity = np.arange(1.5e-9,1.6e-9, 0.2e-9)
    UpperVelocity = np.array([6e-9])


    MyCategory = "1102"
    deltaT_seconds = 0
    if not os.path.exists(os.path.join("output",MyCategory)):
        os.mkdir(os.path.join("output",MyCategory))

    
    
    def DeltaTime2RealTime(configuration):
        tree = ET.parse(configuration)
        root = tree.getroot()
        viscosity = 0.0
        H = 0.0
        velocity = 0.0
        upperVelocity = 0.0
        g = 0.0
        density = 0.0
        deltaT_co = 0
        for child in root:
                if child.tag == "viscosity":
                    viscosity = np.float64(child.text)
                elif child.tag == "H":
                    H = np.float64(child.text)
                elif child.tag == "velocity":
                    velocity = np.float64(child.text)
                elif child.tag == "UpperVelocity":
                    upperVelocity = np.float64(child.text)
                elif child.tag == "g":
                    g = np.float64(child.text)
                elif child.tag == "density":
                    density = np.float64(child.text)
                elif child.tag == "deltaT_coefficient":
                    deltaT_co = np.float64(child.text)
                else:
                    pass
        return (viscosity * H * H / (velocity + upperVelocity)**3 / g / density)**0.25 / deltaT_co

    
    MyPack = [(a,b) for a in MantleViscosity for b in UpperVelocity]
    num = MyPack.__len__()
    for index,it in enumerate(MyPack):
            print("----------------------Round {}:{}----------------------".format(index + 1,num))
            Cur_dir = os.path.join("output", MyCategory,"MantleViscosity_"+str(HalfUp(it[0]))+"UpperVelocity_"+str(HalfUp(it[1])))
            if not os.path.exists(Cur_dir):
                os.mkdir(Cur_dir)
            tree = ET.parse("Conf.xml")
            root = tree.getroot()
            for child in root:
                if child.tag == "MantleViscosity":
                    child.text = str(HalfUp(it[0]))
                elif child.tag == "UpperVelocity":
                    child.text = str(HalfUp(it[1]))
                else:
                    pass
            ExeCwd = Cur_dir + "/"
            tree.write(os.path.join(Cur_dir,"Conf.xml"),encoding="utf-8",xml_declaration=True)
            deltaT_seconds = DeltaTime2RealTime(os.path.join(Cur_dir,"Conf.xml"))
            p = subprocess.Popen("ThinSheet.exe Conf.xml",cwd=ExeCwd)
            p.wait()
            print("return code:")
            print(p.returncode)
            Path = os.walk(Cur_dir).__next__()
            for folder in Path[1]:
                if not os.path.exists(os.path.join(Cur_dir,folder,"picture")):
                    os.mkdir(os.path.join(Cur_dir,folder,"picture"))

            xylim = {}
            xylim["xleft"] = 0.0
            xylim["xright"] = 1.7
            xylim["yleft"] = -1.0
            xylim["yright"] = 0.1
            length = 2e6
            depthline = [10,130,200,300,400]
            Draw.drawPic(deltaT_seconds,5,Cur_dir,"picture","xy.csv",xylim,depthline,length)
            Draw.drawGif(30,Cur_dir,"picture")
            Draw.drawTorque(deltaT_seconds,Cur_dir,"torque.csv")
            

                
if __name__ == "__main__":
    main()
