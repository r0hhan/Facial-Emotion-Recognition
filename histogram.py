
#----------------------------
import matplotlib, numpy as np, sys
import time
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

timeCount = 1

root = Tk.Tk()
f = Figure(figsize=(15,5), dpi=100)
canvas = FigureCanvasTkAgg(f, master=root)
Ysum = np.zeros(7)
readings = []
summation = []
 
def initialize():
        #button = tkinter.Button(self,text="Open File",command=self.OnButtonClick).pack(side=tkinter.TOP)
        
        #self.canvasFig=pltlib.figure(1)
    
    ax1 = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)
   
    #w = Tk.Label(root, text = "First                                                                                                                        Second                                                                                                                        Third")
    #w1=Tk.Label(root, text = "test")
    #w.pack()
   
    
    width = .5
    x=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    data = open("realTimeEmotionData.txt","r").read()
    dataArray = data.split('\n')
    valar = []
    for val in dataArray:
    	#print(val)
    	valar.append(int(float(val)))
    ax1.bar(x, valar, width)   
    ax2.bar(x, valar, width)   
    ax3.bar(x, valar, width)
            
    
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        #self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(Fig, master=self)
        #self.canvas.show()
        #self.canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
    #self.update()
    Refresh()
        
    #button = tkinter.Button(self, text="Example").pack(side=tkinter.TOP)
    #button.after(1000, self.test)
        
    #def refreshFigure(self,x,y):
        
        
def Refresh():
        # file is opened here and some data is taken
        # I've just set some arrays here so it will compile alone
    global timeCount
    global Ysum
    global readings
    global summation
    
    realTimeData = open("realTimeEmotionData.txt","r").read()
    dataArray = realTimeData.split('\n')
    realTimeVals = []
   
    
    for val in dataArray:
        realTimeVals.append(float(val))
    x=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    X = np.array(x)
    Yrt = np.array(realTimeVals)
    Ysum = Ysum + Yrt
    Yavg = Ysum / timeCount 
    readingCount = len(readings)   
    duration = 120

    readings.append(realTimeVals)
    summation = np.zeros(7)
    for i in range(0, readingCount):
        summation = summation + readings[i]
    Ydavg = summation /readingCount 
 
    #print('current average =', avg)
    #print('readings used for average:', readings)

    if len(readings) == duration:
        readings.pop(0)

    #print('readings saved for next time:', readings)
    #YLast = np.array(avg)
    
            
    ax1 = canvas.figure.axes[0]
    ax2 = canvas.figure.axes[1]
    ax3 = canvas.figure.axes[2]
    ax1.clear()
    ax2.clear()
    ax3.clear()
        #ax.set_xlim(x.min(), x.max())
    ax1.set_ylim(0, 100)        
    ax1.bar(X,Yrt,0.5)
    ax2.set_ylim(0, 100)        
    ax2.bar(X,Ydavg,0.5)
    ax3.set_ylim(0, 100)        
    ax3.bar(X,Yavg,0.5)
    canvas.draw()
    
    timeCount = timeCount + 1
    root.after(500, Refresh)
        #button.after(1000, self.test)
        
    
initialize()
root.mainloop()
