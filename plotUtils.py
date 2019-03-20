import matplotlib.pyplot as plt
import numpy as np
def plot_(x_, y_,x_label="Not given", y_label = "Not given", title_ = "Not given", Thresh = None,count=None):
    try:
        if Thresh != None:
            plt.plot(np.arange(len(y_)),np.full(len(y_),Thresh))
        plt.plot(x_,y_)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title_+" "+str(count)+"/"+str(Thresh))
        plt.show()
    except :
        print ("usage: plot_(x_, y_,x_label, y_label , title_ , Thresh ,count")