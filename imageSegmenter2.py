import matplotlib.pyplot as plt
import numpy,time
import pickle
from sklearn.externals import joblib
from PIL import Image
import cv2
from converter import imageprepare
from scipy import misc
from resizeimage import resizeimage
#from draw import Paint
plt.ion()
idx = 0
def cutImage(start, count, mini, maxi, idx):
	#mask = 200
    print("Loaded f")
    mask=100
    img = misc.imread("Hello1.png")
    s1= img[mini:maxi, start:count]
    img = Image.fromarray(s1)
    idx+=1
    img.save("pic"+str(idx)+".png") 
    mnist = imageprepare("pic"+str(idx)+".png")
    for i in range(len(mnist)):
        if mnist[i]<mask:
            mnist[i]=0
    mimage=[mnist[(i)*28:(i+1)*28] for i in range(28)]
    #for i in mimage:
        #print(i)
    plt.subplot(3,3,idx)
    print("Loaded Subplot")
    plt.imshow(mimage, cmap = plt.cm.gray_r, interpolation="nearest")
    #cf = joblib.load("trainedCF.sav")
    #cf2 = pickle.load(open('clfver2.5Both1000P.sav','rb'))
    cf3 = pickle.load(open('clfvM2.6v10e7BothAllkaliP.sav','rb'))
    #cf4 = pickle.load(open('clf120420181e-8kali.pickle','rb'))
    cf5 = pickle.load(open('clf120420184pm1e-8kali.pickle','rb'))    
    #speci = cLconv(cf4.predict([mnist]))
    speci2 = cLconv(cf5.predict([mnist]))
    #print('Prediction',cf.predict([mnist]),cf2.predict([mnist]),cf3.predict([mnist]),speci,speci2)
    print(cf3.predict([mnist]),speci2)
    return cf3.predict([mnist])
def cLconv(a):
	if a<10:
		return a
	if a<36:
		return chr(a+55)
	return chr(a+97-36)
#Paint()
def getText():
    A = cv2.imread("Hello1.png",0)
    print("Loaded k")
    #cv2.imshow('image', A)
    #cv2.waitKey(1000)
    #time.sleep(5)
    arr = numpy.asarray(A)
    #print("Loaded")
    typearr = 0
    start = 0
    tCount = 0
    mode = 0
    breaking = 0
    output = ""
    mini = 0
    maxi = 0
    #print("Loaded")
    OutputStr = ''
    for column in arr.T:
        count = 0
        #print("Loaded T")
        prev = 1
        currmod = 0
        n = 0
        for i in column:
            if i!=255 and prev == 1:
                count += 1
                prev = 0
                if mini == 0:
                    mini = n
                elif n < mini:
                    mini = n
            elif i == 255 and prev == 0:
                prev = 1
                if maxi == 0:
                    maxi = n
                elif n > maxi:
                    maxi =n
            n+=1
        #print("Loaded G",count,currmod,mode)
        
        if count == 0:
            currmod = 0
        else:
            currmod = 1
        if mode != currmod and mode == 0:
            start = tCount
            mode = currmod
        elif mode != currmod and mode == 1:
            #print("Loaded 4")
            OutputStr += cutImage(start, tCount, mini, maxi, breaking)[0]   
            start = tCount
            mode = currmod
            maxi = 0
            mini = 0
            breaking+=1
        tCount += 1
    #time.sleep(3)
    print(OutputStr)
    #plt.show(.1)
    return OutputStr
if __name__=='__main__':
    getText()
