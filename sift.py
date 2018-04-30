import cv2
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def find_hog(img,x1,y1):
    temp = np.zeros((4,4,2))
    bin_ = np.zeros(8)
    for i in range(x1,x1+4):
        for j in range(y1,y1+4):
            temp[i-x1][j-y1][0] = int(img[i][j+1])-int(img[i][j-1])
            temp[i-x1][j-y1][1] = int(img[i+1][j])-int(img[i-1][j])
            ang = math.atan2(temp[i-x1][j-y1][1],temp[i-x1][j-y1][0])
            ang = (ang*360)/(2*3.14)
            if(ang<0):
                ang += 360
            global key_p
            key = key_p
            key2 = [key[1],key[0]]
            key = key2
            temp3 = ((i-key[0])**2+(j-key[1])**2)**0.5
            temp2 = ((temp[i-x1][j-y1][0])**2 + (temp[i-x1][j-y1][1])**2)**0.5
            temp2 += temp3
            if(ang<44):
                bin_[0] += temp2
            elif(ang<88):
                calc1 = abs(ang-44)
                calc2 = abs(ang-88)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[0] += temp2*cal1
                bin_[1] += temp2*cal2
            elif(ang<132):
                calc1 = abs(ang-88)
                calc2 = abs(ang-132)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[1] += temp2*cal1
                bin_[2] += temp2*cal2
            elif(ang<176):
                calc1 = abs(ang-132)
                calc2 = abs(ang-176)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[3] += temp2*cal1
                bin_[4] += temp2*cal2
            elif(ang<220):
                calc1 = abs(ang-176)
                calc2 = abs(ang-220)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[4] += temp2*cal1
                bin_[5] += temp2*cal2
            elif(ang<264):
                calc1 = abs(ang-220)
                calc2 = abs(ang-264)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[5] += temp2*cal1
                bin_[6] += temp2*cal2
            elif(ang<318):
                calc1 = abs(ang-264)
                calc2 = abs(ang-318)
                cal1 = calc1/(calc1+calc2)
                cal2 = calc2/(calc1+calc2)
                bin_[6] += temp2*cal1
                bin_[7] += temp2*cal2
            else:
                bin_[7] += temp2


    return bin_

def hog(img,x1,y1):
    bin_ = []
    for i in range(4):
        for j in range(4):
            temp = find_hog(img,(4*i)+x1,(4*j)+y1)
            bin_.append(temp)
    bin_ = np.array(bin_)
    for i in range(8):
        mi = 1e18
        ma = -1
        for j in range(16):
            mi = min(bin_[j][i],mi)
            ma = max(bin_[j][i],ma)
        for j in range(16):
            bin_[j][i] = (bin_[j][i]-mi)/ma

    return bin_


def main():
    data1 = []
    data2 = []
    for i in range(1,5):
        print(i)
        img  = cv2.imread("ob{}.jpg".format(i),0)
        img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("ob{}.jpg".format(i),img)
        sift = cv2.xfeatures2d.SIFT_create()
        kp   = sift.detect(img,None)
        img22 = cv2.drawKeypoints(img,kp, None)
        cv2.imwrite("ob{}_.png".format(i),img22)
        pts = []
        for j in range(len(kp)):
            pts.append(kp[j].pt)
        kmeans = KMeans(n_clusters=30, random_state=0).fit(pts)
        mean = kmeans.cluster_centers_
        for j in range(len(mean)):
            if(mean[j][0]-5>0 and mean[j][1]-5>0 and mean[j][1]+5<200 and mean[j][1]+5<200):
                temp = []
                x = 0
                for k in range(int(mean[j][0]-5),int(mean[j][0]+5)):
                    y = 0
                    for l in range(int(mean[j][0]-5),int(mean[j][0]+5)):
                        temp.append(img[k][l])
                        y +=1
                    x+=1
                data1.append(temp)
                data2.append(i)


    data1 = np.array(data1)
    data2 = np.array(data2)
    clf = SVC()
    clf.fit(data1, data2)

    img  = cv2.imread("ob6.jpg",0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp   = sift.detect(img,None)
    pts = []
    for i in range(len(kp)):
        pts.append(kp[i].pt)
    kmeans = KMeans(n_clusters=30, random_state=0).fit(pts)
    mean = kmeans.cluster_centers_
    data = []
    for j in range(len(mean)):
        if(mean[j][0]-5>0 and mean[j][1]-5>0 and mean[j][1]+5<200 and mean[j][1]+5<200):
            temp = []
            x = 0
            for k in range(int(mean[j][0]-5),int(mean[j][0]+5)):
                y = 0
                for l in range(int(mean[j][0]-5),int(mean[j][0]+5)):
                    temp.append(img[k][l])
                    y +=1
                x+=1
            data.append(temp)

    data = np.array(data)


    print(clf.predict(data))

if __name__ == "__main__":
    main()
