import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
path_fr  = "./Girl/img"
path_gt = "./Girl/groundtruth_rect.txt"

def find_features(img,x1,y1,x2,y2):
    img2 = img[y1:y2,x1:x2]
    pts = []
    # sift = cv2.xfeatures2d.SIFT_create()yt
    # kp   = sift.detect(img,None)
    # pts = []
    # for j in range(len(kp)):
    #     pts.append([kp[j].pt[0]+x1,kp[j].pt[0]+y1])
    dst = cv2.cornerHarris(img2,2,3,0.04)
    for i in range(0,img2.shape[0]):
       for j in range (0, img2.shape[1]):
          harris = dst[i][j]
          if harris > 1e-5:
              pts.append([j+x1,i+y1])

    kmeans = KMeans(n_clusters=50, random_state=0).fit(pts)
    mean = kmeans.cluster_centers_

    return mean
def main():

    fp = open(path_gt)
    Z = fp.readline().split("\t")
    x, y , w, h = int(Z[0]),  int(Z[1]), int(Z[2]), int(Z[3].split("\n")[0])
    x1, y1, x2, y2 = x, y, x+w, y+h

    img = cv2.imread(os.path.join(path_fr,"0001.jpg"),0)
    img3 = img
    template = img

    cv2.rectangle(img3,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imwrite("./out_bask/img0.png",img3)

    '''Feature Detector'''
    img2 = img[y1:y2,x1:x2]

    pts = find_features(img,x1,y1,x2,y2)

    ct = 0
    pts_copy = list(pts)
    for imgs in sorted(os.listdir(path_fr)):
        if(ct==0):
            ct += 1
            continue

        img = cv2.imread(os.path.join(path_fr,imgs),0)
        print(ct,"*"*10)
        pts2 = []
        for i in range(len(pts_copy)):
            # print(i)
            p = np.zeros((6))
            p = np.array(p)
            p = np.reshape(p,(1,6))
            pt = [[pts[i][1],pts[i][0],1.0]]
            for ite in range(20):
                warp = [[1+p[0][0],p[0][1],p[0][2]],
                        [p[0][3],1+p[0][4],p[0][5]]]
                # print(warp)

                pt = np.array(pt)
                warp = np.array(warp)
                # print(warp.shape)
                val = np.matmul(warp,pt.T)
                val = np.array(val)
                if(val[0][0]<=4 or val[0][0]>=img.shape[0]-4 or val[1][0]<=4 or val[1][0]>=img.shape[1]-4):
                    break
                dif1 = [[int(template[int(pts[i][1])][int(pts[i][0])])-int(img[int(val[0][0])][int(val[1][0])])]]
                dif1 = np.array(dif1)

                jacobian = [[pts[i][1],pts[i][0],1,0,0,0],
                            [0,0,0,pts[i][1],pts[i][0],1]]

                grad_x = int(img[int(val[0][0])][int(val[1][0])+1]) - int(img[int(val[0][0])][int(val[1][0])-1])
                grad_x += int(img[int(val[0][0])+1][int(val[1][0])+1]) - int(img[int(val[0][0])+1][int(val[1][0])-1])
                grad_x += int(img[int(val[0][0])-1][int(val[1][0])+1]) - int(img[int(val[0][0])-1][int(val[1][0])-1])

                grad_y = int(img[int(val[0][0])+1][int(val[1][0])]) - int(img[int(val[0][0])-1][int(val[1][0])])
                grad_y += int(img[int(val[0][0])+1][int(val[1][0])+1]) - int(img[int(val[0][0])-1][int(val[1][0])+1])
                grad_y += int(img[int(val[0][0])+1][int(val[1][0])-1]) - int(img[int(val[0][0])-1][int(val[1][0])-1])

                grad_x /=6
                grad_y /=6
                grad = [[grad_x, grad_y]]


                # print(dif1.shape)
                grad = np.array(grad)
                pro1 = np.matmul(grad,jacobian)
                pro0 = np.matmul(pro1.T,dif1)

                pro2 = np.matmul(pro1.T,pro1)
                pro2 = np.linalg.pinv(pro2)

                dp = np.matmul(pro2,pro0)
                p = p+dp.T
                warp = [[1+p[0][0],p[0][1],p[0][2]],
                        [p[0][3],1+p[0][4],p[0][5]]]

                warp = np.array(warp)
                # print(warp)
            val = np.matmul(warp,pt.T)
            pts2.append(val)
            # print(i)
            # print(warp)
        x1_ = 1e18
        y1_ = 1e18
        x2_ = -1
        y2_ = -1
        for i in range(len(pts2)):
            x1_ = min(pts2[i][1],x1_)
            x2_ = max(pts2[i][1],x2_)

            y1_ = min(pts2[i][0],y1_)
            y2_ = max(pts2[i][0],y2_)
        # print(int(x1),int(y1),int(x2),int(y2))
        # print(int(x1_),int(y1_),int(x2_),int(y2_))
        img3 = img
        cv2.rectangle(img3  ,(int(x1_),int(y1_)),(int(x2_),int(y2_)),(255,0,0),2)
        cv2.imwrite("./out_bask/img{}.png".format(ct),img3)

        x1 = x1_
        y1 = y1_
        x2 = x2_
        y2 = y2_

        template = img
        # if(np.remainder(ct,5)==0):
        #     pts = find_features(template,int(x1),int(y1),int(x2),int(y2))

        ct += 1
        # if(ct==20):
        #     break








if __name__=="__main__":
    main()
