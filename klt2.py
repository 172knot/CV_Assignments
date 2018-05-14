import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter
path_fr  = "./Girl/img"
path_gt = "./Girl/groundtruth_rect.txt"

def find_features(img,x1,y1,x2,y2):
    img2 = img[y1:y2,x1:x2]
    pts = []
    # sift = cv2.xfeatures2d.SIFT_create()
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

    kmeans = KMeans(n_clusters=min(1,len(pts)), random_state=4).fit(pts)
    mean = kmeans.cluster_centers_
    print(len(mean))

    return mean
def main():

    fp = open(path_gt)
    Z = fp.readline().split("\t")
    x, y , w, h = int(Z[0]),  int(Z[1]), int(Z[2]), int(Z[3].split("\n")[0])
    x1, y1, x2, y2 = x, y, x+w, y+h



    img = cv2.imread(os.path.join(path_fr,"0001.jpg"),0)
    imgprint  = np.copy(img)
    img = gaussian_filter(img, sigma=1)
    # img = cv2.blur(img,(5,5))
    img3 = np.copy(img)
    img3 = img3
    template = np.copy(img)
    template = template/255.0

    '''Feature Detector'''
    imgx = np.copy(img)
    pts = find_features(imgx,x1,y1,x2,y2)
    # pts = []
    # pts.append([63,38])
    # pts.append([70,26])
    # pts.append([79,32])
    # pts.append([85,43])
    # pts.append([60,48])
    # pts.append([68,58])
    # pts.append([86,59])


    img33 = np.copy(imgprint)
    for i in range(len(pts)):
        cv2.circle(img33,(int(pts[i][0]),int(pts[i][1])),2,(255,0,0),-1)
    cv2.imwrite("./out_girl/img0.png",img33)

    ct = 0
    for imgs in sorted(os.listdir(path_fr)):
        if(ct==0):
            ct += 1
            continue

        img = cv2.imread(os.path.join(path_fr,imgs),0)
        imgprint  = np.copy(img)
        img = gaussian_filter(img, sigma=1)
        # img = cv2.blur(img,(5,5))
        imgp = np.copy(img)
        img33 = np.copy(imgprint)
        img = img/255.0
        print(ct, "*"*10)
        pts2 = []
        for i in range(len(pts)):

            p = np.zeros((6))
            p = np.array(p)
            p = np.reshape(p,(1,6))
            for ite in range(20):
                warp = [[1.0+p[0][0],p[0][1],p[0][2]],
                        [p[0][3],1.0+p[0][4],p[0][5]]]


                hessian = np.zeros((6,6))
                hessian = np.array(hessian)
                pro0 = np.zeros((6,1))
                pro0 = np.array(pro0)


                for k in range(-10,11):
                    for l in range(-10,11):

                        pt = [[k,l,1.0]]
                        pt = np.array(pt)
                        warp = np.array(warp)
                        val = np.matmul(warp,pt.T)
                        val = np.array(val)


                        val[0][0] += pts[i][0]
                        val[1][0] += pts[i][1]

                        if(val[0][0]<15 or val[0][0]>=img.shape[1]-15 or val[1][0]<15 or val[1][0]>=img.shape[0]-15):
                            continue

                        dif1 = float(template[int(pts[i][1])+l][int(pts[i][0])+k]) - float(img[int(val[1][0])][int(val[0][0])])
                        dif2 = []
                        dif2.append(dif1)
                        dif2 = np.reshape(dif2,(1,1))

                        jacobian = [[k,l,1,0,0,0],
                                    [0,0,0,k,l,1]]

                        grad_x = float(img[int(val[1][0])][int(val[0][0])+1]) - float(img[int(val[1][0])][int(val[0][0])-1])
                        grad_y = float(img[int(val[1][0])+1][int(val[0][0])]) - float(img[int(val[1][0])-1][int(val[0][0])])
                        # print(np.max(grad_x))


                        grad = [[grad_x, grad_y]]
        #
                        grad = np.array(grad)
                        pro1 = np.matmul(grad,jacobian)
        #
                        pro0 += np.matmul(pro1.T,dif2)
        #
                        hessian += np.matmul(pro1.T,pro1)
                pro2 = np.linalg.pinv(hessian)
                dp = np.matmul(pro2,pro0)


                p = p+dp.T
                # p /= np.max(p)
                warp = [[1+p[0][0],p[0][1],p[0][2]],
                        [p[0][3],1+p[0][4],p[0][5]]]
        #
                warp = np.array(warp)
            # print(warp)
            pt = [[0,0,1.0]]
            pt = np.array(pt)
            val = np.matmul(warp,pt.T)
            val2 = []
            val2.append([val[0][0]+pts[i][0],val[1][0]+pts[i][1]])
            val2 = np.array(val2)
            pts2.append(val2)
        #
        x1_ = 1e18
        y1_ = 1e18
        x2_ = -2
        y2_ = -2

        for i in range(len(pts2)):
            # print(i)
            # print(pts2[i][0])
            x1_ = min(pts2[i][0][0],x1_)
            x2_ = max(pts2[i][0][0],x2_)
            #
            y1_ = min(pts2[i][0][1],y1_)
            y2_ = max(pts2[i][0][1],y2_)
            cv2.circle(img33,(int(pts2[i][0][0]),int(pts2[i][0][1])),2,(255,0,0),-1)
        #
        cv2.imwrite("./out_girl/img{}.png".format(ct),img33)

        x1 = x1_
        y1 = y1_
        x2 = x2_
        y2 = y2_
        pts2 = np.array(pts2)
        pty = []
        for i in range(pts2.shape[0]):
            pty.append([pts2[i][0][0],pts2[i][0][1]])
        # print(pty)
        pts = []
        pts = list(pty)
        template = np.copy(img)
        # pts2 = find_features(imgp,int(x1),int(y1),int(x2),int(y2))
        # print(pts2)
        # pts3 = pts+pts2
        # kmeans = KMeans(n_clusters=min(30,len(pts)), random_state=4).fit(pts)
        # mean = kmeans.cluster_centers_
        # pts = pts2
        ct += 1
        # if(ct==20):
        #     break








if __name__=="__main__":
    main()
