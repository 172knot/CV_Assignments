import numpy as np
import cv2
import os

path_fr  = "./Basketball/img"
path_gt = "./Basketball/groundtruth_rect.txt"

def calc_grad(img):
    img_new = np.zeros((img.shape[0]+2,img.shape[1]+2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_new[i+1][j+1] = img[i][j]


    Ix = np.zeros((img.shape[0],img.shape[1]))
    Iy = np.zeros((img.shape[0],img.shape[1]))
    Ix = np.array(Ix)
    Iy = np.array(Iy)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Ix[i][j] = img_new[i+1][j+2] - img_new[i+1][j]
            Iy[i][j] = img_new[i+2][j+1] - img_new[i][j+1]


    return Ix, Iy

def get_crop_image(img,x1,y1,x2,y2):
    temp = np.zeros((y2-y1, x2-x1))
    temp = np.array(temp)
    for i in range(x1,x2):
        for j in range(y1,y2):
            temp[j-y1][i-x1] = img[j][i]
    temp = np.array(temp)
    return temp

def main():

    fp = open(path_gt)
    Z = fp.readline().split(",")
    x, y , w, h = int(Z[0]),  int(Z[1]), int(Z[2]), int(Z[3].split("\n")[0])
    x1, y1, x2, y2 = x, y, x+w, y+h

    img = cv2.imread(os.path.join(path_fr,"0001.jpg"),0)
    img_cr = get_crop_image(img,x1,y1,x2,y2)
    T = img_cr
    T = np.array(T)
    print(x1,y1,x2,y2)
    ct = 0
    for imgs in sorted(os.listdir(path_fr)):
        if(ct==0):
            ct += 1
            continue
        img = cv2.imread(os.path.join(path_fr,imgs),0)

        p = [0.0,0.0,0.0,0.0,0.0,0.0]
        p = np.reshape(p,(1,6))
        I = np.zeros((y2-y1,x2-x1))
        I2 = np.zeros((y2-y1,x2-x1))

        for ite in range(20):
            print(p)
            warp = [[1+p[0][0],p[0][1],p[0][2]],
                    [p[0][3],1+p[0][4],p[0][5]]]

            warp = np.array(warp)
            for i in range(y1,y2):
                for j in range(x1,x2):
                    pt1 = [[i],
                          [j],
                          [1]]
                    pt1 = np.array(pt1)
                    warp = np.array(warp)
                    val = np.matmul(warp,pt1)
                    I[i-y1][j-x1] = img[int(val[0][0])][int(val[1][0])]

            I = np.array(I)
            I2 = I
            calc1 = [0.0,0.0,0.0,0.0,0.0,0.0]
            calc2 = np.zeros((6,6))
            calc1 = np.array(calc1)
            calc2 = np.array(calc2)
            calc1 = np.reshape(calc1,(1,6))

            x,y = calc_grad(T)


            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    temp = [x[i][j],y[i][j]]
                    temp = np.array(temp)
                    temp = np.reshape(temp,(1,2))
                    jaco = [[i,j,1,0,0,0],
                            [0,0,0,i,j,1]]
                    jaco = np.array(jaco)
                    ans = np.matmul(temp,jaco)
                    temp = [I[i][j] - T[i][j]]
                    temp = np.reshape(temp,(1,1))
                    ans  = np.matmul(ans.T,temp)
                    calc1 += ans.T
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    temp = [x[i][j],y[i][j]]
                    temp = np.array(temp)
                    temp = np.reshape(temp,(1,2))
                    jaco = [[i,j,1,0,0,0],
                            [0,0,0,i,j,1]]
                    jaco = np.array(jaco)
                    ans = np.matmul(temp,jaco)
                    ans = np.matmul(ans.T,ans)
                    ans = np.linalg.pinv(ans)
                    calc2 +=  ans
            dp = np.matmul(calc2,calc1.T)
            ty = np.abs(dp)
            ma = np.max(ty)
            dp = (dp)/((ma))
            dp = np.tanh(dp)
            # print(dp)
            p += dp.T
            T = I
        # print(p)
        I2 = np.array(I2)
        cv2.imwrite("img{}.png".format(ct),I2)
            # print(p)



        print(ct)
        ct+=1
        if(ct==2):
            break




















if(__name__=="__main__"):
    main()
