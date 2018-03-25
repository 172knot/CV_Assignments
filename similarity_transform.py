import cv2
import numpy as np
from numpy.linalg import inv
import math
from PIL import Image

def interpolate(img_temp,x,y):
    sum_0, sum_1, sum_2 = 0, 0 ,0
    sum_0 += img_temp[math.floor(y)][math.floor(x)][0]
    sum_1 += img_temp[math.floor(y)][math.floor(x)][1]
    sum_2 += img_temp[math.floor(y)][math.floor(x)][2]

    sum_0 += img_temp[math.ceil(y)][math.floor(x)][0]
    sum_1 += img_temp[math.ceil(y)][math.floor(x)][1]
    sum_2 += img_temp[math.ceil(y)][math.floor(x)][2]

    sum_0 += img_temp[math.floor(y)][math.ceil(x)][0]
    sum_1 += img_temp[math.floor(y)][math.ceil(x)][1]
    sum_2 += img_temp[math.floor(y)][math.ceil(x)][2]

    sum_0 += img_temp[math.ceil(y)][math.ceil(x)][0]
    sum_1 += img_temp[math.ceil(y)][math.ceil(x)][1]
    sum_2 += img_temp[math.ceil(y)][math.ceil(x)][2]

    sum_0 /= 4
    sum_1 /= 4
    sum_2 /= 4
    sum_ = [sum_0, sum_1, sum_2]
    return sum_

def similarity_t(img,angle=0,sx=1,sy=1,tx=0,ty=0):
    x,y,z=img.shape
    mat1 = [[sx*math.cos(angle*np.pi/180.0), -1*(math.sin(angle*np.pi/180.0)), tx],
            [math.sin(angle*np.pi/180.0), sy*math.cos(angle*np.pi/180.0), ty],
            [0, 0, 1]]
    fact = max(max(sx,sy),1)
    fact = 0
    var_ =  2*(int(fact)+1)*max(x,y)
    out_img = np.zeros([(var_),(var_),3])

    i1, j1=0, 0
    a = int(var_/2)
    for i in range(a-int(x/2),a+int(x/2)):
        j1 = 0
        for j in range(a-int(y/2),a+int(y/2)):
            out_img[i][j] = img[i1][j1]
            j1 += 1
        i1 += 1
    lowx = a-int(x/2)
    highx = a+int(x/2)
    lowy = a-int(y/2)
    highy = a+int(y/2)
    ans_img = np.zeros([var_,var_,3])
    mat1_inv = inv(mat1)
    for i in range(2*a):
        for j in range(2*a):
            mat_temp = [[i-a],[j-a],[1]]
            x1, y1, z1 = np.matmul(mat1_inv, mat_temp)
            x1 += a
            y1 += a
            if(x1>=lowx and x1<highx-1 and y1>=lowy and y1<highy-1):
                ans_img[i][j] = interpolate(out_img,x1,y1)
    return ans_img

def main():
    img = cv2.imread('images3.png')
    x,y,z = img.shape
    ang = input("Enter angle of rotation")
    ang = float(ang)

    tx = input("Enter Shift along x")
    tx = float(tx)
    ty = input("Enter Shift along y")
    ty = float(ty)

    sx = input("Enter Scaling in x")
    sx = float(sx)
    sy = input("Enter Scaling in y")
    sy = float(sy)

    img2 = similarity_t(img,ang,sx,sy,tx,ty)
    print(img2.shape)
    cv2.imwrite('rot_img3.png',img2)

if __name__ == "__main__":
    main()
