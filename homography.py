import numpy as np
import cv2
from numpy.linalg import inv, svd
from similarity_transform import interpolate

def homography_t(img,fact):
    x,y,z = img.shape
    var_ =  (max(x,y))
    mat1 = [[0.6295, -0.0001, 0.0050],[0.0000, 0.3188, -0.3155],[0, 0.0001, 0.6344]]
    out_img = np.zeros([(var_),(var_),3])
    i1, j1 = 0, 0
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
            x1 /= z1
            y1 /= z1
            x1 += a
            y1 += a
            if(x1>=lowx and x1<highx-1 and y1>=lowy and y1<highy-1):
                ans_img[i][j] = interpolate(out_img,x1,y1)
    return ans_img

def main():
    img = cv2.imread('images3.png')
    img2 = homography_t(img,1)
    x = [0,0,999,999]
    y = [150,850,150,850]

    x_t = [227,227,731,723]
    y_t = [121,879,176,823]

    '''exact solution'''

    mat1 = []
    out_mat = []
    for i in range(4):
        mat1.append([0,0,0,-1*x[i],-1*y[i],-1,x[i]*y_t[i],y[i]*y_t[i]])
        mat1.append([x[i],y[i],1,0,0,0,-1*x[i]*x_t[i],-1*y[i]*x_t[i]])
        out_mat.append([-1*y_t[i]])
        out_mat.append([x_t[i]])

    mat1 = np.array(mat1)
    out_mat = np.array(out_mat)
    estimated_homography  = (np.matmul(np.matmul(inv(np.matmul(mat1.T,mat1)),mat1.T),out_mat))
    print(estimated_homography*.6344)
    #
    # '''overdetermined using svd'''
    tempmat = [[estimated_homography[0][0]*0.6344, estimated_homography[1][0]*0.6344, estimated_homography[2][0]*0.6344],
               [estimated_homography[3][0]*0.6344, estimated_homography[4][0]*0.6344, estimated_homography[5][0]*0.6344],
               [estimated_homography[6][0]*0.6344, estimated_homography[7][0]*0.6344, 1*0.6344]]
    tempimg = homography_t(img2,tempmat,2)
    cv2.imwrite('homo_img3.png',img2)
    cv2.imwrite('homo_re_img3.png',tempimg)

if __name__ == "__main__":
    main()
