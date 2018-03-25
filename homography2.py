import numpy as np
import cv2
from numpy.linalg import inv, svd, pinv
from similarity_transform import interpolate
gt = 0
def padding(img):
    var_ =  2000
    out_img = np.zeros([(var_),(var_),3])
    y,x,z = img.shape
    i1, j1 = 0, 0
    a = int(var_/2)
    global gt
    gt = a
    for i in range(a-int(y/2),a+int(y/2)):
        j1 = 0
        for j in range(a-int(x/2),a+int(x/2)):
            out_img[i][j] = img[i1][j1]
            j1 += 1
        i1 += 1
    cv2.imwrite("padded_input.png",out_img)
    return out_img

def homography_t(img,mat1 = [[0.6295, -0.0001, 0.0050],
                             [0.0000, 0.3188, -0.3155],
                             [0, 0.0001, 0.6344]]):
    var_ =  2000
    ans_img = np.zeros([var_,var_,3])
    mat1 =  np.array(mat1)
    mat1_inv = pinv(mat1)
    a = int(var_/2)
    for i in range(var_):
        for j in range(var_):
            mat_temp = [[j-a],[i-a],[1]]
            x1, y1, z1 = np.matmul(mat1_inv, mat_temp)
            x1 /= z1
            y1 /= z1
            x1 += a
            y1 += a
            if(x1>=0 and x1<var_-1 and y1>=0 and y1<var_-1):
                ans_img[i][j] = interpolate(img,x1,y1)
    return ans_img

def homography_estimation(X_, X, pt=1):
    '''exact solution'''
    if(pt==1):
        mat1 = []
        out_mat = []
        for i in range(4):
            mat1.append([0,0,0,-1*X_[i][0],-1*X_[i][1],-1,X_[i][0]*X[i][1],X_[i][1]*X[i][1]])
            mat1.append([X_[i][0],X_[i][1],1,0,0,0,-1*X_[i][0]*X[i][0],-1*X_[i][1]*X[i][0]])
            out_mat.append([-1*X[i][1]])
            out_mat.append([X[i][0]])
        #
        mat1 = np.array(mat1)
        out_mat = np.array(out_mat)
        estimated_homography = np.matmul(pinv(mat1),out_mat)
        return estimated_homography

    '''overdetermined using svd'''

    if(pt==2):
        mat2 = []
        out_mat2 = []
        for i in range(4):
            mat2.append([X_[i][0],X_[i][1],1,0,0,0,-1*X_[i][0]*X[i][0],-1*X_[i][1]*X[i][0],-1*X[i][0]])
            mat2.append([0,0,0,X_[i][0],X_[i][1],1,-1*X_[i][0]*X[i][1],-1*X_[i][1]*X[i][1],-1*X[i][1]])

        mat2 = np.array(mat2)
        u,s,v = svd(mat2)
        v = v.T
        h = v[:,8]
        h = np.array(h)
        h = np.reshape(h,(9,1))
        return h


    if(pt==3):
        mat2 = []
        out_mat2 = []
        for i in range(n):
            mat2.append([0,0,0,-1*x[i],-1*y[i],-1,x[i]*y_t[i],y[i]*y_t[i],y_t[i]])
            mat2.append([x[i],y[i],1,0,0,0,-1*x[i]*x_t[i],-1*y[i]*x_t[i],-1*x_t[i]])

        mat2 = np.array(mat2)
        out_mat2 = np.array(out_mat2)
        u,s,v = svd(mat2)
        v = v.T
        h = v[:,8]
        return h

def main():
    img = cv2.imread('images3.png')
    img = padding(img)
    img2= homography_t(img)
    cv2.imwrite('homography_transformed_image.png',img2)

    global gt
    X_ = [[500-gt, 647-gt],[1500-gt, 647-gt],[500-gt, 1360-gt],[1500-gt, 1360-gt]]
    X  = [[474-gt, 812-gt],[1526-gt, 812-gt],[529-gt, 1168-gt],[1471-gt, 1168-gt]]

    h = homography_estimation(X_, X,2)
    h *= 0.6344

    tempmat = [[h[0][0], h[1][0], h[2][0]],
               [h[3][0], h[4][0], h[5][0]],
               [h[6][0], h[7][0], h[8][0]]]
    print(tempmat)

    tempimg = homography_t(img2,pinv(tempmat))
    cv2.imwrite('homo_re_img3_svd.png',tempimg)

if __name__ == "__main__":
    main()
