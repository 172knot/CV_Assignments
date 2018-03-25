import numpy as np
import cv2
from numpy.linalg import inv
from similarity_transform import interpolate

def affine_t(img):
    x,y,z = img.shape
    var_ =  2*(max(x,y))
    out_img = np.zeros([(var_),(var_),3])
    i1, j1 = 0, 0
    a = int(var_/2)
    for i in range(a-int(x/2),a+int(x/2)):
        j1 = 0
        for j in range(a-int(y/2),a+int(y/2)):
            out_img[i][j] = img[i1][j1]
            j1 += 1
        i1 += 1
    cv2.imwrite('affine_input_img3.png',out_img)
    mat1 = [[1.2, 0.67, 0.89],[1.34, 0, 1],[0, 0, 1]]
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
    img2 = affine_t(img)
    cv2.imwrite('affine_img3.png',img2)

    '''exact prediction'''
    map_mat = [[528, 528, 1473],[243, 912, 1758],[1, 1, 1]]
    init_mat = [[500, 500, 1500],[619, 1325, 1325],[1, 1, 1]]

    inv_init_mat = inv(init_mat)
    estimated_affine_matrix = np.matmul(map_mat, inv_init_mat)

    print(estimated_affine_matrix)

if __name__ == "__main__":
    main()


    # import tensorflow as tf
    # import numpy as np
    # import pickle
    # import os
    # import cv2

    # # def unpickle(file):
    # #     with open(file, 'rb') as fo:
    # #         dict = pickle.load(fo, encoding='bytes')
    # #     return dict
    #
    # #def main():
    # path = '/home/knot/Documents/Semester6/Dl/Assignment1/cifar-10-python/cifar-10-batches-py/train_data'
    #     #for files in os.listdir(path):
    #     #    dict = unpickle(os.path.join(path,files))
    #
    # # if __name__ == "__main__":
    # #     main()
