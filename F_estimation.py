import cv2
import numpy as np
from numpy.linalg import svd

def main():
    ''' Fundamental Matrix Estimation'''

    im1 = cv2.imread("house1.jpg")
    im2 = cv2.imread("house2.jpg")

    X1 = [[187,44],[313,63],[279,168],[83,112], [110,163],[312,195],[276,248],[239,181]]
    X2  = [[186,44],[289,66],[238,175],[99,107],[117,156],[291,206],[239,258],[210,185]]


    for i in range(8):
        cv2.circle(im1, (X1[i][0],X1[i][1]), 2, (255, 0, 255), -1)
        cv2.circle(im2, (X2[i][0],X2[i][1]), 2, (255, 0, 255), -1)

    cv2.imwrite("house11.png",im1)
    cv2.imwrite("house22.png",im2)

    A = []
    X2 = np.array(X2)
    X1 = np.array(X1)
    for i in range(X2.shape[0]):
        A.append([X2[i][0]*X1[i][0], X2[i][1]*X1[i][0], X1[i][0], X2[i][0]*X1[i][1], X2[i][1]*X1[i][1], X1[i][1], X2[i][0], X2[i][1], 1])

    A = np.array(A)
    g1 = [X1[0][0],X1[0][1],1]
    g2 = [X2[0][0],X2[0][1],1]
    g1 = np.array(g1)
    g2 = np.array(g2)

    u,d,v = svd(A)
    v = v.T
    F = v[:,8]
    F = np.array(F)
    F = np.reshape(F,(3,3))
    F /= F[2][2]

    u,d,v = svd(F)
    D = np.diag(d)
    D[2][2] = 0
    F_ = np.matmul(np.matmul(u,D),v)
    F_ /= F_[2][2]
    im11 = im1
    im22 = im2
    L1 = []
    L2 = []

    for i in range(8):
        g1 = [X1[i][0],X1[i][1],1]
        g2 = [X2[i][0],X2[i][1],1]
        g1 = np.array(g1)
        g2 = np.array(g2)

        m1 = np.matmul(g1,F_)
        m2 = np.matmul(F_,g2)
        m1 /= m1[2]
        m2 /= m2[2]
        L1.append(m2)
        L2.append(m1)
        p1 = []
        p2 = []
        for i in range(im1.shape[1]):
            t = (-1 -(i*m2[0]))/m2[1]
            t = round(t)
            if(t<im1.shape[0] and t>=0):
                p1.append([i,int(t)])
        for i in range(im2.shape[1]):
            t = (-1 -(i*m1[0]))/m1[1]
            t = round(t)
            if(t<im2.shape[0] and t>=0):
                p2.append([i,int(t)])
        cv2.line(im11, (p1[0][0],p1[0][1]), (p1[len(p1)-1][0],p1[len(p1)-1][1]), (255, 0, 0), thickness=1, lineType=8, shift=0)
        cv2.line(im22, (p2[0][0],p2[0][1]), (p2[len(p2)-1][0],p2[len(p2)-1][1]), (255, 0, 0), thickness=1, lineType=8, shift=0)

    cv2.imwrite("im1_e.png",im11)
    cv2.imwrite("im2_e.png",im22)

    ''' Estimating the Camera Projecection Matrices'''
    e1 = []
    e2 = []
    em1 = []
    em2 = []

    y = (L1[0][0]-L1[1][0])/((L1[0][1]*L1[1][0])-(L1[0][0]*L1[1][1]))
    x = (-1 - (L1[0][1]*y))/L1[0][0]
    e1 = [x,y,1]

    y = (L2[0][0]-L2[1][0])/((L2[0][1]*L2[1][0])-(L2[0][0]*L2[1][1]))
    x = (-1 - (L2[0][1]*y))/L2[0][0]
    e2 = [x,y,1]

    em1 = [[0, -1, e1[1]],[1,0,-1*e1[0]],[-1*e1[1],e1[0],0]]
    em2 = [[0, -1, e2[1]],[1,0,-1*e2[0]],[-1*e2[1],e2[0],0]]
    em1 = np.array(em1)
    em2 = np.array(em2)

    P2 = [[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0]]
    P2 = np.array(P2)


    t__ = np.matmul(em1,F_)
    a__ = [[e1[0]],[e1[1]],[e1[2]]]
    a__ = np.array(a__)
    P1 = np.hstack((t__,a__))
    print(np.matmul(np.matmul(P1.T,F_),P2))



    '''Estimating the world coordinates subject to a projective transformation'''
    Xp = []
    for i in range(8):
        mat1 = [[X2[i][1]*P2[2][0] - P2[1][0], X2[i][1]*P2[2][1] - P2[1][1], X2[i][1]*P2[2][2] - P2[1][2], X2[i][1]*P2[2][3] - P2[1][3]],
                [-1*X2[i][0]*P2[2][0] + P2[0][0], -1*X2[i][0]*P2[2][1] + P2[0][1], -1*X2[i][0]*P2[2][2] + P2[0][2], -1*X2[i][0]*P2[2][3] + P2[0][3]],
                [X1[i][1]*P1[2][0] - P1[1][0], X1[i][1]*P1[2][1] - P1[1][1], X1[i][1]*P1[2][2] - P1[1][2], X1[i][1]*P1[2][3] - P1[1][3]],
                [-1*X1[i][0]*P1[2][0] + P1[0][0], -1*X1[i][0]*P1[2][1] + P1[0][1], -1*X1[i][0]*P1[2][2] + P1[0][2], -1*X1[i][0]*P1[2][3] + P1[0][3]]]

        mat1 = np.array(mat1)
        u,d,v = svd(mat1)
        v = v.T
        pt1 = v[:,3]
        pt1 /= pt1[3]
        Xp.append(pt1)

    '''Re-Projection of points back on to the image plane'''

    im1_ = cv2.imread("house1.jpg")
    im2_ = cv2.imread("house2.jpg")
    for i in range(8):
        p1 = np.matmul(P1,Xp[i])
        p1 /= p1[2]
        p2 = np.matmul(P2,Xp[i])
        p2 /= p2[2]
        cv2.circle(im1_, (int(p1[0]),int(p1[1])), 2, (255, 0, 255), -1)
        cv2.circle(im2_, (int(p2[0]),int(p2[1])), 2, (255, 0, 255), -1)

    cv2.imwrite("im1_p.png",im1_)
    cv2.imwrite("im2_p.png",im2_)



if __name__=="__main__":
    main()
