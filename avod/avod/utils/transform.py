import numpy as np

def projectToImage(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D
    
def clip3DwithinImage(pts_3D,P,image_size):
    """
    WZN: first project 3D points to image than return index 
    that keep only the points visible to the image
    input see projectToImage(), image_size should be [side,height]
    return the indices of pts_3D
    """
    pts_2D = projectToImage(pts_3D,P)
    indices = np.logical_and(pts_2D[0,:]<image_size[0]-1,pts_2D[0,:]>=0)
    indices = np.logical_and(indices,pts_2D[1,:]>=0)
    indices = np.logical_and(indices,pts_2D[1,:]<image_size[1]-1)
    #print np.amax(pts_2D[:,indices],axis=1), 'image size: ', image_size
    return indices