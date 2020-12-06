import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *

import sys

np.set_printoptions(threshold=sys.maxsize)


print(" \nTo work: images like frame_0001.png, frame_0002.png must be in data/rgb folder.\n")
print ("To work: depth maps for every image, like frame_0001.dat, frame_0002.dat must be in data/depth folder.\n")

dataset_handler = DatasetHandler()



# Shows an Image and Depth Map of Corresponding Image.
i = 0

image = dataset_handler.images_rgb[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')

depth = dataset_handler.depth_maps[i]
plt.title("\n Depth Map and Image " + str(i))
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(depth, cmap='jet')



print("Depth map shape: {0}".format(depth.shape))

v, u = depth.shape
depth_val = depth[v-1, u-1]
print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}".format(i, depth_val))


dataset_handler.k

# Number of frames in the dataset.
print("Number of frames in the dataset: " + str(dataset_handler.num_frames))




### FEATURE EXTRACTION
def extract_features(image):
    """
    Finds keypoints and descriptors for the image.

    Arguments:
    image -- a grayscale image.

    Returns:
    kp -- list of the extracted keypoints (features) in an image.
    des -- list of the keypoint descriptors in an image.
    """

    surf = cv.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(image,None)
    
    return kp, des


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    print(len(kp))
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)
    


def extract_features_dataset(images, extract_features_function):
    """
    Finds keypoints and descriptors for each image in the dataset.

    Arguments:
    images -- a list of grayscale images.
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image.

    Returns:
    kp_list -- a list of keypoints for each image in images.
    des_list -- a list of descriptors for each image in images.
    
    """
    kp_list = []
    des_list = []
    
    for i in range (0, len(images)):
        kp_list.append([])
        des_list.append([])

    for i in range (0, len(images)):
        kp_list[i], des_list[i]= extract_features_function(images[i])
    
    return kp_list, des_list


### FEATURE MATCHING
def match_features(des1, des2):
    """
    Matches features from two images.

    Arguments:
    des1 -- list of the keypoint descriptors in the first image.
    des2 -- list of the keypoint descriptors in the second image.

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor.
    """
    
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    
    # Match descriptors.
    match = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    match = sorted(match, key = lambda x:x.distance)

    return match


def filter_matches_distance(match, dist_threshold):
    """
    Filters matched features from two images by distance between the best matches.

    Arguments:
    match -- list of matched features from two images.
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0). 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold.
    """
    filtered_match = []

    for m in match:
        if m.distance < dist_threshold:
            filtered_match.append(m)

    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualizes corresponding matches in two images.

    Arguments:
    image1 -- the first image in a matched image pair.
    kp1 -- list of the keypoints in the first image.
    image2 -- the second image in a matched image pair.
    kp2 -- list of the keypoints in the second image.
    match -- list of matched features from the pair of images.

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if this function is not used.
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match, None)
    print(len(match))
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    


# Matching Features in Each Subsequent Image Pair in the Dataset.
def match_features_dataset(des_list, match_features):
    """
    Matches features for each subsequent image pair in the dataset.

    Arguments:
    des_list -- a list of descriptors for each image in the dataset.
    match_features -- a function which maches features between a pair of images.

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1.
               
    """
    matches = []
    
    for i in range (0, len(images)):
        matches.append([])

    for i in range (0, (len(images)-1)):
        matches[i]=match_features(des_list[i], des_list[i+1])   
    
    return matches



matches = match_features_dataset(des_list, match_features)

i = 0
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))


print(len(matches))




def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filters matched features by distance for each subsequent image pair in the dataset.

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches.
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1.
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold.
               
    """
    filtered_matches = []

    for i in range (0, len(matches)):
        filtered_matches.append([])
        
    for i in range (0,len(matches)):
        filtered_matches[i]=filter_matches_distance(matches[i],dist_threshold)
    
    return filtered_matches


### TRAJECTORY ESTIMATION
def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimates camera motion from a pair of subsequent image frames.

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame.

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix.
    tvec -- recovered 3x1 translation numpy vector.
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system.
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system.
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
 
    objectpoints = np.zeros((3, len(match)))
    i = 0
    for mi in match:
        queryidx = mi.queryIdx
        trainidx = mi.trainIdx
        
        x1, y1 = kp1[queryidx].pt

        x2, y2 = kp2[trainidx].pt
            
            
        Z = depth1[int(y1), int(x1)]
       
        if Z < 900:
            

            image1_points.append([x1, y1])
            
            x2, y2 = kp2[trainidx].pt
            image2_points.append([x2, y2])
            
            scaled_coord = np.dot(np.linalg.inv(k), np.array([x1, y1, 1]))
            cali_coord = float(Z)/scaled_coord[2]*scaled_coord

            objectpoints[:, i] = cali_coord
            i = i + 1
        else:
            continue

    objectpoints = objectpoints[:,:i]
    print("Size of match, depth below 900: {0}".format(i))
    objectpoints = objectpoints.T
    imagepoints = np.array(image2_points)
    
    print(objectpoints.shape)   #Nx3
    print(imagepoints.shape)     #Nx2
    
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)

    rmat, _ = cv2.Rodrigues(rvec)
    
    
    return rmat, tvec, image1_points, image2_points




# Uses visualize_camera_movement() from m2bk.py 
i=48
image1  = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)




image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
# These visualizations might be helpful for understanding the quality of image points selected for the camera motion estimation.


# Camera Trajectory Estimation
def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimates complete camera trajectory from subsequent image pairs.

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame.

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is given 
                  at the initialization of this function

    """
    trajectory = np.zeros((3, 1))
    
    
    rt=np.eye(4)
    
    rt_i=np.empty((4,4))
    
    for i in range(len(matches)-1):
        
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth = depth_maps[i]
    
        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth)
        
        rt_i[:3, :3] = rmat

        rt_i[:3, 3] = tvec.ravel()

        rt_i[3, :] = [0, 0, 0, 1]
        

        
        rt = np.dot(rt, np.linalg.inv(rt_i))
        
        new_trajectory=rt[:3,3]
        

        new_trajectory=new_trajectory.reshape((3,1))
        
        trajectory=np.append(trajectory,new_trajectory, axis=1)
    
    return trajectory





depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

print("Length of trajectory: {0}".format(trajectory.shape[1]))







print("\n \n \nSUMMARY")

dataset_handler = DatasetHandler()


# Part 1. Features Extraction
images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)


# Part II. Feature Matching
matches = match_features_dataset(des_list, match_features)

# Sets to True if you want to use filtered matches or False otherwise
is_main_filtered_m = True
if is_main_filtered_m:
    dist_threshold = 0.3
    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches

    
# Part III. Trajectory Estimation
depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)


# Prints Trajectory Points
print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))


#Visualizes Trajectory with visualize_trajectory() from m2bk.py
visualize_trajectory(trajectory)