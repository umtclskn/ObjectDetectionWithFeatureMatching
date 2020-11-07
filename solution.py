import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class Solution(object):

        def __init__(self):
            print("Solution initialized...")

        def localize_object(self, image_object, image_scene):
            img1 = np.copy(image_object)
            img2 = np.copy(image_scene)

            img1 = cv.GaussianBlur(img1,(5,5),cv.BORDER_DEFAULT)
            img2 = cv.GaussianBlur(img2,(5,5),cv.BORDER_DEFAULT)

            sift = cv.SIFT_create(50000)

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)

            #-- Matching descriptor vectors with a FLANN based matcher
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            knn_matches = matcher.knnMatch(des1, des2, 2)
            #-- Filter matches using the Lowe's ratio test
            ratio_thresh = 0.8
            good_matches = []
            for m,n in knn_matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            if len(good_matches) <= 4:
                return []
            
            obj = np.empty((len(good_matches),2), dtype=np.float32)
            scene = np.empty((len(good_matches),2), dtype=np.float32)
            for i in range(len(good_matches)):
                #-- Get the keypoints from the good matches
                obj[i,0] = kp1[good_matches[i].queryIdx].pt[0]
                obj[i,1] = kp1[good_matches[i].queryIdx].pt[1]
                scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
                scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]
            H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
            #-- Get the corners from the image_1 ( the object to be "detected" )
            obj_corners = np.empty((4,1,2), dtype=np.float32)
            obj_corners[0,0,0] = 0
            obj_corners[0,0,1] = 0
            obj_corners[1,0,0] = img1.shape[1]
            obj_corners[1,0,1] = 0
            obj_corners[2,0,0] = img1.shape[1]
            obj_corners[2,0,1] = img1.shape[0]
            obj_corners[3,0,0] = 0
            obj_corners[3,0,1] = img1.shape[0]
            scene_corners = cv.perspectiveTransform(obj_corners, H)

            result_points = [
                (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])),  
                (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])),
                (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])), 
                (int(scene_corners[3,0,0]), int(scene_corners[3,0,1]))
            ]

            return result_points



if __name__ == '__main__':
    img1 = cv.imread('Small_area.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('StarMap.png', cv.IMREAD_GRAYSCALE)

    solution = Solution()
    points = solution.localize_object(img1, img2)
    print(points)
