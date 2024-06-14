import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Open video file
cap = cv2.VideoCapture("sample_video_2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the first frame

# Read the first frame
res, frame = cap.read()

# Select ROI in the first frame
region = cv2.selectROI("Select the area in the frame: ", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select the area in the frame: ")

x, y, w, h = [int(a) for a in region]
img_object = frame[int(y):int(y+h), int(x):int(x+w)]
    

sift = cv2.SIFT_create()
keypoints_object, descriptors_object = sift.detectAndCompute(img_object, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cumulative_keypoints = list(keypoints_object)
cumulative_descriptors = descriptors_object
plt.ion()
fig, ax = plt.subplots()
while True:
    
    ret, frame = cap.read()

    if not ret:
        break
    
    keypoints_scene, descriptors_scene = sift.detectAndCompute(frame, None)

    if descriptors_scene is not None and len(descriptors_scene) >= 2:
       
        matches = flann.knnMatch(descriptors_object, descriptors_scene, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 12:
           
            obj_pts = np.float32([keypoints_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            scene_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # Plot scene points
            ax.clear()
            scene_pts_plot = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches])
            ax.scatter(scene_pts_plot[:, 0], scene_pts_plot[:, 1], c='r', marker='o')
            ax.set_title("Scene Points")
            ax.set_xlim([0, frame.shape[1]])
            ax.set_ylim([frame.shape[0], 0])  # Invert y-axis to match image coordinates
            plt.pause(0.001)

            M, mask = cv2.findHomography(obj_pts, scene_pts, cv2.USAC_DEFAULT, 5.0)

            if M is not None:
                
                h, w = img_object.shape[:2]
                obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

               
                scene_corners = cv2.perspectiveTransform(obj_corners, M)

                
                frame = cv2.polylines(frame, [np.int32(scene_corners)], True, (0, 255, 0), 4, cv2.LINE_AA)
                scene_corners[:, 0, 0] = np.clip(scene_corners[:, 0, 0], 0, frame.shape[1] - 1)
                scene_corners[:, 0, 1] = np.clip(scene_corners[:, 0, 1], 0, frame.shape[0] - 1)
       
        img_matches = cv2.drawMatches(img_object, keypoints_object, frame, keypoints_scene, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Good Matches & Object detection", img_matches)
        if len(good_matches) > 15:
            print(scene_corners)
            x = scene_corners[0][0][0]
            y = scene_corners[0][0][1]
            if scene_corners[1][0][0] > scene_corners[0][0][0]:
                w = scene_corners[1][0][0] - scene_corners[0][0][0]
            else:
                w = -scene_corners[1][0][0] + scene_corners[0][0][0]
            if scene_corners[-1][0][1]> scene_corners[0][0][1]:
                h = scene_corners[-1][0][1] - scene_corners[0][0][1]
            else:
                h = -scene_corners[-1][0][1] + scene_corners[0][0][1]
            img_object = frame[int(y):int(y+h), int(x):int(x+w)]
            keypoints_object, descriptors_object = sift.detectAndCompute(img_object, None)
            if descriptors_object is not None and len(descriptors_object) > 0:
                        cumulative_keypoints.extend(keypoints_object)
                        cumulative_descriptors = np.vstack((cumulative_descriptors, descriptors_object))


    # Exit loop
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
