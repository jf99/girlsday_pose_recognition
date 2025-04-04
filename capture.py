import cv2
import json
import mediapipe as mp
import numpy as np
import os

save_dir = 'capturings'

def main():
    cam = cv2.VideoCapture(0)
    frameId = 0
    while cam.isOpened():
        success, bgr_img = cam.read()
        if success == False:
            continue
        bgr_img = cv2.flip(bgr_img, 1)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5) as mp_pose:
            result = mp_pose.process(rgb_img)
            if not result.pose_landmarks:
                continue

            pose_tensor = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark], dtype='float32')

            filename = os.path.join(save_dir, 'frame' + str(frameId) + '.npy')
            file = open(filename, 'wb')
            np.save(file, pose_tensor)
            print(f'frame {frameId}')
            frameId += 1

            mp.solutions.drawing_utils.draw_landmarks(bgr_img, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)


        cv2.imshow('image', bgr_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
