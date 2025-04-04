import cv2
import mediapipe as mp
import numpy as np
import torch

from classes import get_classes
from feature_extract import extract_feature_from_pose
from network import Net

def main():
    classes = get_classes()
    net = Net(len(classes))
    net.load_state_dict(torch.load('best_model.pt'))

    cam = cv2.VideoCapture(0)
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
            feat = torch.Tensor(extract_feature_from_pose(pose_tensor))
            with torch.no_grad():
                one_hot = net(feat)
                one_hot = torch.nn.functional.softmax(one_hot)
                y = 30
                for probability, class_name in zip(one_hot, classes):
                    text = f'{probability:.2f} {class_name}'
                    cv2.putText(bgr_img, text, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
                    y += 30

            mp.solutions.drawing_utils.draw_landmarks(bgr_img, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)


        cv2.imshow('image', bgr_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
