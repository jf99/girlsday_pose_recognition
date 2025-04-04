from math import sqrt
import mediapipe as mp
import numpy as np

def distance(pose_tensor, from_index, to_index):
    dx = pose_tensor[from_index, 0] - pose_tensor[to_index, 0]
    dy = pose_tensor[from_index, 1] - pose_tensor[to_index, 1]
    dz = pose_tensor[from_index, 2] - pose_tensor[to_index, 2]
    return sqrt(dx * dx + dy * dy + dz * dz)

def extract_visibilities(pose_tensor):
    return pose_tensor[:, 3]

def extract_feature_from_pose(pose_tensor):
    assert len(pose_tensor.shape) == 2
    assert pose_tensor.shape[0] == 33 and pose_tensor.shape[1] == 4
    lm = mp.solutions.pose.PoseLandmark
    distances = [
        # left to right
        distance(pose_tensor, lm.LEFT_HEEL, lm.RIGHT_HEEL),
        distance(pose_tensor, lm.LEFT_KNEE, lm.RIGHT_KNEE),
        distance(pose_tensor, lm.LEFT_THUMB, lm.RIGHT_THUMB),
        distance(pose_tensor, lm.LEFT_ELBOW, lm.RIGHT_ELBOW),

        # left thumb
        distance(pose_tensor, lm.LEFT_THUMB, lm.LEFT_HEEL),
        distance(pose_tensor, lm.LEFT_THUMB, lm.LEFT_KNEE),
        distance(pose_tensor, lm.LEFT_THUMB, lm.LEFT_WRIST),
        distance(pose_tensor, lm.LEFT_THUMB, lm.LEFT_SHOULDER),
        distance(pose_tensor, lm.LEFT_THUMB, lm.LEFT_EAR),
        distance(pose_tensor, lm.LEFT_THUMB, lm.NOSE),

        # right thumb
        distance(pose_tensor, lm.RIGHT_THUMB, lm.RIGHT_HEEL),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.RIGHT_KNEE),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.RIGHT_WRIST),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.RIGHT_SHOULDER),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.RIGHT_EAR),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.NOSE),

        # vertical distances
        distance(pose_tensor, lm.NOSE, lm.LEFT_SHOULDER),
        distance(pose_tensor, lm.NOSE, lm.RIGHT_SHOULDER),
        distance(pose_tensor, lm.NOSE, lm.LEFT_WRIST),
        distance(pose_tensor, lm.NOSE, lm.RIGHT_WRIST),
        distance(pose_tensor, lm.NOSE, lm.LEFT_KNEE),
        distance(pose_tensor, lm.NOSE, lm.RIGHT_KNEE),
        distance(pose_tensor, lm.LEFT_SHOULDER, lm.LEFT_KNEE),
        distance(pose_tensor, lm.RIGHT_SHOULDER, lm.RIGHT_KNEE),
        distance(pose_tensor, lm.LEFT_SHOULDER, lm.LEFT_HEEL),
        distance(pose_tensor, lm.RIGHT_SHOULDER, lm.RIGHT_HEEL),
        distance(pose_tensor, lm.LEFT_WRIST, lm.LEFT_HEEL),
        distance(pose_tensor, lm.RIGHT_WRIST, lm.RIGHT_HEEL),

        # crossed limbs
        distance(pose_tensor, lm.LEFT_THUMB, lm.RIGHT_SHOULDER),
        distance(pose_tensor, lm.LEFT_THUMB, lm.RIGHT_WRIST),
        distance(pose_tensor, lm.LEFT_THUMB, lm.RIGHT_HEEL),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.LEFT_SHOULDER),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.LEFT_WRIST),
        distance(pose_tensor, lm.RIGHT_THUMB, lm.LEFT_HEEL),
        distance(pose_tensor, lm.LEFT_HEEL, lm.RIGHT_KNEE),
        distance(pose_tensor, lm.LEFT_KNEE, lm.RIGHT_SHOULDER),
        distance(pose_tensor, lm.RIGHT_HEEL, lm.LEFT_KNEE),
        distance(pose_tensor, lm.RIGHT_KNEE, lm.LEFT_SHOULDER)
    ]
    #print(distances)

    visibilities = extract_visibilities(pose_tensor)
    #print(visibilities)
    return np.concatenate([distances, visibilities])
