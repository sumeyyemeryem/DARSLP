import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic # Holistic model
# video_shape = {"width":149, "height":166}

video_shape = {"width":210, "height":260}

normal_palet = {
    'face': (243, 236, 176),
    "pose": (176, 30, 104),
    "left_hand": (255,0, 0),
    "right_hand":(255, 0, 0),
    "line": (173, 231, 146)
}
red_palet = {
    'face': (255,0, 0),
    "pose": (255,0, 0),
    "left_hand": (255,0, 0),
    "right_hand":(255, 0, 0),
    "line": (255,0, 0)
}
blue_palet = {
    'face': (0, 0, 255),
    "pose": (0, 0, 255),
    "left_hand": (0, 0, 255),
    "right_hand":(0, 0, 255),
    "line": (0, 0, 255)
}

mp_face_mesh = mp.solutions.face_mesh

# Pose keypoints (upper body only)
POSE_UPPER_BODY_IDX = [11, 13, 491, 12, 14, 512, 23, 24]  # [left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist, left hip, right hip] 

def extract_indices(connections):
    return sorted(set([idx for conn in connections for idx in conn]))

# Extract indices for facial regions
HEAD_CONTOUR_IDX = extract_indices(mp_face_mesh.FACEMESH_FACE_OVAL)
LEFT_EYEBROW_IDX = extract_indices(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_EYEBROW_IDX = extract_indices(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
LEFT_EYE_IDX = extract_indices(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = extract_indices(mp_face_mesh.FACEMESH_RIGHT_EYE)
OUTER_LIPS_IDX = extract_indices(mp_face_mesh.FACEMESH_LIPS)

FACE_SELECTED_IDX = sorted(set(
    HEAD_CONTOUR_IDX + LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX +
    LEFT_EYE_IDX + RIGHT_EYE_IDX + OUTER_LIPS_IDX
))

def reindex_connections(connections, selected_idx):
    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_idx)}
    reindexed = []
    for start, end in connections:
        if start in index_map and end in index_map:
            reindexed.append((index_map[start], index_map[end]))
    return reindexed

mp_face_mesh = mp.solutions.face_mesh

HEAD_CONTOUR_EDGES = reindex_connections(mp_face_mesh.FACEMESH_FACE_OVAL, FACE_SELECTED_IDX)
LEFT_EYEBROW_EDGES = reindex_connections(mp_face_mesh.FACEMESH_LEFT_EYEBROW, FACE_SELECTED_IDX)
RIGHT_EYEBROW_EDGES = reindex_connections(mp_face_mesh.FACEMESH_RIGHT_EYEBROW, FACE_SELECTED_IDX)
LEFT_EYE_EDGES = reindex_connections(mp_face_mesh.FACEMESH_LEFT_EYE, FACE_SELECTED_IDX)
RIGHT_EYE_EDGES = reindex_connections(mp_face_mesh.FACEMESH_RIGHT_EYE, FACE_SELECTED_IDX)
OUTER_LIPS_EDGES = reindex_connections(mp_face_mesh.FACEMESH_LIPS, FACE_SELECTED_IDX)

# Final edge list to visualize face structure
FACE_CONNECTIONS = (
    HEAD_CONTOUR_EDGES + LEFT_EYEBROW_EDGES + RIGHT_EYEBROW_EDGES +
    LEFT_EYE_EDGES + RIGHT_EYE_EDGES + OUTER_LIPS_EDGES
)

POSE_CONNECTIONS = [
            (0, 1), (1, 2),  # Right Shoulder → Right Elbow → Right Wrist
            (3, 4), (4, 5),  # Left Shoulder → Left Elbow → Left Wrist
            (0, 3),          # Right Shoulder → Left Shoulder
            (3, 7), (0, 6),  # Shoulder → Hips
            (6, 7)           # Hips
        ]

HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS

def translate_back_to_pixel(normalized_keypoints, fixed_shoulder_distance=86, image_shape={"width":210, "height":260}):

    neck_coords = np.array([image_shape["width"] / 2.0, image_shape["height"] / 2.0,  0.0])

    # Translate normalized keypoints back to pixel coordinates
    pixel_keypoints = (normalized_keypoints * fixed_shoulder_distance) + neck_coords

    return pixel_keypoints

def draw_keypoints_2(keypoints, frame=None, multip_factor=2, palet=normal_palet, is_face_included=False):
    """
    Draws keypoints and connections on a given frame.

    Parameters:
    - keypoints: numpy array containing keypoints in the order [pose, right_hand, left_hand, face].
    - frame: The image/frame where keypoints should be drawn.
    - multip_factor: Scaling factor for keypoints.
    - palet: Dictionary containing colors for pose, hands, and face.
    - is_face_included: Boolean to indicate if face landmarks should be drawn.

    Returns:
    - The updated frame with keypoints and connections drawn.
    """
    if frame is None:
        frame = np.zeros((260 * multip_factor, 210 * multip_factor, 3))  # Default blank image (3x enlarged for clarity)

    # Extract and scale keypoints
    pose_keypoints = (keypoints[:8, :] * multip_factor).astype("int32")  # Only 8 upper body keypoints
    right_hand_keypoints = (keypoints[8:29, :] * multip_factor).astype("int32")  # 21 right hand keypoints
    left_hand_keypoints = (keypoints[29:50, :] * multip_factor).astype("int32")  # 21 left hand keypoints
    if is_face_included:
        face_keypoints = (keypoints[50:, :] * multip_factor).astype("int32")  # 128 face keypoints

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        keypoint1 = tuple(pose_keypoints[start_idx][:2])
        keypoint2 = tuple(pose_keypoints[end_idx][:2])
        if not (np.all(keypoint1 == 0) and np.all(keypoint2 == 0)):
            cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=2)

    # Hand connections remain unchanged
    for connection in mp_holistic.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(left_hand_keypoints) and end_idx < len(left_hand_keypoints):
            keypoint1 = tuple(left_hand_keypoints[start_idx][:2])
            keypoint2 = tuple(left_hand_keypoints[end_idx][:2])
            if not (np.all(keypoint1 == 0) and np.all(keypoint2 == 0)):
                cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=2)

        if start_idx < len(right_hand_keypoints) and end_idx < len(right_hand_keypoints):
            keypoint1 = tuple(right_hand_keypoints[start_idx][:2])
            keypoint2 = tuple(right_hand_keypoints[end_idx][:2])
            if not (np.all(keypoint1 == 0) and np.all(keypoint2 == 0)):
                cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=2)
                
    if is_face_included:
        for connection in FACE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(face_keypoints) and end_idx < len(face_keypoints):
                keypoint1 = tuple(face_keypoints[start_idx][:2])
                keypoint2 = tuple(face_keypoints[end_idx][:2])
                if not (np.all(keypoint1 == 0) and np.all(keypoint2 == 0)):  # Skip missing keypoints
                    cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=1)

    # Face landmarks (if included)
    if is_face_included:
        for keypoint in face_keypoints:
            frame = cv2.circle(frame, tuple(keypoint[:2]), 1, palet["face"], 1)

    # Draw Individual Keypoints
    for keypoint in pose_keypoints:
        frame = cv2.circle(frame, tuple(keypoint[:2]), 2, palet["pose"], 2)

    if np.any(left_hand_keypoints):
        for keypoint in left_hand_keypoints:
            frame = cv2.circle(frame, tuple(keypoint[:2]), 2, palet["left_hand"], 2)

    if np.any(right_hand_keypoints):
        for keypoint in right_hand_keypoints:
            frame = cv2.circle(frame, tuple(keypoint[:2]), 2, palet["right_hand"], 2)

    return frame


def draw_keypoints(keypoints, frame, multip_factor=1, palet=normal_palet, is_face_included=False, only_hands=False, video_shape={"width": 210, "height": 260}):

  if frame is None:
      frame = np.zeros((video_shape["height"] *multip_factor , video_shape["width"]*multip_factor,3))
  else:
    frame = cv2.resize(frame, (0,0), fx=multip_factor, fy=multip_factor) 
  if is_face_included:
    pose_keypoints = keypoints[:23, :].astype("int32") * multip_factor
    face_keypoints = keypoints[23:491, :].astype("int32") * multip_factor
    left_hand_keypoints = keypoints[491:512, :].astype("int32") * multip_factor
    right_hand_keypoints = keypoints[512:533, :].astype("int32") * multip_factor
  else:
    pose_keypoints = keypoints[:23, :].astype("int32") * multip_factor
    left_hand_keypoints = keypoints[23:44, :].astype("int32") * multip_factor
    right_hand_keypoints = keypoints[44:65, :].astype("int32") * multip_factor

  # for connection in mp_holistic.FACEMESH_TESSELATION:
  #   start_idx = connection[0]
  #   end_idx = connection[1]
  #   if(not start_idx > face_keypoints.shape[0] and not end_idx > face_keypoints.shape[0]):
  #     keypoint1 = (int(face_keypoints[start_idx, 0]), int(face_keypoints[start_idx, 1]))
  #     keypoint2 = (int(face_keypoints[end_idx, 0]), int(face_keypoints[end_idx, 1]))
  #     cv2.line(frame, keypoint1,keypoint2, (173, 231, 146), thickness = 2)

  if only_hands:
      # Process only hand keypoints for visualization
      all_hand_keypoints = np.vstack((left_hand_keypoints, right_hand_keypoints))[:, :2]
      min_x, min_y = np.min(all_hand_keypoints, axis=0)
      max_x, max_y = np.max(all_hand_keypoints, axis=0)
      center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
      hand_width, hand_height = max_x - min_x, max_y - min_y

      # Set scaling multiplier to make hands larger
      scaling_multiplier = 1  # Adjust this value as needed
      scaling_factor = min((video_shape["width"] * 0.9) / hand_width, (video_shape["height"] * 0.9) / hand_height) * scaling_multiplier

      # Calculate shifts to center the hands in the frame
      shift_x = (video_shape["width"] / 2) - (center_x * scaling_factor)
      shift_y = (video_shape["height"] / 2) - (center_y * scaling_factor)

      # Apply scaling and centering to left and right hand keypoints
      left_hand_keypoints = (left_hand_keypoints[:, :2] - [center_x, center_y]) * scaling_factor + [video_shape["width"] / 2, video_shape["height"] / 2]
      right_hand_keypoints = (right_hand_keypoints[:, :2] - [center_x, center_y]) * scaling_factor + [video_shape["width"] / 2, video_shape["height"] / 2]

      # Draw hand connections and keypoints
      for connection in mp_holistic.HAND_CONNECTIONS:
          start_idx = connection[0]
          end_idx = connection[1]

          if start_idx < left_hand_keypoints.shape[0] and end_idx < left_hand_keypoints.shape[0]:
              keypoint1 = (int(left_hand_keypoints[start_idx, 0]), int(left_hand_keypoints[start_idx, 1]))
              keypoint2 = (int(left_hand_keypoints[end_idx, 0]), int(left_hand_keypoints[end_idx, 1]))
              cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=2)

          if start_idx < right_hand_keypoints.shape[0] and end_idx < right_hand_keypoints.shape[0]:
              keypoint1 = (int(right_hand_keypoints[start_idx, 0]), int(right_hand_keypoints[start_idx, 1]))
              keypoint2 = (int(right_hand_keypoints[end_idx, 0]), int(right_hand_keypoints[end_idx, 1]))
              cv2.line(frame, keypoint1, keypoint2, palet["line"], thickness=2)

      for keypoint in left_hand_keypoints:
          frame = cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, palet["left_hand"], -1)
      for keypoint in right_hand_keypoints:
          frame = cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, palet["right_hand"], -1)


  else:
    for connection in mp_holistic.POSE_CONNECTIONS:
      start_idx = connection[0]
      end_idx = connection[1]
      if((not start_idx >=  pose_keypoints.shape[0]) and (not end_idx >= pose_keypoints.shape[0])):
        keypoint1 = (int(pose_keypoints[start_idx, 0]), int(pose_keypoints[start_idx, 1]))
        keypoint2 = (int(pose_keypoints[end_idx, 0]), int(pose_keypoints[end_idx, 1]), )
        if not (keypoint1[0] == 0 and keypoint1[1] == 0 and keypoint2[0] == 0 and keypoint2[1] == 0):
          cv2.line(frame, keypoint1,keypoint2, palet["line"], thickness = 2)

    for connection in  mp_holistic.HAND_CONNECTIONS:
      start_idx = connection[0]
      end_idx = connection[1]
      if(not start_idx >=  left_hand_keypoints.shape[0] and not end_idx >=  left_hand_keypoints.shape[0]):
        keypoint1 = (int(left_hand_keypoints[start_idx, 0]), int(left_hand_keypoints[start_idx,1]))
        keypoint2 = (int(left_hand_keypoints[end_idx, 0]), int(left_hand_keypoints[end_idx, 1]))
        if not (keypoint1[0] == 0 and keypoint1[1] == 0 and keypoint2[0] == 0 and keypoint2[1] == 0):
          cv2.line(frame, keypoint1,keypoint2,  palet["line"], thickness = 2)
    for connection in mp_holistic.HAND_CONNECTIONS:
      start_idx = connection[0]
      end_idx = connection[1]
      if(not start_idx >=  right_hand_keypoints.shape[0] and not end_idx >=  right_hand_keypoints.shape[0]):
        keypoint1 = (int(right_hand_keypoints[start_idx, 0]), int(right_hand_keypoints[start_idx,1]))
        keypoint2 = (int(right_hand_keypoints[end_idx, 0]), int(right_hand_keypoints[end_idx, 1]))
        if not (keypoint1[0] == 0 and keypoint1[1] == 0 and keypoint2[0] == 0 and keypoint2[1] == 0):
          cv2.line(frame, keypoint1,keypoint2,  palet["line"], thickness = 2)

    if is_face_included:
      for keypoint in face_keypoints:
        frame = cv2.circle(frame, (keypoint[0], keypoint[1]) , 1,  palet["face"], 1)
    for keypoint in pose_keypoints:
      frame = cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])) , 1,  palet["pose"], 2)
    if(np.any(left_hand_keypoints)):
      for keypoint in left_hand_keypoints:
        frame = cv2.circle(frame, (keypoint[0], keypoint[1]) , 1,  blue_palet["left_hand"],2)
    if(np.any(right_hand_keypoints)):
      for keypoint in right_hand_keypoints:
        frame = cv2.circle(frame, (keypoint[0], keypoint[1]) , 1,  blue_palet["right_hand"], 2)

  return frame