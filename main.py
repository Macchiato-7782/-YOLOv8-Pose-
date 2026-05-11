import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Constants and Thresholds ---
# For YOLOv8 Pose Model
MODEL_PATH = 'yolov8n-pose.pt' # Or use yolov8m-pose.pt, yolov8l-pose.pt for potentially better accuracy but slower
MIN_CONF_DETECTION = 0.5 # Minimum confidence for a person detection by YOLO
MIN_CONF_KPT = 0.3       # Minimum confidence for a keypoint to be considered visible/reliable

# Keypoints for determining 'Full Body' visibility and calculating Bounding Box (YOLO indices)
# Indices: [0 (Nose), 5 (LShoulder), 6 (RShoulder), 11 (LHip), 12 (RHip), 15 (LAnkle), 16 (RAnkle)]
REQUIRED_KPTS_FOR_FULL_BODY = [0, 5, 6, 11, 12, 15, 16]
MIN_KPTS_FOR_FULL_BODY = 5 # Minimum number of required keypoints visible for 'Full Body' status and valid box/tracking

# Keypoints for Angle calculation (YOLO indices)
# We'll check Left Shoulder=5, Left Hip=11, Left Knee=13
ANGLE_KPTS_INDICES = {
    'shoulder': 5, # Left Shoulder
    'hip': 11,     # Left Hip (Vertex of the angle)
    'knee': 13     # Left Knee
}
MIN_CONF_FOR_ANGLE_KPTS = 0.3 # Minimum confidence for the specific angle keypoints

# Fall Detection Criteria Thresholds
HORIZONTAL_AR_THRESHOLD = 0.6     # Ratio of height / width for horizontal pose detection
ANGLE_THRESHOLD = 140              # Degrees for the hip angle (e.g., < 60 degrees is considered a collapsed torso)
MIN_FALL_POSE_DURATION = 1.0      # Seconds a potential fall pose (low AR or low Angle) must be held

# Tracking Parameters
TRACKING_DISTANCE_THRESHOLD = 150 # Max pixel distance for matching a person across frames
HISTORY_CLEANUP_INTERVAL = 5      # Seconds to keep track of a person after they disappear


# cap = cv2.VideoCapture("videoplayback.mp4")  # Video file
# cap = cv2.VideoCapture(0)  # Webcam (may pick iPhone via Continuity Camera)
cap = cv2.VideoCapture(1)  # Try built-in FaceTime camera


calculated_angles = []

# --- Helper Function ---
def calculate_angle(a, b, c):
    """Calculates the angle in degrees between three points (shoulder, hip, knee)."""
    a = np.array(a) # Shoulder
    b = np.array(b) # Hip (vertex)
    c = np.array(c) # Knee

    # Ensure points are distinct to avoid errors (though confidence check should largely prevent this)
    if np.array_equal(a, b) or np.array_equal(b, c):
         return 180.0 # Assume upright if points are the same

    # Calculate angles relative to the positive x-axis
    vec_ba = a - b
    vec_bc = c - b

    # Compute the angle using atan2 for robustness across quadrants
    radians = np.arctan2(vec_bc[1], vec_bc[0]) - np.arctan2(vec_ba[1], vec_ba[0])
    angle = np.abs(np.degrees(radians)) # Convert to degrees

    # Normalize angle to be between 0 and 180
    if angle > 180.0:
        angle = 360 - angle

    # print(f"Angle calculation: Shoulder={a}, Hip={b}, Knee={c}, Calculated Angle={angle:.2f}") # Detailed Debug
    return angle

# --- Load Model ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure you have 'ultralytics' installed (`pip install ultralytics`)")
    print(f"and the model file '{MODEL_PATH}' exists or is accessible.")
    exit()

# --- Person Tracking State ---
# Stores state for each tracked person
person_history = {}
next_person_id = 1
last_cleanup_time = time.time()

# --- Main Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break # End of video

    current_time = time.time()

    # Run YOLO inference
    # Setting verbose=False reduces console output
    # conf=MIN_CONF_DETECTION filters initial person detection confidence
    results = model(frame, conf=MIN_CONF_DETECTION, verbose=False)[0]

    # Keep track of which history IDs were matched by a full-body detection this frame
    matched_full_body_pids_this_frame = set()

    # Store minimal drawing info for persons detected as full-body this frame
    # This helps in drawing custom labels later over the results.plot() output
    drawing_info_this_frame = {} # {pid: {'box': (x1,y1,x2,y2), 'center': (cx,cy), 'status_color': (B,G,R), 'label_text': '...'}}


    # --- Phase 1: Process Raw Detections, Extract Info, Check Full Body, Update History for Full Body ---
    # results.keypoints gives a list of Keypoints objects, one for each person detection
    for i, pose in enumerate(results.keypoints):

        # Check for missing keypoint data
        if pose.conf is None or len(pose.conf) == 0 or pose.xy is None or len(pose.xy) == 0 or len(pose.xy[0]) < 17:
             # print(f"Skipping detection {i}: Incomplete keypoint data.") # Debug
             continue # Skip this pose if keypoint data is missing or malformed

        keypoints = pose.xy[0].cpu().numpy() # Shape (17, 2)
        confs = pose.conf[0].cpu().numpy() # Shape (17,)

        # --- Check for 'Full Body' visibility & Calculate Bounding Box from Keypoints ---
        visible_relevant_kpts_coords = []
        for idx in REQUIRED_KPTS_FOR_FULL_BODY:
            # Check if index is valid for the keypoints array, keypoint is visible (>0) and confident
            if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_KPT:
                visible_relevant_kpts_coords.append(keypoints[idx])

        # Determine if 'Full Body' is visible
        is_full_body_visible_this_detection = (len(visible_relevant_kpts_coords) >= MIN_KPTS_FOR_FULL_BODY)

        # Calculate box coordinates from visible relevant keypoints (Needed for AR, Center, and Label position)
        x1, y1, x2, y2, w, h, center = None, None, None, None, None, None, None
        aspect_ratio = None

        if len(visible_relevant_kpts_coords) >= 2: # Need at least 2 points to define a box min/max
             try:
                 visible_relevant_kpts_coords = np.array(visible_relevant_kpts_coords)
                 x1_kpt, y1_kpt = visible_relevant_kpts_coords.min(axis=0)
                 x2_kpt, y2_kpt = visible_relevant_kpts_coords.max(axis=0)
                 x1, y1, x2, y2 = map(int, [x1_kpt, y1_kpt, x2_kpt, y2_kpt])
                 w, h = x2 - x1, y2 - y1
                 if w > 0 and h > 0:
                     center = ((x1 + x2) // 2, (y1 + y2) // 2)
                     aspect_ratio = h / w
                 else:
                     # Invalid box dimensions
                     x1, y1, x2, y2 = None, None, None, None # Invalidate box if dimensions are non-positive
             except ValueError:
                 # print(f"Detection {i}: Could not calculate min/max for bounding box.") # Debug
                 pass # Ignore if min/max calculation fails


        # If we can't get a valid center or box, this detection isn't usable for tracking/logic
        if center is None or x1 is None:
            # print(f"Skipping detection {i}: Could not derive valid box/center.") # Debug
            continue


        # --- Match with Person History & Update State (Only if Full Body) ---
        matched_id = None

        # Find matching person in history based on the calculated center
        for pid, data in list(person_history.items()): # Iterate copy in case we add new ID
             dist = np.sqrt((center[0] - data['center'][0])**2 + (center[1] - data['center'][1])**2)
             # Check distance AND that this history ID hasn't been matched by another (preferably full-body) detection this frame
             # Simple check: if pid already matched by a full-body, skip subsequent matches for it in this frame
             if dist < TRACKING_DISTANCE_THRESHOLD and pid not in matched_full_body_pids_this_frame:
                 matched_id = pid
                 break

        if matched_id is None:
             # This detection didn't match any existing track
             # ONLY create a new track if it's a full body detection
             if is_full_body_visible_this_detection:
                  matched_id = next_person_id
                  next_person_id += 1
                  person_history[matched_id] = {
                      'center': center,
                      'is_potential_fall_pose': False,
                      'fall_pose_start_time': None,
                      'fall_detected': False,
                      'last_seen': current_time,
                      'current_frame_box': (x1, y1, x2, y2), # Store box for drawing this frame
                      'is_full_body_this_frame': True # Mark that this update came from a full body detection
                  }
                  matched_full_body_pids_this_frame.add(matched_id)
                  # print(f"New person (Full Body): ID {matched_id}") # Debug)

        # If a match was found (either existing or new full body)
        if matched_id is not None:
             person_data = person_history[matched_id]

             # Always update location and last seen time for matched tracks
             person_data['center'] = center
             person_data['last_seen'] = current_time
             person_data['current_frame_box'] = (x1, y1, x2, y2) # Store box for drawing this frame
             person_data['is_full_body_this_frame'] = is_full_body_visible_this_detection # Mark the type of detection

             # ONLY update fall state if the CURRENT detection is Full Body
             if is_full_body_visible_this_detection:
                  matched_full_body_pids_this_frame.add(matched_id) # Ensure this PID is marked as having a full body match

                  # Calculate Angle (Left Shoulder-Left Hip-Left Knee) if available
                  angle = None # Reset angle check for this full-body update
                  angle_kpts_coords = {}
                  angle_kpts_available = True
                  for name, idx in ANGLE_KPTS_INDICES.items():
                       # Use keypoints from the current detection
                       if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0 and confs[idx] > MIN_CONF_FOR_ANGLE_KPTS:
                           angle_kpts_coords[name] = (keypoints[idx][0], keypoints[idx][1])
                       else:
                           angle_kpts_available = False
                           break # Need all three points for the angle

                  if angle_kpts_available:
                       try:
                            angle = calculate_angle(angle_kpts_coords['shoulder'], angle_kpts_coords['hip'], angle_kpts_coords['knee'])
                            # User requested angle logging - keep for debugging
                            calculated_angles.append(angle)
                       except Exception as e:
                            # print(f"Person {matched_id}: Error calculating angle: {e}") # Debug
                            angle = None

                  # Determine if currently in a "Potential Fall Pose" based on Full Body AR/Angle
                  is_potential_fall_pose_this_frame = False
                  if aspect_ratio is not None and aspect_ratio < HORIZONTAL_AR_THRESHOLD:
                       is_potential_fall_pose_this_frame = True
                       # print(f"Person {matched_id}: Full Body -> Low AR ({aspect_ratio:.2f})") # Debug)

                  if angle is not None and angle < ANGLE_THRESHOLD:
                       is_potential_fall_pose_this_frame = True
                       # print(f"Person {matched_id}: Full Body -> Low Angle ({angle:.2f})") # Debug)


                  # Update State and Timer based on Potential Fall Pose (only if the source was full body)
                  if is_potential_fall_pose_this_frame:
                       if not person_data['is_potential_fall_pose']:
                           # Transition from normal to potential fall pose
                           person_data['is_potential_fall_pose'] = True
                           person_data['fall_pose_start_time'] = current_time
                           # print(f"Person {matched_id}: Started potential fall pose at {current_time:.2f}") # Debug)
                  else:
                       # Not in a potential fall pose this frame (for a full body detection)
                       if person_data['is_potential_fall_pose']:
                           # Transition from potential fall pose to normal state (clears timer)
                           person_data['is_potential_fall_pose'] = False
                           person_data['fall_pose_start_time'] = None
                           # print(f"Person {matched_id}: Exited potential fall pose") # Debug)

                  # Check for Fall Detection based on Duration (only if currently in potential pose state)
                  fall_detected_for_person = False
                  if person_data['is_potential_fall_pose'] and person_data['fall_pose_start_time'] is not None:
                        duration = current_time - person_data['fall_pose_start_time']
                        # print(f"Person {matched_id}: Potential fall duration: {duration:.2f}s") # Debug)
                        if duration >= MIN_FALL_POSE_DURATION:
                            fall_detected_for_person = True
                            # print(f"Person {matched_id}: >>> FALL DETECTED! <<<") # Debug)

                  person_data['fall_detected'] = fall_detected_for_person # Update the fall status for this person

             # Prepare drawing info for this matched person (regardless of full body status this frame)
             color = (0, 0, 255) if person_data['fall_detected'] else (0, 255, 0) # Red/Green
             if person_data['is_potential_fall_pose'] and not person_data['fall_detected']:
                  color = (0, 165, 255) # Orange/Yellow for "Potential Fall Pose" state

             label = f"ID {matched_id}: {'Fall' if person_data['fall_detected'] else 'Normal'}"
             # Add duration for potential/confirmed fall states
             if person_data['is_potential_fall_pose'] and person_data['fall_pose_start_time'] is not None:
                  duration_str = f"({current_time - person_data['fall_pose_start_time']:.1f}s)"
                  label += " " + duration_str
             # Optional debug info
             # if aspect_ratio is not None: label += f" AR={aspect_ratio:.1f}"
             # if angle is not None: label += f" Angle={angle:.0f}"
             # label += f" FB={str(person_data.get('is_full_body_this_frame', False))[0]}" # Add FB status char

             drawing_info_this_frame[matched_id] = {
                  'box': person_data['current_frame_box'], # Use the box from this frame's detection
                  'status_color': color,
                  'label_text': label
             }
        # else: # Detection was not Full Body and didn't match an existing track - ignored for history

    # --- Phase 2: History Cleanup ---
    # Remove persons from history who haven't been seen (full body or not) for a while
    if current_time - last_cleanup_time > HISTORY_CLEANUP_INTERVAL:
        ids_to_remove = [
            pid for pid, data in person_history.items()
            if current_time - data['last_seen'] > HISTORY_CLEANUP_INTERVAL
        ]
        for pid in ids_to_remove:
            # print(f"Removing person {pid} from history (not seen for > {HISTORY_CLEANUP_INTERVAL}s)") # Debug)
            del person_history[pid]
        last_cleanup_time = current_time

    # --- Phase 3: Drawing Results ---

    # Use YOLO's built-in plot function to draw all detections and keypoints
    # This includes persons detected as full-body and those not.
    plotted_frame = results.plot() # This returns a new frame with drawings

    # Draw custom labels for tracked persons who were matched this frame
    # (using info prepared in Phase 1)
    for pid, info in drawing_info_this_frame.items():
        box = info['box']
        if box is not None: # Ensure we have a valid box from this frame's detection for position
            x1, y1, x2, y2 = box
            color = info['status_color']
            label = info['label_text']

            # Draw the label text
            text_y_pos = max(y1 - 10, 20) # Put text 10px above box, but at least 20px from top
            # Ensure text doesn't go off the left edge either
            text_x_pos = max(x1, 5)
            cv2.putText(plotted_frame, label, (text_x_pos, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # --- Display Frame ---
    cv2.imshow("Fall Detection (Combined AR + Angle + Duration + Full Body Filter)", plotted_frame)


    # --- Exit Condition ---
    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC key pressed. Exiting.")
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()

# Print mean of angles if any were calculated (as requested by user)
print("Mean of calculated angles:", np.mean(calculated_angles) if calculated_angles else "No angles calculated")
