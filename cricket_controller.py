"""
=============================================================
  CITY CRICKET - Gesture Controller
  Uses webcam + MediaPipe to detect batting swing
  and fire a right-click to hit the ball in game.
=============================================================
  Tracks: Shoulders, Elbows, Wrists, Hands (HolisticLandmarker)
  Swing detected → right-click fired automatically

  Compatible with: mediapipe >= 0.10.x (Tasks API)
=============================================================
"""

import cv2
import numpy as np
import pyautogui
import time
import os
import urllib.request
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ──────────────────────────────────────────────
#  SETTINGS  (tweak these for your comfort)
# ──────────────────────────────────────────────
WEBCAM_INDEX          = 0       # 0 = default webcam
SWING_VELOCITY_THRESH = 0.10    # Speed needed to START a swing (raise to reduce sensitivity)
SWING_END_THRESH      = 0.03    # Speed that marks swing END (when arm slows to a stop)
SWING_ARC_THRESH      = 35      # Min elbow angle change during swing
MIN_SWING_DURATION    = 0.08    # Swing must last at least this many seconds (avoid jitter)
MAX_SWING_DURATION    = 2.5     # If swing goes on this long, reset without firing
COOLDOWN_SECONDS      = 1.5     # Seconds between allowed clicks
HISTORY_FRAMES        = 6       # Frames tracked for velocity
USE_RIGHT_HAND        = True    # True = right arm, False = left arm
SHOW_DEBUG_INFO       = True    # Show debug panel on screen
CLICK_AT_CENTER       = True    # Click screen center (where ball is in City Cricket)
MODEL_FILE            = "holistic_landmarker.task"
MODEL_URL             = ("https://storage.googleapis.com/mediapipe-models/"
                         "holistic_landmarker/holistic_landmarker/float16/"
                         "latest/holistic_landmarker.task")

# MediaPipe Holistic Pose landmark indices
# Right side: shoulder=12, elbow=14, wrist=16
# Left  side: shoulder=11, elbow=13, wrist=15
if USE_RIGHT_HAND:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 12, 14, 16
else:
    SHOULDER_IDX, ELBOW_IDX, WRIST_IDX = 11, 13, 15

# ──────────────────────────────────────────────
#  MODEL DOWNLOAD
# ──────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_FILE):
        print(f"[INFO] Downloading holistic model (~{4}MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[INFO] Model downloaded!")
    else:
        print("[INFO] Model file found.")

# ──────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────

def get_angle(a, b, c):
    """Angle at point B formed by A-B-C (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def lm_to_xy(lm, w, h):
    """Normalised landmark → pixel (x, y)."""
    return int(lm.x * w), int(lm.y * h)


def draw_status_box(frame, text, color, position=(20, 20)):
    x, y = position
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x-8, y-th-8), (x+tw+8, y+8), (0, 0, 0), -1)
    cv2.rectangle(frame, (x-8, y-th-8), (x+tw+8, y+8), color, 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_info_panel(frame, velocity, elbow_angle, angle_delta, cooldown_left, swing_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    px = w - 275
    cv2.rectangle(overlay, (px-10, 10), (w-10, 225), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "[ CRICKET CONTROLLER ]", (px-5, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1, cv2.LINE_AA)
    lines = [
        (f"  Wrist Speed : {velocity:.4f}",
         (0, 230, 120) if velocity > SWING_VELOCITY_THRESH else (180,180,180)),
        (f"  Elbow Angle : {elbow_angle:.1f} deg", (180, 180, 180)),
        (f"  Angle Delta : {angle_delta:.1f}",
         (0, 200, 255) if angle_delta > SWING_ARC_THRESH else (180,180,180)),
        (f"  Cooldown    : {cooldown_left:.1f}s",
         (100,100,255) if cooldown_left > 0 else (180,180,180)),
        (f"  Shots fired : {swing_count}", (255, 200, 0)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (px, 55 + i*36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
#  SWING DETECTOR  (State Machine)
# ──────────────────────────────────────────────
#
#  States:
#    IDLE     → arm at rest, waiting for a swing to start
#    SWINGING → wrist is moving fast, swing is in progress
#
#  A click fires only when:
#    1. Wrist speeds up past SWING_VELOCITY_THRESH  (enters SWINGING)
#    2. Wrist slows back down past SWING_END_THRESH (swing is complete)
#  This means ONE complete motion = ONE click.

class SwingDetector:
    STATE_IDLE     = "IDLE"
    STATE_SWINGING = "SWINGING"

    def __init__(self):
        self.wrist_history    = deque(maxlen=HISTORY_FRAMES)
        self.angle_history    = deque(maxlen=HISTORY_FRAMES)
        self.last_click_time  = 0
        self.swing_count      = 0
        self.swing_flash      = 0
        self.state            = self.STATE_IDLE
        self.swing_start_time = 0
        self.peak_velocity    = 0
        self.peak_arc         = 0

    @property
    def state_label(self):
        return self.state

    def update(self, shoulder_n, elbow_n, wrist_n):
        """Returns (swing_detected, velocity, elbow_angle, angle_delta)."""
        now = time.time()
        self.wrist_history.append(np.array(wrist_n))
        self.angle_history.append(get_angle(shoulder_n, elbow_n, wrist_n))

        if len(self.wrist_history) < HISTORY_FRAMES:
            return False, 0.0, 0.0, 0.0

        velocity      = np.linalg.norm(self.wrist_history[-1] - self.wrist_history[0])
        angle_delta   = abs(self.angle_history[-1] - self.angle_history[0])
        elbow_angle   = self.angle_history[-1]
        cooldown_left = max(0, COOLDOWN_SECONDS - (now - self.last_click_time))
        swing_detected = False

        # ── STATE: IDLE → waiting for a swing to begin
        if self.state == self.STATE_IDLE:
            if (velocity    > SWING_VELOCITY_THRESH and
                angle_delta > SWING_ARC_THRESH      and
                cooldown_left == 0):
                self.state            = self.STATE_SWINGING
                self.swing_start_time = now
                self.peak_velocity    = velocity
                self.peak_arc         = angle_delta

        # ── STATE: SWINGING → waiting for swing to finish
        elif self.state == self.STATE_SWINGING:
            swing_duration = now - self.swing_start_time
            self.peak_velocity = max(self.peak_velocity, velocity)
            self.peak_arc      = max(self.peak_arc, angle_delta)

            # Swing COMPLETE: arm has slowed back down after MIN_SWING_DURATION
            if velocity < SWING_END_THRESH and swing_duration > MIN_SWING_DURATION:
                swing_detected       = True
                self.last_click_time = now
                self.swing_count    += 1
                self.swing_flash     = 20
                self.state           = self.STATE_IDLE
                print(f"[SHOT #{self.swing_count}]  Peak Speed={self.peak_velocity:.4f}  "
                      f"Peak Arc={self.peak_arc:.1f}deg  Duration={swing_duration:.2f}s")

            # Safety reset: swing went on too long (false positive)
            elif swing_duration > MAX_SWING_DURATION:
                self.state = self.STATE_IDLE

        if self.swing_flash > 0:
            self.swing_flash -= 1

        return swing_detected, velocity, elbow_angle, angle_delta


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    ensure_model()

    print("=" * 55)
    print("  CITY CRICKET GESTURE CONTROLLER")
    print("=" * 55)
    print(f"  Swing speed threshold : {SWING_VELOCITY_THRESH}")
    print(f"  Arm arc threshold     : {SWING_ARC_THRESH} degrees")
    print(f"  Cooldown              : {COOLDOWN_SECONDS}s")
    print(f"  Hand tracked          : {'RIGHT' if USE_RIGHT_HAND else 'LEFT'}")
    print("=" * 55)
    print("  Press  Q  to quit")
    print("  Press  S  to toggle debug panel")
    print("  Press  C  to manually test click")
    print("=" * 55)

    # ── Build HolisticLandmarker (live stream mode)
    latest_result = {"result": None, "timestamp": 0}

    def result_callback(result, output_image, timestamp_ms):
        latest_result["result"]    = result
        latest_result["timestamp"] = timestamp_ms

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=result_callback,
        min_pose_detection_confidence=0.6,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.6,
        min_hand_landmarks_confidence=0.6,
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screen_w, screen_h = pyautogui.size()
    click_x = screen_w  // 2
    click_y = screen_h  // 2
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0.0

    detector   = SwingDetector()
    show_debug = SHOW_DEBUG_INFO
    frame_idx  = 0

    with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read from webcam.")
                break

            frame     = cv2.flip(frame, 1)   # mirror
            h, w      = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Send frame to MediaPipe (async)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, frame_idx)
            frame_idx += 1

            result = latest_result["result"]

            velocity    = 0.0
            elbow_angle = 0.0
            angle_delta = 0.0
            swing_fired = False
            cooldown_left = max(0, COOLDOWN_SECONDS - (time.time() - detector.last_click_time))

            if result and result.pose_landmarks:
                # pose_landmarks is a flat List[NormalizedLandmark] in HolisticLandmarkerResult
                pose_lms = result.pose_landmarks

                if len(pose_lms) > max(SHOULDER_IDX, ELBOW_IDX, WRIST_IDX):
                    sn = [pose_lms[SHOULDER_IDX].x, pose_lms[SHOULDER_IDX].y]
                    en = [pose_lms[ELBOW_IDX].x,    pose_lms[ELBOW_IDX].y]
                    wn = [pose_lms[WRIST_IDX].x,    pose_lms[WRIST_IDX].y]

                    swing_fired, velocity, elbow_angle, angle_delta = detector.update(sn, en, wn)

                    # ── Draw arm joints
                    sx, sy = lm_to_xy(pose_lms[SHOULDER_IDX], w, h)
                    ex, ey = lm_to_xy(pose_lms[ELBOW_IDX],    w, h)
                    wx, wy = lm_to_xy(pose_lms[WRIST_IDX],    w, h)

                    swing_color = (0, 80, 255) if detector.swing_flash > 0 else (0, 255, 120)
                    cv2.line(frame, (sx, sy), (ex, ey), (200, 200, 200), 3)
                    cv2.line(frame, (ex, ey), (wx, wy), (200, 200, 200), 3)
                    cv2.circle(frame, (sx, sy), 13, (255, 180,  0),  -1)
                    cv2.circle(frame, (ex, ey), 13, (255,  80,200),  -1)
                    cv2.circle(frame, (wx, wy), 15, swing_color,     -1)
                    cv2.putText(frame, "S", (sx-5, sy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(frame, "E", (ex-5, ey+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(frame, "W", (wx-5, wy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                # ── Draw skeleton lines
                connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
                for conn in connections:
                    s, e = conn.start, conn.end
                    if s < len(pose_lms) and e < len(pose_lms):
                        x1, y1 = lm_to_xy(pose_lms[s], w, h)
                        x2, y2 = lm_to_xy(pose_lms[e], w, h)
                        cv2.line(frame, (x1,y1), (x2,y2), (80, 80, 80), 1)

            # ── Draw hand landmarks (right or left depending on setting)
            hand_lms_list = None
            if result:
                hand_lms_list = result.right_hand_landmarks if USE_RIGHT_HAND else result.left_hand_landmarks
            if hand_lms_list:
                conns = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
                for conn in conns:
                    if conn.start < len(hand_lms_list) and conn.end < len(hand_lms_list):
                        x1, y1 = lm_to_xy(hand_lms_list[conn.start], w, h)
                        x2, y2 = lm_to_xy(hand_lms_list[conn.end],   w, h)
                        cv2.line(frame, (x1,y1), (x2,y2), (0, 180, 255), 2)
                for lm in hand_lms_list:
                    cv2.circle(frame, lm_to_xy(lm, w, h), 4, (0, 255, 200), -1)

            # ── Fire click if swing detected
            if swing_fired:
                if CLICK_AT_CENTER:
                    pyautogui.click(click_x, click_y, button='right')
                else:
                    pyautogui.click(button='right')


            # ── Flash overlay on swing
            if detector.swing_flash > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0, 60, 200), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                draw_status_box(frame,
                                f"  SHOT! #{detector.swing_count}  ",
                                (0, 120, 255),
                                position=(w//2 - 120, h//2))

            # ── Status label: shows current detector state
            if result and result.pose_landmarks:
                if cooldown_left > 0:
                    status = f"WAIT {cooldown_left:.1f}s"
                    color  = (100, 100, 255)   # blue
                elif detector.state == SwingDetector.STATE_SWINGING:
                    status = "SWINGING..."
                    color  = (0, 200, 255)     # yellow
                else:
                    status = "READY"
                    color  = (0, 255, 100)     # green
                draw_status_box(frame, f" {status} ", color, position=(20, h-55))

            # ── Debug panel
            if show_debug:
                draw_info_panel(frame, velocity, elbow_angle, angle_delta,
                                cooldown_left, detector.swing_count)

            # ── Top bar
            cv2.rectangle(frame, (0,0), (w, 38), (15,15,15), -1)
            cv2.putText(frame,
                        "City Cricket Controller  |  Q=Quit  S=Debug  C=Test Click"
                        f"  |  Threshold: {SWING_VELOCITY_THRESH}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.imshow("City Cricket - Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n[EXIT] Controller stopped.")
                break
            elif key == ord('s'):
                show_debug = not show_debug
            elif key == ord('c'):
                pyautogui.click(click_x, click_y, button='right')
                print(f"[TEST] Manual right-click at ({click_x}, {click_y})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Total shots fired: {detector.swing_count}")
    print("  Thanks for playing! [Cricket]")


if __name__ == "__main__":
    main()
