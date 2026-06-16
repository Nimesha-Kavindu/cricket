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
import win32gui
import win32con
import win32api
import win32process
import pydirectinput
import ctypes

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ──────────────────────────────────────────────
#  SETTINGS  (tweak these for your comfort)
# ──────────────────────────────────────────────
WEBCAM_INDEX          = 0       # 0 = default webcam
SWING_VELOCITY_THRESH = 0.10    # Speed needed to trigger a swing (raise = less sensitive)
SWING_ARC_THRESH      = 35      # Min elbow angle change in degrees
COOLDOWN_SECONDS      = 3.0     # Seconds to wait before next shot
HISTORY_FRAMES        = 6       # Frames tracked for velocity
WINDOW_SCALE          = 0.5     # Display window size (0.5 = half, 0.4 = smaller, 1.0 = full)
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
#  GAME WINDOW CLICK HELPER
# ──────────────────────────────────────────────

# Keywords to search for in the game window title
GAME_WINDOW_KEYWORDS = ["city cricket", "cricket", "city"]

_game_hwnd = None   # cached window handle


def list_all_windows():
    """Print all visible window titles — use this to find City Cricket's title."""
    print("\n[DEBUG] Visible windows:")
    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                print(f"  > {title}")
    win32gui.EnumWindows(_enum, None)
    print()


def find_game_window():
    """Find the City Cricket game window handle by title keyword."""
    found = []
    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if any(kw in title for kw in GAME_WINDOW_KEYWORDS):
                found.append(hwnd)
    win32gui.EnumWindows(_enum, None)
    return found[0] if found else None


def fire_spacebar():
    """
    Send SPACEBAR to City Cricket using ALL available methods:

    1. Focus the game window (ALT trick + SetForegroundWindow via ctypes)
    2. PostMessage WM_KEYDOWN  → works for WM-based / old Windows games
    3. pydirectinput.press()   → uses SendInput, works for DirectInput games
                                 (requires game to be in focus — step 1 handles this)

    Using all three methods ensures the key reaches the game regardless of
    whether it uses Win32 messages, DirectInput, or GetAsyncKeyState.
    """
    global _game_hwnd

    # Refresh handle if stale
    if not _game_hwnd or not win32gui.IsWindow(_game_hwnd):
        _game_hwnd = find_game_window()

    if _game_hwnd:
        # ── Step 1: Focus the game window
        try:
            win32gui.ShowWindow(_game_hwnd, win32con.SW_RESTORE)
        except Exception:
            pass

        try:
            # ALT key trick: briefly press ALT so Windows allows focus steal
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
            # Use raw ctypes call — avoids Python exception on error
            ctypes.windll.user32.AllowSetForegroundWindow(-1)
            ctypes.windll.user32.SetForegroundWindow(_game_hwnd)
        except Exception:
            pass

        time.sleep(0.06)   # let the game receive focus

        # ── Step 2: PostMessage (for WM-based / old Windows games)
        try:
            win32api.PostMessage(_game_hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
            time.sleep(0.05)
            win32api.PostMessage(_game_hwnd, win32con.WM_KEYUP,   win32con.VK_SPACE, 0)
        except Exception as e:
            print(f"[WARN] PostMessage failed: {e}")

    else:
        print("[WARN] Game window not found! Open City Cricket, then press W to list windows.")

    # ── Step 3: pydirectinput SendInput (for DirectInput / GetAsyncKeyState games)
    # This fires to whatever window is currently in focus (hopefully the game)
    try:
        pydirectinput.press('space')
    except Exception as e:
        print(f"[WARN] pydirectinput failed: {e}")

    return _game_hwnd is not None

# ──────────────────────────────────────────────
#  SWING DETECTOR
# ──────────────────────────────────────────────
#
#  Fires the spacebar IMMEDIATELY when the swing is detected
#  (as soon as wrist speed + arc cross the thresholds).
#  Then locks out for COOLDOWN_SECONDS before it can fire again.
#  This gives ZERO lag — the hit registers the moment you swing.

class SwingDetector:
    STATE_IDLE     = "IDLE"
    STATE_SWINGING = "SWINGING"

    def __init__(self):
        self.wrist_history   = deque(maxlen=HISTORY_FRAMES)
        self.angle_history   = deque(maxlen=HISTORY_FRAMES)
        self.last_click_time = 0
        self.swing_count     = 0
        self.swing_flash     = 0
        self.state           = self.STATE_IDLE

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

        if self.state == self.STATE_IDLE:
            # ── Fire IMMEDIATELY when swing threshold crossed
            if (velocity    > SWING_VELOCITY_THRESH and
                angle_delta > SWING_ARC_THRESH      and
                cooldown_left == 0):
                swing_detected       = True
                self.last_click_time = now
                self.swing_count    += 1
                self.swing_flash     = 20
                self.state           = self.STATE_SWINGING
                print(f"[SHOT #{self.swing_count}]  Speed={velocity:.4f}  "
                      f"Arc={angle_delta:.1f}deg")

        elif self.state == self.STATE_SWINGING:
            # ── Stay in SWINGING until arm slows back to near rest
            # This prevents re-triggering during the follow-through
            if velocity < SWING_VELOCITY_THRESH * 0.4:
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

    # ── Create a resizable display window
    WIN_NAME = "City Cricket - Gesture Controller"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

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
                success = fire_spacebar()
                if not success:
                    print("[WARN] City Cricket window not found! Press W to list all open windows.")


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
                        "Cricket Controller  |  Q=Quit  S=Debug  C=Test Click  W=List Windows"
                        f"  |  Threshold: {SWING_VELOCITY_THRESH}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            # ── Resize frame for display (does not affect tracking accuracy)
            if WINDOW_SCALE != 1.0:
                dw = int(w * WINDOW_SCALE)
                dh = int(h * WINDOW_SCALE)
                display_frame = cv2.resize(frame, (dw, dh))
            else:
                display_frame = frame

            cv2.imshow(WIN_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("\n[EXIT] Controller stopped.")
                break
            elif key == ord('s'):
                show_debug = not show_debug
            elif key == ord('c'):
                success = fire_spacebar()
                status_msg = "OK (spacebar sent)" if success else "FALLBACK (game not found)"
                print(f"[TEST] Manual spacebar fired  [{status_msg}]")
            elif key == ord('w'):
                list_all_windows()
                print("       --> Add the game title keyword to GAME_WINDOW_KEYWORDS in the script")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Total shots fired: {detector.swing_count}")
    print("  Thanks for playing! [Cricket]")


if __name__ == "__main__":
    main()
