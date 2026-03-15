import cv2
import mediapipe as mp
import numpy as np
import time
import simpleaudio as sa
import threading
import queue
import tkinter as tk
from tkinter import font as tkfont

# ================== 配置参数 ==================
MISSING_FACE_THRESHOLD = 5.0           # 无人脸超过5秒触发离开警报（短暂低头/转头不会触发）
SIT_LOST_TIMEOUT = 30                  # 人脸丢失后仍视为坐着的最大缓冲时间（秒），超过则重置久坐计时
YAW_THRESHOLD = 20                      # 左右转头阈值（度，相对于基准）
PITCH_THRESHOLD = 25                    # 低头/抬头阈值
ROLL_THRESHOLD = 15                      # 头部倾斜阈值（用于姿态角度）
POSE_OFF_DURATION = 0.8                  # 姿态异常持续0.8秒触发

VERTICAL_AXIS_THRESHOLD = 3.9            # 中轴倾角偏差阈值（度）
HORIZONTAL_AXIS_THRESHOLD = 3.6          # 水平线倾角偏差阈值
VERTICAL_OFFSET_THRESHOLD = 0.96          # 眼睛中心偏离画面中心超过15%时触发

REPEAT_ALERT_INTERVAL = 6.0               # 坐姿不良重复提醒间隔（秒）

# 久坐提醒参数
SIT_DURATION = 2700                       # 久坐时间（秒），默认45分钟
BREAK_DURATION = 300                       # 休息时间（秒），默认5分钟

# 关键点索引（MediaPipe Face Mesh）
IDX_NOSE = 1
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263
IDX_NOSE_BRIDGE = 168                      # 鼻梁中央
IDX_CHIN = 152
# 遮挡检测点：鼻尖、下巴、上下嘴唇、左右嘴角、左右鼻孔
OCCLUSION_POINTS = [1, 152, 13, 14, 61, 291, 2, 19]

# ================== 初始化 MediaPipe ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================== 音频蜂鸣 ==================
def beep():
    """播放一个短促的蜂鸣声（约140ms）"""
    try:
        sample_rate = 44100
        t = np.linspace(0, 0.14, int(sample_rate * 0.14))
        wave = (0.1 * np.sin(880 * 2 * np.pi * t)).astype(np.float32)
        wave = (wave * 32767).astype(np.int16)
        play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
        play_obj.wait_done()
    except Exception as e:
        print(f"音频播放失败: {e}")

# ================== 弹窗管理器（独立Tkinter线程） ==================
class BreakWindowManager:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.root = None
        self.window = None
        self.time_var = None

    def _run(self):
        root = tk.Tk()
        root.withdraw()
        self.root = root

        def process_queue():
            try:
                while True:
                    cmd, args = self.q.get_nowait()
                    if cmd == 'show':
                        self._show_window(args['duration'])
                    elif cmd == 'close':
                        if self.window:
                            self.window.destroy()
                            self.window = None
            except queue.Empty:
                pass
            root.after(100, process_queue)

        root.after(100, process_queue)
        root.mainloop()

    def _show_window(self, duration):
        if self.window:
            return
        window = tk.Toplevel(self.root)
        window.title("久坐提醒")
        window.attributes('-fullscreen', True)
        window.attributes('-alpha', 0.8)
        window.attributes('-topmost', True)
        window.configure(bg='black')
        window.overrideredirect(True)

        large_font = tkfont.Font(size=48, weight='bold')
        label_msg = tk.Label(window, text="久坐提醒！请起身活动", fg='red', bg='black', font=large_font)
        label_msg.pack(expand=True)

        time_var = tk.StringVar()
        time_var.set(f"{duration//60}:{duration%60:02d} 分钟后自动关闭")
        label_time = tk.Label(window, textvariable=time_var, fg='yellow', bg='black', font=('Arial', 36))
        label_time.pack(expand=True)

        self.window = window
        self.time_var = time_var

        def countdown(remaining):
            if remaining <= 0:
                window.destroy()
                self.window = None
                return
            mins, secs = divmod(remaining, 60)
            time_var.set(f"{mins}:{secs:02d} 分钟后自动关闭")
            window.after(1000, countdown, remaining - 1)

        window.after(1000, countdown, duration - 1)

    def show(self, duration):
        self.q.put(('show', {'duration': duration}))

    def close(self):
        self.q.put(('close', {}))

# ================== 全局状态 ==================
class State:
    def __init__(self):
        self.last_face_seen = None
        self.pose_off_start = None
        self.last_alert_times = {
            'axis': 0,
            'occlusion': 0,
            'missing': 0,
            'pose': 0,
            'vertical_offset': 0
        }
        self.baseline = None
        self.last_pose = None
        self.occlusion_active = False
        self.any_alert_active = False

        # 久坐相关（带缓冲）
        self.sit_accumulated = 0.0        # 累计坐姿时间（秒）
        self.sit_lost_start = None         # 人脸丢失开始时间
        self.last_break_alert_time = 0

state = State()

# ================== 姿态估计 ==================
def estimate_head_pose(landmarks, img_w, img_h):
    nose = landmarks[IDX_NOSE]
    left_eye = landmarks[IDX_LEFT_EYE_OUTER]
    right_eye = landmarks[IDX_RIGHT_EYE_OUTER]

    xN = nose.x * img_w
    yN = nose.y * img_h
    xL = left_eye.x * img_w
    yL = left_eye.y * img_h
    xR = right_eye.x * img_w
    yR = right_eye.y * img_h

    vx = xR - xL
    vy = yR - yL
    inter_eye = np.hypot(vx, vy) + 1e-6

    eye_mid_x = (xL + xR) / 2
    eye_mid_y = (yL + yR) / 2

    dx = xN - eye_mid_x
    dy = yN - eye_mid_y

    yaw = np.arctan2(dx, inter_eye) * 180 / np.pi
    pitch = np.arctan2(dy, inter_eye) * 180 / np.pi
    roll = np.arctan2(vy, vx) * 180 / np.pi

    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

# ================== 中轴/水平线夹角 ==================
def compute_axis_angles(landmarks, img_w, img_h):
    nb = landmarks[IDX_NOSE_BRIDGE]
    chin = landmarks[IDX_CHIN]
    left_eye = landmarks[IDX_LEFT_EYE_OUTER]
    right_eye = landmarks[IDX_RIGHT_EYE_OUTER]

    vx_vert = (chin.x - nb.x) * img_w
    vy_vert = (chin.y - nb.y) * img_h
    vertical_angle = np.arctan2(vx_vert, vy_vert) * 180 / np.pi

    vx_horiz = (right_eye.x - left_eye.x) * img_w
    vy_horiz = (right_eye.y - left_eye.y) * img_h
    horizontal_angle = np.arctan2(vy_horiz, vx_horiz) * 180 / np.pi

    return vertical_angle, horizontal_angle

# ================== 遮挡检测 ==================
def check_occlusion(landmarks):
    for idx in OCCLUSION_POINTS:
        if not landmarks[idx]:
            return True
    return False

# ================== 坐姿不良警报触发 ==================
def trigger_alert(reason, alert_type, current_time):
    last = state.last_alert_times[alert_type]
    if current_time - last >= REPEAT_ALERT_INTERVAL:
        state.last_alert_times[alert_type] = current_time
        print(f"[{time.strftime('%H:%M:%S')}] ALERT: {reason}")
        threading.Thread(target=beep, daemon=True).start()
        state.any_alert_active = True
    else:
        state.any_alert_active = True

# ================== 主循环 ==================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    break_manager = BreakWindowManager()

    print("Posture detection started. Press 'q' to quit, 'c' to calibrate baseline.")
    print(f"无人脸警报延迟: {MISSING_FACE_THRESHOLD}s, 久坐缓冲: {SIT_LOST_TIMEOUT}s")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        present = False
        pose = None
        vert_angle = None
        horiz_angle = None
        occlusion = False
        vertical_offset = 0.0
        any_alert = False

        if results.multi_face_landmarks:
            present = True
            landmarks = results.multi_face_landmarks[0].landmark
            pose = estimate_head_pose(landmarks, img_w, img_h)
            vert_angle, horiz_angle = compute_axis_angles(landmarks, img_w, img_h)
            occlusion = check_occlusion(landmarks)

            left_eye = landmarks[IDX_LEFT_EYE_OUTER]
            right_eye = landmarks[IDX_RIGHT_EYE_OUTER]
            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            vertical_offset = abs(eye_center_y - 0.5)

            state.last_face_seen = current_time
            state.last_pose = pose

            # 1. 遮挡检测
            if occlusion:
                if not state.occlusion_active:
                    state.occlusion_active = True
                trigger_alert("Face occluded (nose/mouth/chin)", 'occlusion', current_time)
                any_alert = True
            else:
                state.occlusion_active = False

            # 2. 姿态角度异常（相对于基准）
            pose_out = False
            if pose and state.baseline:
                dy = pose['yaw'] - state.baseline['yaw']
                dp = pose['pitch'] - state.baseline['pitch']
                dr = pose['roll'] - state.baseline['roll']
                pose_out = (abs(dy) > YAW_THRESHOLD or
                            abs(dp) > PITCH_THRESHOLD or
                            abs(dr) > ROLL_THRESHOLD)
            if pose_out:
                if not state.pose_off_start:
                    state.pose_off_start = current_time
                if current_time - state.pose_off_start >= POSE_OFF_DURATION:
                    trigger_alert("Pose angle deviation (head turned/tilted)", 'pose', current_time)
                    any_alert = True
            else:
                state.pose_off_start = None

            # 3. 轴线角度异常（绝对阈值）
            if vert_angle is not None and horiz_angle is not None:
                abs_vert = abs(vert_angle)
                abs_horiz = abs(horiz_angle)
                if abs_vert > VERTICAL_AXIS_THRESHOLD or abs_horiz > HORIZONTAL_AXIS_THRESHOLD:
                    reason = f"Head tilt: Vert {abs_vert:.1f}°  Horiz {abs_horiz:.1f}°"
                    trigger_alert(reason, 'axis', current_time)
                    any_alert = True

            # 4. 眼睛中心垂直偏移检测
            if vertical_offset > VERTICAL_OFFSET_THRESHOLD:
                reason = f"Eye level offset: {vertical_offset:.2f} (> {VERTICAL_OFFSET_THRESHOLD:.2f})"
                trigger_alert(reason, 'vertical_offset', current_time)
                any_alert = True

        else:
            # 无人脸检测
            if state.last_face_seen is not None:
                since = current_time - state.last_face_seen
                if since > MISSING_FACE_THRESHOLD:
                    trigger_alert("No face detected (possible leave)", 'missing', current_time)
                    any_alert = True
            state.pose_off_start = None
            state.occlusion_active = False

        state.any_alert_active = any_alert

        # ========== 久坐计时（带缓冲） ==========
        if present and not occlusion:
            # 有人脸且未被遮挡 -> 认为正在坐着
            if state.sit_lost_start is not None:
                # 之前丢失，现在恢复，累计时间保持不变（继续累计）
                state.sit_lost_start = None
            # 累计坐姿时间
            delta = current_time - prev_time
            state.sit_accumulated += delta
        else:
            # 无人脸或遮挡 -> 开始丢失计时
            if state.sit_lost_start is None:
                state.sit_lost_start = current_time
            else:
                lost_duration = current_time - state.sit_lost_start
                if lost_duration > SIT_LOST_TIMEOUT:
                    # 丢失超过缓冲，重置累计时间
                    state.sit_accumulated = 0.0
                    state.sit_lost_start = None  # 等待下次有人再重新计时

        # 久坐提醒触发
        if state.sit_accumulated >= SIT_DURATION:
            if current_time - state.last_break_alert_time >= BREAK_DURATION:
                print(f"Sit accumulated reached {state.sit_accumulated:.0f}s, showing break window")
                break_manager.show(BREAK_DURATION)
                state.last_break_alert_time = current_time
                # 重置累计时间，开始新的周期
                state.sit_accumulated = 0.0

        # ========== 绘制信息 ==========
        cv2.line(frame, (img_w//2, 0), (img_w//2, img_h), (255, 255, 255), 1)
        cv2.line(frame, (0, img_h//2), (img_w, img_h//2), (255, 255, 255), 1)
        cv2.rectangle(frame, (int(img_w*0.3), int(img_h*0.2)),
                      (int(img_w*0.7), int(img_h*0.8)), (0, 255, 0), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for idx in OCCLUSION_POINTS:
                if landmarks[idx]:
                    x = int(landmarks[idx].x * img_w)
                    y = int(landmarks[idx].y * img_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            x_nose = int(landmarks[IDX_NOSE].x * img_w)
            y_nose = int(landmarks[IDX_NOSE].y * img_h)
            cv2.circle(frame, (x_nose, y_nose), 4, (0, 0, 255), -1)

            x_left = int(landmarks[IDX_LEFT_EYE_OUTER].x * img_w)
            y_left = int(landmarks[IDX_LEFT_EYE_OUTER].y * img_h)
            x_right = int(landmarks[IDX_RIGHT_EYE_OUTER].x * img_w)
            y_right = int(landmarks[IDX_RIGHT_EYE_OUTER].y * img_h)
            cv2.circle(frame, (x_left, y_left), 3, (255, 0, 0), -1)
            cv2.circle(frame, (x_right, y_right), 3, (255, 0, 0), -1)

            x_nb = int(landmarks[IDX_NOSE_BRIDGE].x * img_w)
            y_nb = int(landmarks[IDX_NOSE_BRIDGE].y * img_h)
            x_chin = int(landmarks[IDX_CHIN].x * img_w)
            y_chin = int(landmarks[IDX_CHIN].y * img_h)
            cv2.line(frame, (x_nb, y_nb), (x_chin, y_chin), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x_left, y_left), (x_right, y_right), (255, 0, 0), 2, cv2.LINE_AA)

            if pose:
                cv2.putText(frame, f"Yaw: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Roll: {pose['roll']:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if vert_angle is not None:
                cv2.putText(frame, f"Vert: {vert_angle:.1f}°  Horiz: {horiz_angle:.1f}°",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"|Vert|: {abs(vert_angle):.2f}°  |Horiz|: {abs(horiz_angle):.2f}°",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Eye offset: {vertical_offset:.3f} (thresh {VERTICAL_OFFSET_THRESHOLD})",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            occ_text = "Occlusion: Yes" if occlusion else "Occlusion: No"
            color = (0, 0, 255) if occlusion else (0, 255, 0)
            cv2.putText(frame, occ_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if state.baseline:
                cv2.putText(frame, f"Baseline: Y {state.baseline['yaw']:.1f} P {state.baseline['pitch']:.1f} R {state.baseline['roll']:.1f}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Baseline: not set", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            # 警报时间
            time_since_axis = current_time - state.last_alert_times['axis']
            time_since_vertical = current_time - state.last_alert_times['vertical_offset']
            cv2.putText(frame, f"Last axis: {time_since_axis:.1f}s  Last vert: {time_since_vertical:.1f}s",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 无人脸倒计时
            if not present:
                if state.last_face_seen is not None:
                    elapsed = current_time - state.last_face_seen
                    remain = max(0, MISSING_FACE_THRESHOLD - elapsed)
                    cv2.putText(frame, f"No face in {remain:.1f}s -> alert",
                                (img_w-300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        else:
            cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示久坐计时和缓冲状态
        if state.sit_lost_start is not None:
            lost_remain = max(0, SIT_LOST_TIMEOUT - (current_time - state.sit_lost_start))
            status = f"Sit: {int(state.sit_accumulated//60)}:{int(state.sit_accumulated%60):02d} (lost pause, {lost_remain:.0f}s to reset)"
        else:
            status = f"Sit: {int(state.sit_accumulated//60)}:{int(state.sit_accumulated%60):02d} / {SIT_DURATION//60}min"
        cv2.putText(frame, status, (img_w-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (img_w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        alert_count = sum(1 for t in state.last_alert_times if state.last_alert_times[t] > 0)
        cv2.putText(frame, f"Alerts: {alert_count}", (img_w-100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Posture Detection (runs in background)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if state.last_pose:
                state.baseline = state.last_pose.copy()
                print(f"Calibrated: yaw={state.baseline['yaw']:.1f} pitch={state.baseline['pitch']:.1f} roll={state.baseline['roll']:.1f}")
            else:
                print("Calibration failed: no pose data")

    cap.release()
    cv2.destroyAllWindows()
    break_manager.close()

if __name__ == "__main__":
    main()
