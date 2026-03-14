import cv2
import mediapipe as mp
import numpy as np
import time
import simpleaudio as sa
import threading
import tkinter as tk
from tkinter import font as tkfont

# ================== 配置参数 ==================
MISSING_FACE_THRESHOLD = 1.2          # 无人脸超过1.2秒触发
YAW_THRESHOLD = 20                     # 左右转头阈值（度，相对于基准）
PITCH_THRESHOLD = 25                    # 低头/抬头阈值
ROLL_THRESHOLD = 15                      # 头部倾斜阈值（用于姿态角度）
POSE_OFF_DURATION = 0.8                  # 姿态异常持续0.8秒触发

VERTICAL_AXIS_THRESHOLD = 3.9            # 中轴倾角偏差阈值（度）
HORIZONTAL_AXIS_THRESHOLD = 2.7          # 水平线倾角偏差阈值

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
mp_drawing = mp.solutions.drawing_utils

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

# ================== 全局状态 ==================
class State:
    def __init__(self):
        self.last_face_seen = None
        self.pose_off_start = None
        self.last_alert_times = {
            'axis': 0,
            'occlusion': 0,
            'missing': 0,
            'pose': 0
        }
        self.baseline = None               # 姿态基准 {yaw, pitch, roll}
        self.last_pose = None               # 最近一次姿态
        self.occlusion_active = False       # 当前是否遮挡中
        self.any_alert_active = False       # 是否有任意警报正在持续

        # 久坐相关
        self.sit_start_time = None          # 本次坐下的开始时间（有人脸且无遮挡？通常只要有人脸就算坐下）
        self.break_active = False           # 是否正在显示弹窗
        self.last_break_alert_time = 0      # 上次弹窗的时间，用于冷却

state = State()

# ================== 弹窗显示（独立线程） ==================
def show_break_window(duration):
    """显示全屏半透明倒计时窗口，持续duration秒后自动关闭"""
    def run_tk():
        root = tk.Tk()
        root.title("久坐提醒")
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.8)       # 半透明
        root.attributes('-topmost', True)    # 置顶
        root.configure(bg='black')
        root.overrideredirect(True)           # 无边框

        # 使用大字体
        large_font = tkfont.Font(size=48, weight='bold')
        label_msg = tk.Label(root, text="久坐提醒！请起身活动", fg='red', bg='black', font=large_font)
        label_msg.pack(expand=True)

        time_var = tk.StringVar()
        time_var.set(f"{duration//60}:{duration%60:02d} 分钟后自动关闭")
        label_time = tk.Label(root, textvariable=time_var, fg='yellow', bg='black', font=('Arial', 36))
        label_time.pack(expand=True)

        # 倒计时更新函数
        def countdown(remaining):
            if remaining <= 0:
                root.destroy()
                state.break_active = False
                return
            mins, secs = divmod(remaining, 60)
            time_var.set(f"{mins}:{secs:02d} 分钟后自动关闭")
            root.after(1000, countdown, remaining - 1)

        root.after(1000, countdown, duration - 1)  # 立即开始倒计时（减1秒因为已经过1秒）
        root.mainloop()

    if not state.break_active:
        state.break_active = True
        threading.Thread(target=run_tk, daemon=True).start()

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

# ================== 坐姿不良警报触发（带重复间隔） ==================
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

    print("Posture detection started. Press 'q' to quit, 'c' to calibrate baseline.")

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
        any_alert = False

        if results.multi_face_landmarks:
            present = True
            landmarks = results.multi_face_landmarks[0].landmark
            pose = estimate_head_pose(landmarks, img_w, img_h)
            vert_angle, horiz_angle = compute_axis_angles(landmarks, img_w, img_h)
            occlusion = check_occlusion(landmarks)

            state.last_face_seen = current_time
            state.last_pose = pose

            # 坐姿检测逻辑（保持不变）
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

        else:
            # 无人脸检测
            if state.last_face_seen is not None:
                since = current_time - state.last_face_seen
                if since > MISSING_FACE_THRESHOLD:
                    trigger_alert("No face detected", 'missing', current_time)
                    any_alert = True
            state.pose_off_start = None
            state.occlusion_active = False

        state.any_alert_active = any_alert

        # ========== 久坐计时和提醒 ==========
        # 定义“坐着”的条件：有人脸且没有被遮挡（可选，可根据需要调整）
        if present and not occlusion:
            if state.sit_start_time is None:
                state.sit_start_time = current_time
            else:
                sit_duration = current_time - state.sit_start_time
                # 如果达到久坐时间，并且没有正在显示的弹窗，并且距离上次弹窗已经超过休息时长（避免连续弹）
                if sit_duration >= SIT_DURATION and not state.break_active:
                    if current_time - state.last_break_alert_time >= BREAK_DURATION:
                        print(f"Sit duration reached {sit_duration:.0f}s, showing break window")
                        show_break_window(BREAK_DURATION)
                        state.last_break_alert_time = current_time
                        # 重置计时器，开始新的久坐周期（从0开始）
                        state.sit_start_time = current_time
        else:
            # 如果人离开或被遮挡，重置久坐计时器
            state.sit_start_time = None

        # ========== 绘制信息（全部英文） ==========
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
            occ_text = "Occlusion: Yes" if occlusion else "Occlusion: No"
            color = (0, 0, 255) if occlusion else (0, 255, 0)
            cv2.putText(frame, occ_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if state.baseline:
                cv2.putText(frame, f"Baseline: Y {state.baseline['yaw']:.1f} P {state.baseline['pitch']:.1f} R {state.baseline['roll']:.1f}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Baseline: not set", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            time_since_axis = current_time - state.last_alert_times['axis']
            cv2.putText(frame, f"Last axis alert: {time_since_axis:.1f}s ago",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 显示久坐计时
            if state.sit_start_time is not None:
                sit_elapsed = current_time - state.sit_start_time
                cv2.putText(frame, f"Sit time: {int(sit_elapsed//60)}:{int(sit_elapsed%60):02d} / {SIT_DURATION//60}min",
                            (img_w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Sit time: paused", (img_w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        else:
            cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

if __name__ == "__main__":
    main()
