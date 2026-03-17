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
# 无人脸检测
MISSING_FACE_THRESHOLD = 5.0           # 无人脸超过5秒触发离开警报
SIT_LOST_TIMEOUT = 30                  # 人脸丢失后仍视为坐着的最大缓冲时间（秒）

# 姿态角度阈值（相对于基准）
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 25
ROLL_THRESHOLD = 15
POSE_OFF_DURATION = 0.8                 # 姿态异常持续0.8秒触发

# 中轴/水平线角度阈值（绝对值）
VERTICAL_AXIS_THRESHOLD = 3.9
HORIZONTAL_AXIS_THRESHOLD = 3.6
VERTICAL_OFFSET_THRESHOLD = 0.096         # 眼睛中心偏离画面中心超过15%

# 脖子前伸检测
NECK_EXTEND_THRESHOLD = 1.06              # 面部面积超过基准的1.2倍视为前伸
NECK_EXTEND_DURATION = 1.0                # 脖子前伸持续1秒触发警报

# 坐姿不良重复提醒间隔
REPEAT_ALERT_INTERVAL = 6.0               # 秒

# 久坐提醒参数
SIT_DURATION = 2700                       # 久坐时间（秒），默认45分钟
BREAK_DURATION = 300                       # 休息时间（秒），默认5分钟

# ================== MediaPipe 关键点索引 ==================
IDX_NOSE = 1
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263
IDX_NOSE_BRIDGE = 168
IDX_CHIN = 152
# 遮挡检测点：鼻尖、下巴、上下嘴唇、左右嘴角、左右鼻孔
OCCLUSION_POINTS = [1, 152, 13, 14, 61, 291, 2, 19]
# 用于计算面部边界框的关键点（面部轮廓 + 额头等，这里简单使用所有点，实际可用轮廓点，但用所有点也行）
# 我们使用所有 landmark 来求最小包围矩形

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

# ================== 弹窗管理器 ==================
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
        self.neck_off_start = None                # 脖子前伸计时开始
        self.last_alert_times = {
            'axis': 0,
            'occlusion': 0,
            'missing': 0,
            'pose': 0,
            'vertical_offset': 0,
            'neck': 0                              # 脖子前伸警报
        }
        self.baseline_pose = None                  # 姿态基准 {yaw, pitch, roll}
        self.baseline_face_area = None              # 脖子前伸基准面积
        self.last_pose = None
        self.occlusion_active = False
        self.any_alert_active = False

        # 久坐相关
        self.sit_accumulated = 0.0
        self.sit_lost_start = None
        self.last_break_alert_time = 0

state = State()

# ================== 工具函数 ==================
def compute_face_area(landmarks, img_w, img_h):
    """计算面部边界框面积（使用所有 landmark 的最小包围矩形）"""
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    return width * height

def estimate_head_pose(landmarks, img_w, img_h):
    """三点法估计 yaw/pitch/roll"""
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

def compute_axis_angles(landmarks, img_w, img_h):
    """中轴线与垂直方向夹角、水平线与水平方向夹角"""
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

def check_occlusion(landmarks):
    """检查遮挡：任一关键点缺失"""
    for idx in OCCLUSION_POINTS:
        if not landmarks[idx]:
            return True
    return False

def trigger_alert(reason, alert_type, current_time):
    """触发警报（带重复间隔控制）"""
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
    print(f"无人脸延迟: {MISSING_FACE_THRESHOLD}s, 久坐缓冲: {SIT_LOST_TIMEOUT}s, 脖子前伸阈值: {NECK_EXTEND_THRESHOLD}x")

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
        face_area = 0.0
        any_alert = False

        if results.multi_face_landmarks:
            present = True
            landmarks = results.multi_face_landmarks[0].landmark
            pose = estimate_head_pose(landmarks, img_w, img_h)
            vert_angle, horiz_angle = compute_axis_angles(landmarks, img_w, img_h)
            occlusion = check_occlusion(landmarks)

            # 眼睛垂直偏移
            left_eye = landmarks[IDX_LEFT_EYE_OUTER]
            right_eye = landmarks[IDX_RIGHT_EYE_OUTER]
            eye_center_y = (left_eye.y + right_eye.y) / 2.0
            vertical_offset = abs(eye_center_y - 0.5)

            # 面部面积（用于脖子前伸）
            face_area = compute_face_area(landmarks, img_w, img_h)

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
            if pose and state.baseline_pose:
                dy = pose['yaw'] - state.baseline_pose['yaw']
                dp = pose['pitch'] - state.baseline_pose['pitch']
                dr = pose['roll'] - state.baseline_pose['roll']
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

            # 3. 轴线角度异常（绝对值）
            if vert_angle is not None and horiz_angle is not None:
                abs_vert = abs(vert_angle)
                abs_horiz = abs(horiz_angle)
                if abs_vert > VERTICAL_AXIS_THRESHOLD or abs_horiz > HORIZONTAL_AXIS_THRESHOLD:
                    reason = f"Head tilt: Vert {abs_vert:.1f}°  Horiz {abs_horiz:.1f}°"
                    trigger_alert(reason, 'axis', current_time)
                    any_alert = True

            # 4. 眼睛垂直偏移
            if vertical_offset > VERTICAL_OFFSET_THRESHOLD:
                reason = f"Eye level offset: {vertical_offset:.2f} (> {VERTICAL_OFFSET_THRESHOLD:.2f})"
                trigger_alert(reason, 'vertical_offset', current_time)
                any_alert = True

            # 5. 脖子前伸检测
            if state.baseline_face_area is not None and face_area > 0:
                area_ratio = face_area / state.baseline_face_area
                if area_ratio > NECK_EXTEND_THRESHOLD:
                    if state.neck_off_start is None:
                        state.neck_off_start = current_time
                    if current_time - state.neck_off_start >= NECK_EXTEND_DURATION:
                        reason = f"Neck forward: area {area_ratio:.2f}x baseline"
                        trigger_alert(reason, 'neck', current_time)
                        any_alert = True
                else:
                    state.neck_off_start = None
            else:
                state.neck_off_start = None

        else:
            # 无人脸
            if state.last_face_seen is not None:
                since = current_time - state.last_face_seen
                if since > MISSING_FACE_THRESHOLD:
                    trigger_alert("No face detected (possible leave)", 'missing', current_time)
                    any_alert = True
            state.pose_off_start = None
            state.occlusion_active = False
            state.neck_off_start = None

        state.any_alert_active = any_alert

        # ========== 久坐计时（带缓冲） ==========
        if present and not occlusion:
            if state.sit_lost_start is not None:
                state.sit_lost_start = None
            delta = current_time - prev_time
            state.sit_accumulated += delta
        else:
            if state.sit_lost_start is None:
                state.sit_lost_start = current_time
            else:
                lost_duration = current_time - state.sit_lost_start
                if lost_duration > SIT_LOST_TIMEOUT:
                    state.sit_accumulated = 0.0
                    state.sit_lost_start = None

        if state.sit_accumulated >= SIT_DURATION:
            if current_time - state.last_break_alert_time >= BREAK_DURATION:
                print(f"Sit accumulated reached {state.sit_accumulated:.0f}s, showing break window")
                break_manager.show(BREAK_DURATION)
                state.last_break_alert_time = current_time
                state.sit_accumulated = 0.0

        # ========== 绘制信息 ==========
        # 参考线
        cv2.line(frame, (img_w//2, 0), (img_w//2, img_h), (255, 255, 255), 1)
        cv2.line(frame, (0, img_h//2), (img_w, img_h//2), (255, 255, 255), 1)
        cv2.rectangle(frame, (int(img_w*0.3), int(img_h*0.2)),
                      (int(img_w*0.7), int(img_h*0.8)), (0, 255, 0), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # 绘制遮挡检测点（绿色）
            for idx in OCCLUSION_POINTS:
                if landmarks[idx]:
                    x = int(landmarks[idx].x * img_w)
                    y = int(landmarks[idx].y * img_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 鼻尖（红）
            x_nose = int(landmarks[IDX_NOSE].x * img_w)
            y_nose = int(landmarks[IDX_NOSE].y * img_h)
            cv2.circle(frame, (x_nose, y_nose), 4, (0, 0, 255), -1)

            # 眼角（蓝）
            x_left = int(landmarks[IDX_LEFT_EYE_OUTER].x * img_w)
            y_left = int(landmarks[IDX_LEFT_EYE_OUTER].y * img_h)
            x_right = int(landmarks[IDX_RIGHT_EYE_OUTER].x * img_w)
            y_right = int(landmarks[IDX_RIGHT_EYE_OUTER].y * img_h)
            cv2.circle(frame, (x_left, y_left), 3, (255, 0, 0), -1)
            cv2.circle(frame, (x_right, y_right), 3, (255, 0, 0), -1)

            # 中轴线（红虚线）
            x_nb = int(landmarks[IDX_NOSE_BRIDGE].x * img_w)
            y_nb = int(landmarks[IDX_NOSE_BRIDGE].y * img_h)
            x_chin = int(landmarks[IDX_CHIN].x * img_w)
            y_chin = int(landmarks[IDX_CHIN].y * img_h)
            cv2.line(frame, (x_nb, y_nb), (x_chin, y_chin), (0, 0, 255), 2, cv2.LINE_AA)

            # 水平线（蓝虚线）
            cv2.line(frame, (x_left, y_left), (x_right, y_right), (255, 0, 0), 2, cv2.LINE_AA)

            # 文字信息
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

            if state.baseline_pose:
                cv2.putText(frame, f"Baseline: Y {state.baseline_pose['yaw']:.1f} P {state.baseline_pose['pitch']:.1f} R {state.baseline_pose['roll']:.1f}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Baseline: not set", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            # 脖子前伸信息
            if state.baseline_face_area is not None and face_area > 0:
                area_ratio = face_area / state.baseline_face_area
                cv2.putText(frame, f"Face area ratio: {area_ratio:.2f}x (thresh {NECK_EXTEND_THRESHOLD})",
                            (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Face area ratio: N/A", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # 警报计时
            time_since_axis = current_time - state.last_alert_times['axis']
            time_since_neck = current_time - state.last_alert_times['neck']
            cv2.putText(frame, f"Last axis: {time_since_axis:.1f}s  Last neck: {time_since_neck:.1f}s",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 无人脸倒计时
            if not present and state.last_face_seen is not None:
                elapsed = current_time - state.last_face_seen
                remain = max(0, MISSING_FACE_THRESHOLD - elapsed)
                cv2.putText(frame, f"No face in {remain:.1f}s -> alert",
                            (img_w-300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        else:
            cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 久坐计时显示
        if state.sit_lost_start is not None:
            lost_remain = max(0, SIT_LOST_TIMEOUT - (current_time - state.sit_lost_start))
            status = f"Sit: {int(state.sit_accumulated//60)}:{int(state.sit_accumulated%60):02d} (lost pause, {lost_remain:.0f}s to reset)"
        else:
            status = f"Sit: {int(state.sit_accumulated//60)}:{int(state.sit_accumulated%60):02d} / {SIT_DURATION//60}min"
        cv2.putText(frame, status, (img_w-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
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
            if results.multi_face_landmarks:
                # 记录基准
                state.baseline_pose = pose.copy() if pose else None
                # 记录基准面部面积
                if face_area > 0:
                    state.baseline_face_area = face_area
                    print(f"Calibrated: pose={state.baseline_pose}, face_area={state.baseline_face_area:.0f}")
                else:
                    print("Calibration failed: face area zero")
            else:
                print("Calibration failed: no face detected")

    cap.release()
    cv2.destroyAllWindows()
    break_manager.close()

if __name__ == "__main__":
    main()
