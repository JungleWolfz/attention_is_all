import cv2
import mediapipe as mp
import numpy as np
import time
import simpleaudio as sa
import threading
from collections import deque

# ================== 配置参数 ==================
MISSING_FACE_THRESHOLD = 1.2          # 无人脸超过1.2秒触发
YAW_THRESHOLD = 20                     # 左右转头阈值（度，相对于基准）
PITCH_THRESHOLD = 25                    # 低头/抬头阈值
ROLL_THRESHOLD = 15                      # 头部倾斜阈值（用于姿态角度）
POSE_OFF_DURATION = 0.8                  # 姿态异常持续0.8秒触发

VERTICAL_AXIS_THRESHOLD = 2.7            # 中轴倾角偏差阈值（度）
HORIZONTAL_AXIS_THRESHOLD = 2.7          # 水平线倾角偏差阈值

REPEAT_ALERT_INTERVAL = 6.0               # 重复提醒间隔（秒）

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
        # 生成一个正弦波（频率880Hz，时长0.14秒）
        sample_rate = 44100
        t = np.linspace(0, 0.14, int(sample_rate * 0.14))
        wave = (0.1 * np.sin(880 * 2 * np.pi * t)).astype(np.float32)
        # 转换为16位整数
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

state = State()

# ================== 姿态估计 ==================
def estimate_head_pose(landmarks, img_w, img_h):
    """三点法估计 yaw/pitch/roll（相对于基准，这里只返回原始值）"""
    nose = landmarks[IDX_NOSE]
    left_eye = landmarks[IDX_LEFT_EYE_OUTER]
    right_eye = landmarks[IDX_RIGHT_EYE_OUTER]

    xN = nose.x * img_w
    yN = nose.y * img_h
    xL = left_eye.x * img_w
    yL = left_eye.y * img_h
    xR = right_eye.x * img_w
    yR = right_eye.y * img_h

    # 两眼连线
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
    """计算中轴线与垂直方向的夹角、水平线与水平方向的夹角"""
    nb = landmarks[IDX_NOSE_BRIDGE]
    chin = landmarks[IDX_CHIN]
    left_eye = landmarks[IDX_LEFT_EYE_OUTER]
    right_eye = landmarks[IDX_RIGHT_EYE_OUTER]

    # 中轴线向量
    vx_vert = (chin.x - nb.x) * img_w
    vy_vert = (chin.y - nb.y) * img_h
    vertical_angle = np.arctan2(vx_vert, vy_vert) * 180 / np.pi   # 向右为正

    # 水平线向量
    vx_horiz = (right_eye.x - left_eye.x) * img_w
    vy_horiz = (right_eye.y - left_eye.y) * img_h
    horizontal_angle = np.arctan2(vy_horiz, vx_horiz) * 180 / np.pi

    return vertical_angle, horizontal_angle

# ================== 遮挡检测 ==================
def check_occlusion(landmarks):
    """检查关键点是否全部存在"""
    for idx in OCCLUSION_POINTS:
        if not landmarks[idx]:
            return True
    return False

# ================== 警报触发（带重复间隔） ==================
def trigger_alert(reason, alert_type, current_time):
    last = state.last_alert_times[alert_type]
    if current_time - last >= REPEAT_ALERT_INTERVAL:
        state.last_alert_times[alert_type] = current_time
        print(f"[{time.strftime('%H:%M:%S')}] 警报: {reason}")
        # 在独立线程中播放蜂鸣，避免阻塞主循环
        threading.Thread(target=beep, daemon=True).start()
        state.any_alert_active = True
    else:
        # 虽未达到间隔，但只要有异常，仍标记为活跃
        state.any_alert_active = True

def clear_alert_if_needed(current_time):
    """如果所有异常都已消失，重置 any_alert_active"""
    # 该函数在每帧结束后调用，如果检测到没有异常，则清除
    pass  # 实际在每帧最后判断

# ================== 主循环 ==================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("坐姿检测已启动。按 'q' 退出，按 'c' 校准当前姿态。")

    # 用于计算帧率（可选）
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 初始化标志
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

            # ========== 检测逻辑 ==========
            # 1. 遮挡检测
            if occlusion:
                if not state.occlusion_active:
                    state.occlusion_active = True
                trigger_alert("面部被遮挡（鼻/嘴/下巴等）", 'occlusion', current_time)
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
                    trigger_alert("姿态角度异常（长时间转头/低头）", 'pose', current_time)
                    any_alert = True
            else:
                state.pose_off_start = None

            # 3. 轴线角度异常（绝对阈值）
            if vert_angle is not None and horiz_angle is not None:
                abs_vert = abs(vert_angle)
                abs_horiz = abs(horiz_angle)
                if abs_vert > VERTICAL_AXIS_THRESHOLD or abs_horiz > HORIZONTAL_AXIS_THRESHOLD:
                    reason = f"头部歪斜：中轴{abs_vert:.1f}° 水平{abs_horiz:.1f}°"
                    trigger_alert(reason, 'axis', current_time)
                    any_alert = True

        else:
            # 无人脸检测
            if state.last_face_seen is not None:
                since = current_time - state.last_face_seen
                if since > MISSING_FACE_THRESHOLD:
                    trigger_alert("无人脸超时，请回到屏幕前", 'missing', current_time)
                    any_alert = True
            # 重置其他状态
            state.pose_off_start = None
            state.occlusion_active = False

        # 更新全局活跃标志
        state.any_alert_active = any_alert

        # ========== 绘制信息 ==========
        # 绘制参考线
        cv2.line(frame, (img_w//2, 0), (img_w//2, img_h), (255, 255, 255), 1)
        cv2.line(frame, (0, img_h//2), (img_w, img_h//2), (255, 255, 255), 1)
        cv2.rectangle(frame, (int(img_w*0.3), int(img_h*0.2)),
                      (int(img_w*0.7), int(img_h*0.8)), (0, 255, 0), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # 绘制关键点
            for idx in OCCLUSION_POINTS:
                if landmarks[idx]:
                    x = int(landmarks[idx].x * img_w)
                    y = int(landmarks[idx].y * img_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 绘制鼻尖（红色）
            x_nose = int(landmarks[IDX_NOSE].x * img_w)
            y_nose = int(landmarks[IDX_NOSE].y * img_h)
            cv2.circle(frame, (x_nose, y_nose), 4, (0, 0, 255), -1)

            # 绘制眼角（蓝色）
            x_left = int(landmarks[IDX_LEFT_EYE_OUTER].x * img_w)
            y_left = int(landmarks[IDX_LEFT_EYE_OUTER].y * img_h)
            x_right = int(landmarks[IDX_RIGHT_EYE_OUTER].x * img_w)
            y_right = int(landmarks[IDX_RIGHT_EYE_OUTER].y * img_h)
            cv2.circle(frame, (x_left, y_left), 3, (255, 0, 0), -1)
            cv2.circle(frame, (x_right, y_right), 3, (255, 0, 0), -1)

            # 绘制中轴线（鼻梁到下巴）
            x_nb = int(landmarks[IDX_NOSE_BRIDGE].x * img_w)
            y_nb = int(landmarks[IDX_NOSE_BRIDGE].y * img_h)
            x_chin = int(landmarks[IDX_CHIN].x * img_w)
            y_chin = int(landmarks[IDX_CHIN].y * img_h)
            cv2.line(frame, (x_nb, y_nb), (x_chin, y_chin), (0, 0, 255), 2, cv2.LINE_AA)

            # 绘制水平线（两眼连线）
            cv2.line(frame, (x_left, y_left), (x_right, y_right), (255, 0, 0), 2, cv2.LINE_AA)

            # 显示角度信息
            if pose:
                cv2.putText(frame, f"Yaw: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Roll: {pose['roll']:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if vert_angle is not None:
                cv2.putText(frame, f"中轴: {vert_angle:.1f}° 水平: {horiz_angle:.1f}°",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"偏差: |中轴| {abs(vert_angle):.2f}°  |水平| {abs(horiz_angle):.2f}°",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if occlusion:
                cv2.putText(frame, "遮挡: 是", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "遮挡: 否", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示基准状态
            if state.baseline:
                cv2.putText(frame, f"基准: Y {state.baseline['yaw']:.1f} P {state.baseline['pitch']:.1f} R {state.baseline['roll']:.1f}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "基准: 未校准", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            # 显示计时器
            cv2.putText(frame, f"距上次轴线提醒: {current_time - state.last_alert_times['axis']:.1f}s",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            cv2.putText(frame, "无人脸", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示帧率和状态
        fps = 1.0 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (img_w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"警报计数: {sum(state.last_alert_times[t] > 0 for t in state.last_alert_times)}",
                    (img_w-150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('坐姿检测 - 后台持续运行', frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # 校准当前姿态
            if state.last_pose:
                state.baseline = state.last_pose.copy()
                print(f"校准完成: 基准 yaw={state.baseline['yaw']:.1f} pitch={state.baseline['pitch']:.1f} roll={state.baseline['roll']:.1f}")
            else:
                print("校准失败：未检测到有效姿态")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()