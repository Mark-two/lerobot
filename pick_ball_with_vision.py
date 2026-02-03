"""
视觉引导抓取脚本
使用 RealSense 相机识别小球位置，然后用机械臂抓取并放置到目标位置
"""
import time
import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pathlib import Path
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO

# --- 引入 LeRobot 的核心硬件接口 ---
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

# ================= 配置 =================
ROBOT_CONFIG = SO101FollowerConfig(
    port="/dev/ttyACM0",
)

# YOLO 模型路径
YOLO_MODEL_PATH = '/home/kang/Documents/yolo/runs/detect/runs/detect/cat_ball3_train2/weights/best.pt'

# 任务位置 (米) - 机器人基座坐标系下
INIT_POS = np.array([0.25, 0.0, 0.1])       # 初始/安全位置
# 两个目标位置，循环交替
TARGET_POSITIONS = [
    np.array([0.2, -0.15, 0.02]),   # 目标位置 1
    np.array([0.3, 0.05, 0.02]),    # 目标位置 2
]

# 相机到机器人基座的坐标变换参数 (需要根据实际标定调整!)
# 假设相机在机器人前方，朝向机器人
# 这里是一个示例变换，你需要根据实际情况标定
CAMERA_TO_ROBOT_OFFSET = np.array([0.0, 0.0, 0.0])  # [x, y, z] 偏移
CAMERA_TO_ROBOT_ROTATION = np.eye(3)  # 旋转矩阵 (如果相机坐标系与机器人不同)

# 相机坐标系到机器人坐标系的简单映射
# RealSense 相机坐标系: x-右, y-下, z-前
# 机器人坐标系: x-前, y-左, z-上
# 相机位置: 在机器人基座右侧 10cm (即 y = -0.1m)
CAMERA_OFFSET_Y = -0.03  # 相机在机器人右侧 10cm

def camera_to_robot_coords(camera_point):
    """
    将相机坐标系下的点转换到机器人基座坐标系
    相机在机器人右侧 10cm，朝向与机器人相同
    """
    x_cam, y_cam, z_cam = camera_point
    
    # 坐标轴映射:
    # 相机 z (前) -> 机器人 x (前)
    # 相机 x (右) -> 机器人 -y (右是-y方向)
    # 相机 y (下) -> 机器人 -z (下是-z方向)
    
    x_robot = z_cam  + 0.03         # 相机前方距离 -> 机器人前方
    y_robot = -x_cam - 0.05  # 相机右侧 -> 机器人右侧(-y)，加上相机位置偏移
    z_robot = -y_cam          # 相机下方 -> 机器人上方(取反)
    
    print(f"  [坐标变换调试]")
    print(f"    相机坐标: x={x_cam:.4f}, y={y_cam:.4f}, z={z_cam:.4f}")
    print(f"    机器人坐标: x={x_robot:.4f}, y={y_robot:.4f}, z={z_robot:.4f}")
    print(f"    相机偏移: CAMERA_OFFSET_Y = {CAMERA_OFFSET_Y}")
    
    return np.array([x_robot, y_robot, z_robot])


# 夹爪控制
GRIPPER_OPEN = 50.0
GRIPPER_CLOSE = 0.0

# 电机名称
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# ================= 机器人控制函数 =================
def solve_ik_to_position(configuration, position_task, target_pos, end_effector_frame, max_iterations=200, tolerance=0.001):
    """求解 IK 到目标位置"""
    target_rotation = pin.utils.rpyToMatrix(0, np.pi, 0)
    target_pose = pin.SE3(target_rotation, target_pos)
    position_task.set_target(target_pose)
    
    dt = 0.1
    for iteration in range(max_iterations):
        velocity = solve_ik(configuration, [position_task], dt=dt, solver="quadprog", safety_break=False)
        configuration.integrate_inplace(velocity, dt)
        
        current_pose = configuration.get_transform_frame_to_world(end_effector_frame)
        pos_error = np.linalg.norm(current_pose.translation - target_pos)
        
        if pos_error < tolerance:
            print(f"  IK 在第 {iteration+1} 次迭代收敛, 误差: {pos_error*1000:.2f} mm")
            break
    
    return configuration.q.copy()


def move_to_joints(robot, current_q_rad, target_q_rad, steps=100, dt=0.02):
    """平滑移动到目标关节角度"""
    for i in range(steps):
        interp_q_rad = current_q_rad + (target_q_rad - current_q_rad) * ((i+1)/steps)
        interp_q_deg = np.rad2deg(interp_q_rad)
        action = {f"{name}.pos": interp_q_deg[idx] for idx, name in enumerate(MOTOR_NAMES)}
        robot.send_action(action)
        time.sleep(dt)


def set_gripper(robot, gripper_pos, duration=0.5):
    """控制夹爪开合"""
    obs = robot.get_observation()
    current_q = np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])
    
    steps = int(duration / 0.02)
    start_gripper = current_q[5]
    
    for i in range(steps):
        interp_gripper = start_gripper + (gripper_pos - start_gripper) * ((i+1)/steps)
        current_q[5] = interp_gripper
        action = {f"{name}.pos": current_q[idx] for idx, name in enumerate(MOTOR_NAMES)}
        robot.send_action(action)
        time.sleep(0.02)


def get_current_q(robot):
    """读取当前关节角度"""
    obs = robot.get_observation()
    current_q = np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])
    return current_q, np.deg2rad(current_q)


# ================= 视觉检测函数 =================
def detect_ball_position(pipeline, align, model, intrinsics, timeout=10.0):
    """
    使用 RealSense 和 YOLO 检测小球位置
    返回: 相机坐标系下的 3D 坐标 [x, y, z] (米), 如果没检测到返回 None
    """
    start_time = time.time()
    detected_positions = []
    
    print("正在检测小球位置...")
    
    while time.time() - start_time < timeout:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # YOLO 推理
        results = model(color_image, stream=True, verbose=False, conf=0.3)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 获取边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                
                if u >= 640 or v >= 480:
                    continue
                
                # 获取深度
                dist = aligned_depth_frame.get_distance(u, v)
                
                if dist > 0.05 and dist < 1.0:  # 有效距离范围 5cm - 1m
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], dist)
                    
                    print(f"  检测到物体: {model.names[cls_id]}, 置信度: {conf:.2f}")
                    print(f"  相机坐标: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}) m")
                    
                    detected_positions.append(point_3d)
                    
                    # 显示检测结果
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.putText(color_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Ball Detection', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 如果检测到多个位置，取平均
        if len(detected_positions) >= 5:
            avg_pos = np.mean(detected_positions, axis=0)
            print(f"  平均相机坐标: ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}) m")
            return avg_pos
    
    if len(detected_positions) > 0:
        return np.mean(detected_positions, axis=0)
    
    return None


def main():
    # ========== 1. 初始化 RealSense ==========
    print("正在初始化 RealSense 相机...")
    
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            print("发现设备，正在重置...")
            dev.hardware_reset()
        print("等待设备重连中 (5秒)...")
        time.sleep(5)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")
    
    align = rs.align(rs.stream.color)
    
    # 获取内参
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
    
    # ========== 2. 加载 YOLO 模型 ==========
    print("正在加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # ========== 3. 初始化机器人 ==========
    print("正在连接机器人...")
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()
    print("机器人连接成功！")
    
    # 加载 URDF
    urdf_path = Path("SO101/so101_new_calib.urdf")
    if not urdf_path.exists():
        print(f"错误: 找不到 URDF 文件: {urdf_path}")
        pipeline.stop()
        return
    
    print("正在加载 Pinocchio 模型...")
    pin_robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dirs=["."])
    configuration = pink.Configuration(pin_robot.model, pin_robot.data, pin_robot.q0)
    
    end_effector_frame = "gripper_frame_link"
    
    position_task = FrameTask(
        end_effector_frame,
        position_cost=1.0,
        orientation_cost=[0.1, 0.1, 0.0],
    )
    position_task.lm_damping = 1e-4
    
    current_q_deg, current_q_rad = get_current_q(robot)
    configuration.update(current_q_rad)
    
    cycle_count = 0  # 循环计数器
    
    try:
        while True:
            cycle_count += 1
            # 选择当前目标位置 (交替)
            target_pos = TARGET_POSITIONS[(cycle_count - 1) % len(TARGET_POSITIONS)]
            print(f"\n{'='*50}")
            print(f"========== 第 {cycle_count} 次抓取循环 ==========")
            print(f"目标位置: {target_pos}")
            print(f"{'='*50}")
            
            # ========== 步骤 1: 移动到初始位置 ==========
            print("\n[步骤 1] 移动到初始位置...")
            q_init = solve_ik_to_position(configuration, position_task, INIT_POS, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_init)
            time.sleep(0.5)
            
            # ========== 步骤 2: 打开夹爪 ==========
            print("\n[步骤 2] 打开夹爪...")
            set_gripper(robot, GRIPPER_OPEN)
            time.sleep(0.3)
            
            # ========== 步骤 3: 检测小球位置 ==========
            print("\n[步骤 3] 检测小球位置...")
            camera_ball_pos = detect_ball_position(pipeline, align, model, intrinsics, timeout=5.0)
            
            if camera_ball_pos is None:
                print("未检测到小球！等待3秒后重试...")
                time.sleep(3.0)
                continue  # 跳过这次循环，重新检测
            
            # 将相机坐标转换为机器人坐标
            ball_pos_robot = camera_to_robot_coords(camera_ball_pos)
            print(f"  机器人坐标系下的小球位置: {ball_pos_robot}")
            
            # 安全检查：确保位置在机器人工作空间内
            if ball_pos_robot[0] < 0.1 or ball_pos_robot[0] > 0.35:
                print("警告: 小球 X 坐标超出范围，调整中...")
                ball_pos_robot[0] = np.clip(ball_pos_robot[0], 0.15, 0.30)
            if abs(ball_pos_robot[1]) > 0.2:
                print("警告: 小球 Y 坐标超出范围，调整中...")
                ball_pos_robot[1] = np.clip(ball_pos_robot[1], -0.15, 0.15)
            if ball_pos_robot[2] < 0.0 or ball_pos_robot[2] > 0.2:
                print("警告: 小球 Z 坐标超出范围，调整中...")
                ball_pos_robot[2] = np.clip(ball_pos_robot[2], 0.01, 0.15)
            
            print(f"  最终抓取位置: {ball_pos_robot}")
            
            # ========== 步骤 4: 移动到小球上方 ==========
            print("\n[步骤 4] 移动到小球上方...")
            ball_above_pos = ball_pos_robot.copy()
            ball_above_pos[2] += 0.05  # 先到上方 5cm
            
            current_q_deg, current_q_rad = get_current_q(robot)
            configuration.update(current_q_rad)
            q_above = solve_ik_to_position(configuration, position_task, ball_above_pos, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_above)
            time.sleep(0.3)
            
            # ========== 步骤 5: 下降到小球位置 ==========
            print("\n[步骤 5] 下降到小球位置...")
            current_q_deg, current_q_rad = get_current_q(robot)
            configuration.update(current_q_rad)
            q_ball = solve_ik_to_position(configuration, position_task, ball_pos_robot, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_ball, steps=150)
            time.sleep(0.3)
            
            # ========== 步骤 6: 闭合夹爪抓取 ==========
            print("\n[步骤 6] 闭合夹爪抓取小球...")
            set_gripper(robot, GRIPPER_CLOSE, duration=2.0)  # 慢慢闭合，避免夹飞
            time.sleep(0.5)
            
            # ========== 步骤 7: 提起 ==========
            print("\n[步骤 7] 提起小球...")
            current_q_deg, current_q_rad = get_current_q(robot)
            configuration.update(current_q_rad)
            q_lift = solve_ik_to_position(configuration, position_task, ball_above_pos, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_lift)
            time.sleep(0.3)
            
            # ========== 步骤 8: 移动到目标位置 ==========
            print(f"\n[步骤 8] 移动到目标位置 {target_pos}...")
            current_q_deg, current_q_rad = get_current_q(robot)
            configuration.update(current_q_rad)
            q_target = solve_ik_to_position(configuration, position_task, target_pos, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_target)
            time.sleep(0.3)
            
            # ========== 步骤 9: 打开夹爪放下 ==========
            print("\n[步骤 9] 打开夹爪放下小球...")
            set_gripper(robot, GRIPPER_OPEN)
            time.sleep(0.3)
            
            # ========== 步骤 10: 回到初始位置 ==========
            print("\n[步骤 10] 回到初始位置...")
            current_q_deg, current_q_rad = get_current_q(robot)
            configuration.update(current_q_rad)
            q_home = solve_ik_to_position(configuration, position_task, INIT_POS, end_effector_frame)
            current_q_deg, current_q_rad = get_current_q(robot)
            move_to_joints(robot, current_q_rad, q_home)
            
            print(f"\n========== 第 {cycle_count} 次抓取完成！ ==========")
            print("等待 1 秒后开始下一次循环... (按 Ctrl+C 退出)")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n正在断开连接...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("已断开连接。")


if __name__ == "__main__":
    main()