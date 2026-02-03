import time
import numpy as np
import torch
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pathlib import Path

# --- 引入 LeRobot 的核心硬件接口 ---
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

# ================= 配置 =================
ROBOT_CONFIG = SO101FollowerConfig(
    port="/dev/ttyACM0",
) 

# 任务位置 (米)
INIT_POS = np.array([0.25, 0.0, 0.1])      # 初始位置
BALL_POS = np.array([0.25, 0.0, 0.01])     # 小球位置
TARGET_POS = np.array([0.2, -0.1, 0.02])    # 目标位置

# 夹爪控制 (度数，取决于你的校准)
GRIPPER_OPEN = 50.0    # 打开夹爪
GRIPPER_CLOSE = 0.0   # 闭合夹爪

# 电机名称
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def solve_ik_to_position(configuration, position_task, target_pos, end_effector_frame, max_iterations=200, tolerance=0.001):
    """求解 IK 到目标位置"""
    target_rotation = pin.utils.rpyToMatrix(0, np.pi, 0)  # 夹爪朝下
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


def print_position_error(configuration, end_effector_frame, target_pos, label=""):
    """打印位置误差"""
    actual_pose = configuration.get_transform_frame_to_world(end_effector_frame)
    pos_error = actual_pose.translation - target_pos
    pos_error_norm = np.linalg.norm(pos_error)
    print(f"  {label} 目标: {target_pos}, 实际: {actual_pose.translation}")
    print(f"  {label} 位置误差: {pos_error*1000} mm, 范数: {pos_error_norm*1000:.2f} mm")


def main():
    # 1. 初始化机器人
    print("正在连接机器人...")
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()
    print("机器人连接成功！")

    # 2. 加载 URDF
    urdf_path = Path("SO101/so101_new_calib.urdf")
    if not urdf_path.exists():
        print(f"错误: 找不到 URDF 文件: {urdf_path}")
        return

    print("正在加载 Pinocchio 模型...")
    pin_robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dirs=["."])
    configuration = pink.Configuration(pin_robot.model, pin_robot.data, pin_robot.q0)
    
    end_effector_frame = "gripper_frame_link"
    
    # 定义 IK 任务
    position_task = FrameTask(
        end_effector_frame,
        position_cost=1.0,
        orientation_cost=[0.1, 0.1, 0.0],  # 牺牲 yaw
    )
    position_task.lm_damping = 1e-4

    # 获取初始关节角度
    current_q_deg, current_q_rad = get_current_q(robot)
    print(f"当前关节角度 (度): {current_q_deg}")
    configuration.update(current_q_rad)

    try:
        # ========== 步骤 1: 到初始位置 ==========
        print("\n[步骤 1] 移动到初始位置...")
        q_init = solve_ik_to_position(configuration, position_task, INIT_POS, end_effector_frame)
        current_q_deg, current_q_rad = get_current_q(robot)
        move_to_joints(robot, current_q_rad, q_init)
        time.sleep(0.3)
        
        # 更新当前位置并打印误差
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        print_position_error(configuration, end_effector_frame, INIT_POS, "初始位置")

        # ========== 步骤 2: 打开夹爪 ==========
        print("\n[步骤 2] 打开夹爪...")
        set_gripper(robot, GRIPPER_OPEN)
        time.sleep(0.3)

        # ========== 步骤 3: 到小球位置 ==========
        print("\n[步骤 3] 移动到小球位置...")
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        q_ball = solve_ik_to_position(configuration, position_task, BALL_POS, end_effector_frame)
        current_q_deg, current_q_rad = get_current_q(robot)
        move_to_joints(robot, current_q_rad, q_ball, steps=150)  # 慢一点
        time.sleep(0.3)
        
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        print_position_error(configuration, end_effector_frame, BALL_POS, "小球位置")

        # ========== 步骤 4: 闭合夹爪 (抓取) ==========
        print("\n[步骤 4] 闭合夹爪抓取小球...")
        set_gripper(robot, GRIPPER_CLOSE, duration=0.8)
        time.sleep(0.5)

        # ========== 步骤 5: 到目标位置 ==========
        print("\n[步骤 5] 移动到目标位置...")
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        q_target = solve_ik_to_position(configuration, position_task, TARGET_POS, end_effector_frame)
        current_q_deg, current_q_rad = get_current_q(robot)
        move_to_joints(robot, current_q_rad, q_target)
        time.sleep(0.3)
        
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        print_position_error(configuration, end_effector_frame, TARGET_POS, "目标位置")

        # ========== 步骤 6: 打开夹爪 (放下) ==========
        print("\n[步骤 6] 打开夹爪放下小球...")
        set_gripper(robot, GRIPPER_OPEN)
        time.sleep(0.3)

        # ========== 步骤 7: 回到初始位置 ==========
        print("\n[步骤 7] 回到初始位置...")
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        q_home = solve_ik_to_position(configuration, position_task, INIT_POS, end_effector_frame)
        current_q_deg, current_q_rad = get_current_q(robot)
        move_to_joints(robot, current_q_rad, q_home)
        time.sleep(0.3)
        
        current_q_deg, current_q_rad = get_current_q(robot)
        configuration.update(current_q_rad)
        print_position_error(configuration, end_effector_frame, INIT_POS, "返回初始位置")

        print("\n========== 抓取任务完成！ ==========")
        print("按 Ctrl+C 退出...")
        
        # 保持位置
        final_q_deg, _ = get_current_q(robot)
        final_action = {f"{name}.pos": final_q_deg[idx] for idx, name in enumerate(MOTOR_NAMES)}
        while True:
            robot.send_action(final_action)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n正在断开连接...")
        robot.disconnect()
        print("已断开连接。")

if __name__ == "__main__":
    main()