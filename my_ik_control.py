import time
import numpy as np
import torch
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pathlib import Path

# --- 引入 LeRobot 的核心硬件接口 ---
# 这行代码价值千金，它帮我们搞定了 calibration
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.so101_follower import SO101FollowerConfig

# ================= 配置 =================
# 你的配置文件名称 (通常在 lerobot/configs/robot/ 下)
# SO-100 的典型配置名，需要设置串口
ROBOT_CONFIG = SO101FollowerConfig(
    port="/dev/ttyACM0",  # 修改为你的实际串口，或使用 /dev/ttyUSB0
) 

# IK 目标 (米)
TARGET_POS = np.array([0.25, 0.0, 0.02]) 

def main():
    # 1. 初始化机器人 (这会自动连接电机、读取校准文件!)
    # 这一步会根据配置自动找到端口，不需要你手动写 /dev/ttyACM0
    print("正在连接机器人并加载校准数据...")
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()
    
    print("机器人连接成功！电机状态已读取。")

    # 2. 准备 IK 求解器 (Pink)
    # 我们需要加载 URDF。需要从 SO-ARM100 仓库下载:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path = Path("SO101/so101_new_calib.urdf")
    
    if not urdf_path.exists():
        print(f"错误: 找不到 URDF 文件: {urdf_path}")
        print("请从以下地址下载 URDF 文件:")
        print("https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf")
        print(f"并放置到: {urdf_path.absolute()}")
        return

    # === [关键修改] 使用 Pinocchio 加载模型，不再使用 pink.models ===
    print("正在加载 Pinocchio 模型...")
    # package_dirs=["."] 告诉 pinocchio 在当前目录下寻找 mesh 文件
    pin_robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dirs=["."])
    
    # 创建 Pink 配置对象 (这是最新版 Pink 的标准用法)
    # q0 是初始姿态
    configuration = pink.Configuration(pin_robot.model, pin_robot.data, pin_robot.q0)
    # ==========================================================
    
    # 3. 获取当前关节角度 (从硬件读取)
    # 使用 get_observation() 读取当前状态
    obs = robot.get_observation()
    # 返回的是字典格式 {"shoulder_pan.pos": val, ...}
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    current_q = np.array([obs[f"{name}.pos"] for name in motor_names])
    print(f"当前关节角度 (度): {current_q}")
    
    # 转换为弧度 (如果配置的是 use_degrees=True)
    current_q_rad = np.deg2rad(current_q)
    
    # ⚠️ 注意: LeRobot 的关节顺序必须和 URDF 一致
    # SO-101 URDF 通常包含夹爪(6个关节), Pinocchio 读取的也是6个
    # 如果报错维度不对，可能需要在这里手动拼接/切片
    
    # 更新 Pink 模型到当前硬件的姿态
    configuration.update(current_q_rad)

    # 4. 定义 IK 任务
    # SO-101 是 5-DOF 机械臂，无法同时满足位置和完整姿态
    # 策略：只约束位置 + 部分姿态（pitch），放弃 yaw（绕z轴旋转）
    end_effector_frame = "gripper_frame_link"  # URDF 中的末端坐标系名称
    
    # 位置任务 - 高权重
    position_task = FrameTask(
        end_effector_frame,
        position_cost=1.0,      # 位置权重高
        orientation_cost=[0.1, 0.1, 0.0],  # [roll, pitch, yaw] - 牺牲 yaw (绕z轴)
    )
    position_task.lm_damping = 1e-4
    
    # 设定目标: 位置 TARGET_POS, 姿态 竖直向下 (pitch = pi)
    target_rotation = pin.utils.rpyToMatrix(0, np.pi, 0) 
    target_pose = pin.SE3(target_rotation, TARGET_POS)
    position_task.set_target(target_pose)
    
    # 5. 迭代求解 IK (多次迭代以收敛)
    print(f"开始求解 IK 到目标位置: {TARGET_POS}")
    
    dt = 0.1
    max_iterations = 200
    for iteration in range(max_iterations):
        velocity = solve_ik(configuration, [position_task], dt=dt, solver="quadprog", safety_break=False)
        configuration.integrate_inplace(velocity, dt)
        
        # 检查收敛
        current_pose = configuration.get_transform_frame_to_world(end_effector_frame)
        pos_error = np.linalg.norm(current_pose.translation - TARGET_POS)
        
        if pos_error < 0.001:  # 1mm 精度
            print(f"IK 在第 {iteration+1} 次迭代收敛")
            break
    
    q_target = configuration.q.copy()
    
    # 计算最终的位置误差
    final_pose = configuration.get_transform_frame_to_world(end_effector_frame)
    pos_error = final_pose.translation - TARGET_POS
    pos_error_norm = np.linalg.norm(pos_error)
    
    print(f"目标关节角 (Rad): {q_target}")
    print(f"目标关节角 (Deg): {np.rad2deg(q_target)}")
    print(f"=== IK 求解结果 ===")
    print(f"目标位置: {TARGET_POS}")
    print(f"IK 解的末端位置: {final_pose.translation}")
    print(f"位置误差 (x, y, z): {pos_error * 1000} mm")
    print(f"位置误差范数: {pos_error_norm * 1000:.3f} mm")

    # 6. 执行移动 (利用 LeRobot 接口)
    print("开始移动...")
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    steps = 100
    for i in range(steps):
        # 简单的线性插值 (弧度)
        interp_q_rad = current_q_rad + (q_target - current_q_rad) * ((i+1)/steps)
        
        # 转换为度 (LeRobot 使用度)
        interp_q_deg = np.rad2deg(interp_q_rad)
        
        # 构建 action 字典
        action = {f"{name}.pos": interp_q_deg[idx] for idx, name in enumerate(motor_names)}
        
        # 发送命令
        robot.send_action(action)
        time.sleep(0.02) # 50Hz

    print("动作完成！到达目标位置。")
    
    # 读取实际到达的关节角度，计算实际末端位置误差
    time.sleep(0.5)  # 等待稳定
    obs_final = robot.get_observation()
    actual_q = np.array([obs_final[f"{name}.pos"] for name in motor_names])
    actual_q_rad = np.deg2rad(actual_q)
    
    # 更新配置到实际位置
    configuration.update(actual_q_rad)
    actual_pose = configuration.get_transform_frame_to_world(end_effector_frame)
    
    actual_pos_error = actual_pose.translation - TARGET_POS
    actual_pos_error_norm = np.linalg.norm(actual_pos_error)
    
    print(f"\n=== 执行后实际误差 ===")
    print(f"目标关节角 (Deg): {np.rad2deg(q_target)}")
    print(f"实际关节角 (Deg): {actual_q}")
    print(f"关节角误差 (Deg): {np.rad2deg(q_target) - actual_q}")
    print(f"目标位置: {TARGET_POS}")
    print(f"实际末端位置: {actual_pose.translation}")
    print(f"位置误差 (x, y, z): {actual_pos_error * 1000} mm")
    print(f"位置误差范数: {actual_pos_error_norm * 1000:.3f} mm")
    
    print("\n按 Ctrl+C 退出并断开连接...")
    
    # 保持位置 - 持续发送最终目标位置
    final_q_deg = np.rad2deg(q_target)
    final_action = {f"{name}.pos": final_q_deg[idx] for idx, name in enumerate(motor_names)}
    
    try:
        while True:
            robot.send_action(final_action)
            time.sleep(0.05)  # 20Hz 保持位置
    except KeyboardInterrupt:
        print("\n正在断开连接...")
        robot.disconnect()
        print("已断开连接。")

if __name__ == "__main__":
    main()