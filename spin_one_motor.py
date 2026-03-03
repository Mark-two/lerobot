import time
import sys

# 导入
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
except ImportError:
    print("❌ 错误：请先运行 source .venv/bin/activate")
    sys.exit(1)

# --- 配置 ---
PORT = '/dev/ttyACM0'  # 你的串口
MY_ID = 4              # 你的电机 ID

def pack_speed_bit15(speed_val):
    """
    根据文档：BIT15为方向位。
    正数: 0 ~ 32767
    负数: 绝对值 | (1<<15)
    """
    val = int(speed_val)
    if val >= 0:
        return min(val, 32767)
    else:
        return min(abs(val), 32767) | (1 << 15)

def main():
    print(f"--- 终极综合修复测试 (ID={MY_ID}) ---")

    motor_name = "final_fix"
    motors_config = {motor_name: Motor(MY_ID, "sts3215", MotorNormMode.RANGE_M100_100)}
    bus = FeetechMotorsBus(port=PORT, motors=motors_config)
    
    try:
        bus.connect()
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    try:
        # ==========================================
        # 1. 诊断与参数修复
        # ==========================================
        print("\n[Step 1] 读取与修复关键参数...")

        # 修改参数前：解锁 + 关扭矩
        bus.write("Lock", motor_name, 0)
        bus.write("Torque_Enable", motor_name, 0)
        time.sleep(0.1)

        # A. 强制 Mode 1 (速度模式)
        print("   -> 写入 Operating_Mode = 1 (速度闭环)...")
        bus.write("Operating_Mode", motor_name, 1)

        # B. 设置加速度
        print("   -> 写入 Acceleration = 50...")
        bus.write("Acceleration", motor_name, 50)

        # C. 检查 P 参数
        try:
            p_val = bus.read("P_Coefficient", motor_name, normalize=False)
            if p_val == 0:
                print("   ⚠️ 发现 P 值为 0，修复为 128...")
                bus.write("P_Coefficient", motor_name, 128)
        except Exception:
            pass

        # 读回模式确认
        mode_val = bus.read("Operating_Mode", motor_name, normalize=False)
        print(f"   -> 当前 Operating_Mode = {mode_val}")

        # 上锁
        bus.write("Lock", motor_name, 1)
        time.sleep(0.1)

        # ==========================================
        # 2. 运行测试 (BIT15 方向位由库自动处理)
        # ==========================================
        print("\n[Step 2] 运行测试 (目标速度 1500)...")

        # 开扭矩
        bus.write("Torque_Enable", motor_name, 1)

        # 正转
        target = 1500
        packed_val = pack_speed_bit15(target)
        print(f"   -> 发送指令: {target} (Hex: 0x{packed_val:04X})")
        bus.write("Goal_Velocity", motor_name, target)

        print("   -> 正在运行... (5秒)")
        for i in range(5):
            try:
                # 库会自动做符号位解码
                spd = bus.read("Present_Velocity", motor_name, normalize=False)
                print(f"      [{5-i}s] 反馈速度: {spd}")
            except Exception:
                print(f"      [{5-i}s] ...")
            time.sleep(1)

        # 反转
        target = -1500
        packed_val = pack_speed_bit15(target)
        print(f"\n   -> 发送反转: {target} (Hex: 0x{packed_val:04X})")
        bus.write("Goal_Velocity", motor_name, target)
        time.sleep(3)

    finally:
        # 停止 + 卸力
        try:
            bus.write("Goal_Velocity", motor_name, 0)
            bus.write("Torque_Enable", motor_name, 0)
        except Exception:
            pass
        bus.disconnect()
        print("✅ 测试结束。")

if __name__ == "__main__":
    main()