import time
import sys

# 1. 导入库 (根据之前 grep 到的正确路径)
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
except ImportError:
    print("❌ 错误：请先运行 source .venv/bin/activate")
    sys.exit(1)

# --- 配置 ---
PORT = '/dev/ttyACM1'  # 你的串口
MY_ID = 4              # 你的电机 ID

def main():
    print(f"--- 正在切换回【舵机模式】测试 (ID={MY_ID}) ---")

    # 初始化
    motors_config = {
        "servo_test": Motor(MY_ID, "sts3215", MotorNormMode.RANGE_M100_100)
    }
    bus = FeetechMotorsBus(port=PORT, motors=motors_config)
    
    try:
        bus.connect()
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # ==========================================
    # 核心步骤：修改寄存器变回舵机
    # ==========================================
    print("\n[1/3] 正在修改 EPROM 配置...")
    
    try:
        # 1. 解锁
        bus._write(55, 1, MY_ID, 0)
        
        # 2. 恢复最大角度限制 (关键！)
        # 舵机模式必须有范围，STS3215 默认一圈是 0-4095
        print("   -> 设置 Min Limit (Addr 9) = 0")
        bus._write(9, 2, MY_ID, 0)
        
        print("   -> 设置 Max Limit (Addr 11) = 4095")
        bus._write(11, 2, MY_ID, 4095)
        
        # 3. 设置驱动模式为 0 (位置模式)
        print("   -> 设置 Mode (Addr 33) = 0")
        bus._write(33, 1, MY_ID, 0)
        
        # 4. 上锁
        bus._write(55, 1, MY_ID, 1)
        time.sleep(0.1) # 等待保存
        print("   -> 配置完成：现在它是舵机了。")

    except Exception as e:
        print(f"❌ 配置失败: {e}")
        return

    # ==========================================
    # 摆动测试
    # ==========================================
    print("\n[2/3] 开始摆动测试 (Wiggle Test)")
    
    # 开启扭矩
    bus._write(40, 1, MY_ID, 1)
    
    # 定义几个位置点 (0 ~ 4095)
    # 2048 是中间位置
    targets = [2048, 1000, 3000, 2048]
    
    for pos in targets:
        print(f"   -> 目标位置: {pos}")
        # 地址 42: Goal Position
        bus._write(42, 2, MY_ID, pos)
        
        # 等待转动
        time.sleep(1.0)
        
        # 读取当前位置 (地址 56: Present Position)
        try:
            val, _, _ = bus._read(56, 2, MY_ID)
            print(f"      实际位置: {val}")
        except:
            print("      读取位置失败")

    print("\n[3/3] 测试结束")
    # 卸力
    bus._write(40, 1, MY_ID, 0)
    bus.disconnect()

if __name__ == "__main__":
    main()