import time
import sys

# 导入
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


# --- 配置 ---
PORT = '/dev/ttyACM0'       # 你的串口
MOTOR_IDS = [1, 2, 3]       # 要测试的3个电机ID

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

def main() -> bool:
    print(f"--- 3电机联合测试 (ID={MOTOR_IDS}) ---")

    # 1. 批量定义电机配置
    motors_config = {}
    for mid in MOTOR_IDS:
        # 给每个电机起个名字，比如 "motor_1", "motor_2"
        motors_config[f"motor_{mid}"] = Motor(mid, "sts3215", MotorNormMode.RANGE_M100_100)
    
    bus = FeetechMotorsBus(port=PORT, motors=motors_config)
    
    try:
        bus.connect()
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    interrupted = False

    try:
        # ==========================================
        # 1. 批量诊断与初始化
        # ==========================================
        print("\n[Step 1] 正在初始化所有电机...")
        
        for name in motors_config:
            mid = motors_config[name].id
            print(f"   🔧 配置电机 ID: {mid} ({name})...")
            
            # 解锁 + 关扭矩
            bus.write("Lock", name, 0)
            bus.write("Torque_Enable", name, 0)
            
            # --- 关键修复步骤 ---
            # A. 强制 Mode 1
            bus.write("Operating_Mode", name, 1)
            
            # B. ⚠️【重要】强制设置速度限制 (地址46) 
            # 防止新电机的限速为0导致不转。虽然库里可能没映射这个名字，我们用底层写比较稳
            # 如果你在库里加了映射，可以用 bus.write("Running_Speed_Limit", name, 3400)
            # 这里为了保险，直接用底层 _write 写地址 46
            try:
                bus._write(46, 2, mid, 3400) 
            except:
                pass

            # C. 设置加速度
            bus.write("Acceleration", name, 50)
            
            # D. 清除限位 (防止当作舵机卡住)
            bus._write(9, 2, mid, 0)
            bus._write(11, 2, mid, 0)

            # E. 检查 P 参数
            try:
                p_val = bus.read("P_Coefficient", name, normalize=False)
                if p_val == 0:
                    print(f"      ⚠️ (ID {mid}) P值为0，修复为 128...")
                    bus.write("P_Coefficient", name, 128)
            except:
                pass
            
            # 上锁
            bus.write("Lock", name, 1)

        time.sleep(0.2) # 等待写入生效

        # ==========================================
        # 2. 运行测试 (正转)
        # ==========================================
        print("\n[Step 2] 全体正转 (目标速度 1500)...")

        # 批量开扭矩
        for name in motors_config:
            bus.write("Torque_Enable", name, 1)
        
        # 批量发指令
        target = 1500
        # 如果你已经修改了库让 Goal_Velocity 映射到 46，且库没处理 Bit15，
        # 那么这里可能需要传 packed_val。
        # 这里假设你的库像上个脚本一样，直接传 target 即可（库内处理或直接写入）
        # 为了稳妥，我们打印一下实际值
        packed_val = pack_speed_bit15(target) 
        
        for name in motors_config:
            # 注意：如果你库里 Goal_Velocity 映射的是 46，请确保传入的值符合你的库要求
            # 如果你的库没有自动处理 Bit15，请用 bus.write("Goal_Velocity", name, packed_val)
            bus.write("Goal_Velocity", name, target)
        
        print("   -> 正在运行... (5秒)")
        for i in range(5):
            print(f"   [{5-i}s] 速度反馈:")
            for name in motors_config:
                try:
                    spd = bus.read("Present_Velocity", name, normalize=False)
                    print(f"      {name}: {spd}", end="  ")
                except:
                    print(f"      {name}: Err", end="  ")
            print("") # 换行
            time.sleep(1)

        # ==========================================
        # 3. 运行测试 (反转)
        # ==========================================
        print("\n[Step 3] 全体反转 (目标速度 -1500)...")
        
        target = -1500
        for name in motors_config:
            bus.write("Goal_Velocity", name, target)
            
        time.sleep(3)
        
        # 打印一次最终速度
        print("   -> 最终速度快照:")
        for name in motors_config:
             try:
                spd = bus.read("Present_Velocity", name, normalize=False)
                print(f"      {name}: {spd}")
             except: pass

    except KeyboardInterrupt:
        print("\n🛑 用户中断...")
        interrupted = True

    finally:
        print("\n[Step 4] 停止与卸力...")
        for name in motors_config:
            try:
                bus.write("Goal_Velocity", name, 0)
                bus.write("Torque_Enable", name, 0)
            except Exception:
                pass
        bus.disconnect()
        print("✅ 测试结束。")

    return not interrupted

if __name__ == "__main__":
    round_idx = 1
    while True:
        print(f"\n========== 第 {round_idx} 轮 ==========")
        should_continue = main()
        if not should_continue:
            print("👋 已停止循环。")
            break
        round_idx += 1
        time.sleep(1)