import time
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


model_path = '/home/kang/Documents/yolo/runs/detect/runs/detect/cat_ball3_train2/weights/best.pt'

def main():

    # 在 main 函数的最开头（pipeline 之前）加入：
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            print("发现设备，正在重置...")
            dev.hardware_reset()
        
        # 重置后，设备会断开连接再重连，必须强制等待几秒
        print("等待设备重连中 (5秒)...")
        time.sleep(5)


    # 1. 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置流：分辨率可以根据你的训练数据调整，常用 640x480 或 1280x720
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 配置流：彩色
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动管道
    profile = pipeline.start(config)

    # 获取深度传感器的深度标度（Depth Scale）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    # 创建对齐对象（rs.align），将深度流对齐到彩色流
    # 这是获取准确坐标的关键！
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 2. 加载 YOLO 模型
    # 将路径替换为你训练好的 .pt 文件路径
    print("Loading YOLO model...")
    model = YOLO(model_path)  # 示例使用官方权重，请换成你的 "best.pt"
    
    # 定义你关心的类别ID (根据你训练的yaml文件)
    # 假设: 15: cat, 32: sports ball (这是COCO数据集的ID，请根据你自己的数据集修改)
    target_classes = [15, 32] 

    try:
        while True:
            # 3. 获取并对齐帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 获取相机内参（用于2D转3D）
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

            # 将图像转换为 numpy 数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. YOLO 推理
            results = model(color_image, stream=True, verbose=False, conf=0.3)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 获取类别 ID
                    cls_id = int(box.cls[0])
                    
                    # 这里的逻辑需要根据你自己的模型类别ID进行过滤
                    # if cls_id not in target_classes: continue 

                    # 获取边界框坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 计算中心点坐标 (u, v)
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)

                    # 边界检查，防止越界
                    if u >= 640 or v >= 480: continue

                    # 5. 获取深度距离
                    # get_distance 返回的是米(meters)
                    dist = aligned_depth_frame.get_distance(u, v)

                    # 如果距离有效（大于0）
                    if dist > 0:
                        # 6. 核心：反投影 (Deprojection) -> Pixel to Point
                        # point_3d 是一个列表 [x, y, z]，单位是米
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], dist)
                        
                        x_real, y_real, z_real = point_3d
                        
                        # 格式化文本
                        coord_text = f"({x_real:.2f}, {y_real:.2f}, {z_real:.2f})m"
                        
                        # 在图像上绘制
                        label = f"{model.names[cls_id]} {coord_text}"
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        cv2.putText(color_image, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 这里你可以添加逻辑将 x_real, y_real, z_real 发送给机器人或保存

            # 显示图像
            cv2.imshow('RealSense YOLO 3D', color_image)
            key = cv2.waitKey(1)
            # 按 'q' 或 ESC 退出
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()