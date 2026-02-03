from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# 1. 定义你要合并的数据集列表
source_repo_ids = [
    "mark-two/test",
    "mark-two/testpart1",
    "mark-two/testpart2"
]

# 2. 定义新数据集的名称
target_repo_id = "mark-two/merged_dataset_all"
root_dir = Path("/home/kang/.cache/huggingface/lerobot")  # 数据集存储路径

# Clean up existing target dataset if it exists
import shutil
target_path = root_dir / target_repo_id
if target_path.exists():
    print(f"Removing existing target dataset at {target_path}")
    shutil.rmtree(target_path)

# 3. 创建目标数据集（使用第一个源数据集的配置作为模板）
# 先加载第一个数据集来获取 fps, features 等信息
first_ds = LeRobotDataset(source_repo_ids[0], root=root_dir / source_repo_ids[0])
features = first_ds.features

# Fix image shapes if they are (H, W, C) to (C, H, W)
for key, ft in features.items():
    if ft["dtype"] in ["image", "video"]:
        if len(ft["shape"]) == 3 and ft["shape"][2] == 3:
             # It seems to be (H, W, C), convert to (C, H, W)
             print(f"Converting shape for {key} from {ft['shape']} to {(3, ft['shape'][0], ft['shape'][1])}")
             features[key]["shape"] = (3, ft["shape"][0], ft["shape"][1])

target_ds = LeRobotDataset.create(
    repo_id=target_repo_id,
    fps=first_ds.fps,
    #root=root_dir,
    robot_type=first_ds.meta.robot_type,
    features=features,
    use_videos=True, # 假设都使用视频
)

print(f"Created target dataset: {target_repo_id}")

# 4. 遍历所有源数据集并合并
for repo_id in source_repo_ids:
    print(f"Merging {repo_id}...")
    source_ds = LeRobotDataset(repo_id, root=root_dir / repo_id)
    
    # 遍历源数据集中的每一集
    for episode_idx in range(source_ds.num_episodes):
        # 获取该集的帧范围
        ep_info = source_ds.meta.episodes[episode_idx]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]
        
        # 逐帧加载并添加到目标数据集
        for i in range(from_idx, to_idx):
            frame = source_ds[i]
            
            # 构造传给 add_frame 的字典
            # 保留 features 中的键，以及 task 和 timestamp
            # Exclude internal keys that are managed by add_frame/dataset
            ignore_keys = {"index", "episode_index", "frame_index", "task_index"}
            new_frame = {k: frame[k] for k in target_ds.features if k in frame and k not in ignore_keys}
            new_frame["task"] = frame["task"]
            if "timestamp" in frame:
                new_frame["timestamp"] = frame["timestamp"]
            
            target_ds.add_frame(new_frame)
        
        # 保存这一集
        target_ds.save_episode()

print("Merging complete. Finalizing...")
target_ds.finalize()

# 5. 上传到 Hugging Face (可选)
# target_ds.push_to_hub()
print("Done!")