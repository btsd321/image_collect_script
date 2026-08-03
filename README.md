# 图像采集工具

## 脚本列表

### 1. image_capture.py
实时采集 ROS2 相机图像的工具。

**功能**：
- 订阅 ROS2 话题（RGB、Depth、CameraInfo）
- 实时显示图像预览
- 按键 '1' 手动保存图像
- 保存格式：`{idx:04d}_{timestamp}.{ext}`

**使用**：
```bash
python3 image_capture.py \
    --rgb-topic /zed/zed_node/left/color/rect/image \
    --depth-topic /zed/zed_node/depth/depth_registered \
    --camera-info-topic /zed/zed_node/left/color/rect/camera_info \
    --output-dir ./captured_images
```

**输出结构**：
```
captured_images/
├── rgb/              # 0001_20240803_123456_789.png
├── depth/            # 0001_20240803_123456_789.png (uint16, mm)
└── camera_info/      # 0001_20240803_123456_789.json (ROS CameraInfo)
```

---

### 2. image_capture_collector.py
处理 `image_capture.py` 采集的数据，重组织为训练数据集。

**功能**：
- 自动解析文件名，提取时间戳
- 支持感知哈希去重（pHash + 汉明距离）
- 转换深度图格式：uint16(mm) → float32(m) → .npy
- 转换相机信息格式：ROS CameraInfo → train_collector 格式
- 支持仅收集 RGB 模式（默认）

**使用**：
```bash
# 仅收集 RGB（默认，带去重）
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data \
    --deduplicate

# 收集完整数据（RGB + Depth + CameraInfo）
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data \
    --full \
    --deduplicate \
    --hamming-threshold 0.05
```

**输出结构**：
```
training_data/
├── rgb/              # {timestamp}.png
├── depth/            # {timestamp}.npy (float32, m) - 仅 --full 模式
└── camera_info/      # {timestamp}.json - 仅 --full 模式
```

**参数说明**：
- `--input`: 输入目录（image_capture.py 的输出）
- `--output`: 输出目录（训练数据集）
- `--full`: 收集完整数据（RGB + Depth + CameraInfo）；不加则仅收集 RGB
- `--deduplicate`: 启用去重
- `--hamming-threshold`: 汉明距离阈值（默认 0.05，越小越严格）

---

## 典型工作流

### 流程 1: 实时采集 + 去重处理
```bash
# 1. 启动 ROS2 相机节点（另一个终端）
ros2 launch zed_wrapper zed2i.launch.py

# 2. 采集图像（实时预览，按 '1' 保存）
python3 image_capture.py \
    --output-dir ./captured_images

# 3. 处理采集数据（去重 + 重组织）
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data \
    --deduplicate \
    --hamming-threshold 0.05
```

### 流程 2: 仅收集 RGB 用于训练
```bash
# 默认模式，仅收集 RGB
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./rgb_dataset \
    --deduplicate
```

---

## 注意事项

1. **文件命名格式**：`image_capture.py` 生成的文件必须符合 `{idx:04d}_{timestamp}.{ext}` 格式
2. **深度图单位**：输入为 uint16(mm)，输出为 float32(m)
3. **去重算法**：基于感知哈希（pHash），仅对相邻样本去重（内存优化）
4. **默认模式**：不加 `--full` 时仅收集 RGB 图像
5. **完整模式**：使用 `--full` 收集 RGB + Depth + CameraInfo

---

## 与 train_collector.py 的区别

| 特性 | train_collector.py | image_capture_collector.py |
|------|-------------------|---------------------------|
| 输入源 | 感知 pipeline 调试数据 | image_capture.py 采集数据 |
| 目录结构 | 递归搜索 task_info.json | 扁平目录 rgb/depth/camera_info |
| 臂侧检测 | 支持 left/right 分组去重 | 不需要（无臂侧区分） |
| 文件命名 | 从 task_info 提取时间戳 | 从文件名解析时间戳 |
| 默认模式 | 仅 RGB（`--full` 开启完整） | 仅 RGB（`--full` 开启完整） |
| 相机信息 | 从 task_info.json 提取 | 从 ROS CameraInfo JSON 转换 |
