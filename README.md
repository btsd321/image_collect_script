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
# 默认：仅收集 RGB + 去重
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data

# 收集完整数据（RGB + Depth + CameraInfo）
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data \
    --full

# 关闭去重，保留全部样本
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data \
    --no-deduplicate
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
- `--no-deduplicate`: 关闭去重；**默认开启去重**
- `--hamming-threshold`: 汉明距离阈值（默认 0.05，越小越严格）

### 去重说明

去重默认开启，用于剔除连续按键保存、或画面几乎没变化而产生的重复样本：

1. 对每张 RGB 图计算感知哈希（pHash）：灰度 → 缩放 16×16 → DCT → 取左上 8×8 低频 → 按中位数二值化，得到 **64 bit** 哈希。低频 + 中位数阈值使其对亮度变化和轻微噪声不敏感，只对构图变化敏感。
2. 按时间戳排序，逐个比较**相邻**两张的归一化汉明距离。
3. 距离小于阈值判为重复，丢弃旧的、保留新的。

阈值粒度（64 bit）：每 1 bit 差异 ≈ 0.0156。

| 阈值 | 容许不同的 bit 数 | 效果 |
|------|------------------|------|
| 0.05（默认） | ≤ 3 | 保守，只去掉几乎完全相同的帧 |
| 0.1 | ≤ 6 | 中等 |
| 0.2 | ≤ 12 | 激进，构图相似即视为重复 |

注意：只比较相邻帧，因此相隔较远的重复画面（例如中途拍了别处又转回来）不会被去掉。

---

## 典型工作流

### 流程 1: 实时采集 + 去重处理
```bash
# 1. 启动 ROS2 相机节点（另一个终端）
ros2 launch zed_wrapper zed2i.launch.py

# 2. 采集图像（实时预览，按 '1' 保存）
python3 image_capture.py \
    --output-dir ./captured_images

# 3. 处理采集数据（去重 + 重组织，去重默认开启）
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./training_data
```

### 流程 2: 仅收集 RGB 用于训练
```bash
# 默认模式：仅收集 RGB + 去重
python3 image_capture_collector.py \
    --input ./captured_images \
    --output ./rgb_dataset
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
