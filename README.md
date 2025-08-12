# FramePack - 统一演示应用（含 RMBG-2.0 集成）

本仓库提供基于 FramePack 的统一 Gradio 演示应用，集成了：
- 视频生成（FramePack）
- RMBG-2.0 背景抠图（图片/视频，支持纯色或图片背景）
- 视频裁剪（FFmpeg）

## 环境要求
- Windows 10/11 或 Linux
- Python 3.10（推荐）
- NVIDIA CUDA 12.x（或使用 CPU 模式，但速度较慢）
- FFmpeg（视频裁剪/编码需要，放到系统 PATH）

## 安装
```bash
conda create -n framepack python=3.10 -y
conda activate framepack

# 安装依赖（已包含 PyTorch 等必需组件）
pip install -r requirements.txt
```

## 运行
```bash
python demo_gradio.py --server 127.0.0.1 --inbrowser
# 可选：--share / --port 7860
```

### RMBG 模型路径（本地模型）
默认从以下位置解析 RMBG-2.0 模型：
- 环境变量 `RMBG_MODEL_PATH` 指向的本地目录（优先）
- `./hf_download/RMBG-2.0`
- 否则回退 `briaai/RMBG-2.0`（需联网）

Windows 设置示例（CMD）：
```bat
set RMBG_MODEL_PATH=E:\your\local\RMBG-2.0
```

## 功能与界面

应用分为三个 Tab：

### 1) 视频生成
- 上传图片、填写提示词，点击“开始生成”
- “高级设置”内包含步数、总时长、显存预留等参数
- 右侧显示生成视频与采样进度

### 2) RMBG-2.0 背景处理
- 顶部可选择设备（auto/cuda/cpu），切换设备会自动卸载并重载模型
- “加载模型/卸载模型”按钮可手动控制模型生命周期，释放显存
- 子页：
  - 图片抠图/背景替换：
    - 可选“背景颜色”或上传“背景图片”（图片优先）；
    - 背景图片采用 cover 缩放并居中裁剪，不拉伸原图；
  - 视频抠图/背景替换：
    - 同上，同时支持可选的输出宽/高/FPS（默认沿用源视频）；
    - 为空输出透明 WebM；有背景（图片/颜色）输出 MP4。

### 3) 视频裁剪（FFmpeg）
- 支持 center/custom/aspect_ratio 三种裁剪方式
- 提供裁剪预览与导出，采用 FFmpeg 编码（已处理 Windows 编码问题与 `-y` 覆盖）

## 常见问题
- 浏览器控制台 manifest.json / 字体 404：可忽略，不影响功能
- h11/Content-Length/CancelledError：多为刷新/断开导致的连接日志
- FFmpeg 未检测到：请安装并加入 PATH
- CUDA 不可用：确认驱动、CUDA 与 PyTorch 匹配
- dtype 错误（float vs half）：已统一使用 FP32 推理，避免算子不兼容
- Windows 子进程 GBK 解码报错：已统一使用 `encoding='utf-8'`

## 目录结构（简要）
```
FramePack/
├── demo_gradio.py                # 统一 Gradio 应用入口
├── RMBG-2.0/                    # RMBG 核心实现
│   ├── background_processor.py
│   ├── processors/
│   ├── core/
│   └── video_cropper.py
├── diffusers_helper/            # FramePack 相关辅助模块
├── hf_download/                 # 模型下载/本地缓存
├── outputs/                     # 结果输出
└── requirements.txt
```

## 许可证与致谢
- 遵循原版 FramePack 项目的许可证，详见 `LICENSE`
- 感谢原作者与开源社区的贡献

---
如果这个项目对你有帮助，欢迎点个 ⭐️ Star 支持！
