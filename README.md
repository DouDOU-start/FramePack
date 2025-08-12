# FramePack - AI视频生成框架

FramePack是一个强大的AI视频生成框架，可以将静态图像转换为生动的动态视频。这是[原版FramePack](https://lllyasviel.github.io/frame_pack_gitpage/)项目的官方实现和桌面软件。

## 🔗 相关链接
- [**论文**](https://arxiv.org/abs/2504.12626) - 技术细节和原理
- [**项目页面**](https://lllyasviel.github.io/frame_pack_gitpage/) - 官方项目展示
- [**GitHub仓库**](https://github.com/DouDOU-start/FramePack) - 源代码和更新

## 🎯 项目特色

FramePack是一种逐帧预测神经网络结构，可以逐步生成视频。它将输入上下文压缩为固定长度，使生成工作量与视频长度无关，能够使用13B模型处理大量帧，即使在笔记本GPU上也能运行。

**核心理念：视频扩散，但感觉像图像扩散。**

### 主要功能
- 🎬 **AI视频生成**: 将静态图像转换为动态视频
- 🎨 **背景处理**: 集成RMBG-2.0进行背景移除和替换
- 🌟 **透明背景支持**: 完整的Alpha通道处理
- ⚡ **高效处理**: 优化的内存使用和GPU加速

## 💻 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 30XX/40XX/50XX系列，支持fp16和bf16
  - 最低6GB显存（生成短视频）
  - 推荐12GB+显存（更好的性能和更长视频）
  - GTX 10XX/20XX系列未经测试，可能不支持
- **内存**: 推荐16GB+系统内存
- **存储**: 至少10GB可用空间（用于模型和临时文件）

### 软件要求
- **操作系统**: Windows 10/11 或 Linux (Ubuntu 18.04+)
- **Python**: 3.10 (推荐) 或 3.9+
- **Conda**: 推荐安装Anaconda或Miniconda
- **CUDA**: 11.8+ 或 12.x
- **FFmpeg**: 用于视频处理（自动检测，如未安装会提示）

### Conda安装（推荐）
如果您还没有安装Conda，请先安装：
- **Anaconda**: [下载地址](https://www.anaconda.com/products/distribution)
- **Miniconda**: [下载地址](https://docs.conda.io/en/latest/miniconda.html) (轻量版，推荐)

### 性能参考
- 生成1分钟视频(60秒，30fps，1800帧)使用13B模型，最低需要6GB GPU内存
- RTX 4090: 约2-5分钟生成1分钟视频
- RTX 3080: 约5-10分钟生成1分钟视频

## 🚀 快速开始

### 方法1: 使用自动安装脚本（推荐）

```bash
# 运行自动安装脚本
python install.py
```

这个脚本会自动：
- 检查系统要求
- 安装PyTorch和所有依赖
- 验证安装是否成功
- 提供下一步操作指南

### 方法2: 手动安装

#### 1. 环境准备

**推荐使用Conda环境（性能更好，依赖管理更稳定）：**

```bash
# 创建framepack专用环境
conda create -n framepack python=3.10 -y

# 激活环境
conda activate framepack
```

**或者使用Python虚拟环境：**

```bash
# 创建虚拟环境
python -m venv framepack_env

# 激活环境
# Windows:
framepack_env\Scripts\activate
# Linux/macOS:
source framepack_env/bin/activate
```

#### 2. 安装依赖

```bash
# 安装PyTorch (CUDA 12.6版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 或者安装CUDA 11.8版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

#### 3. 启动应用

##### 🎯 统一应用 (推荐)
```bash
# 使用启动脚本 (推荐)
python start_unified.py --server 127.0.0.1 --inbrowser

# 仅FramePack
python start_unified.py --module framepack --server 127.0.0.1 --inbrowser [--auto-load]

# 仅RMBG
python start_unified.py --module rmbg --server 127.0.0.1 --inbrowser [--auto-load]
```

##### 其他
不再提供根目录外的额外入口脚本，统一使用 `start_unified.py`。

#### 4. 命令行参数

所有应用都支持以下参数：
- `--share`: 创建公开分享链接
- `--port PORT`: 指定端口号 (默认: 7860)
- `--server ADDRESS`: 指定服务器地址 (默认: 127.0.0.1)
- `--inbrowser`: 自动在浏览器中打开

**统一应用额外参数:**
- `--module {all, framepack, rmbg}`: 限定展示模块
- `--auto-load`: 启动后自动加载对应模块模型

**使用建议:**
- Windows桌面: `--server 127.0.0.1 --inbrowser`
- Linux服务器: `--server 0.0.0.0 --port 7860`
- 公开分享: `--share`
 



## ✨ 功能特色

### 🎬 FramePack视频生成
- **图像到视频**: 将静态图像转换为动态视频
- **可控生成**: 支持提示词控制视频内容和风格
- **长视频支持**: 可生成任意长度的视频
- **高质量输出**: 支持多种分辨率和帧率
- **内存优化**: 智能内存管理，支持低显存设备

### 🎨 RMBG-2.0背景处理
- **背景移除**: 精确的AI背景分割和移除
- **背景替换**: 支持纯色背景替换
- **视频处理**: 逐帧背景处理，支持透明视频输出
- **批量处理**: 支持多文件批量处理

### 🎯 统一应用平台
- **延迟加载**: 启动时不加载模型，节省内存
- **按需使用**: 根据需要加载对应模型
- **智能切换**: 加载模型后自动切换到对应功能
- **内存管理**: 可随时卸载不使用的模型
- **统一界面**: 一个界面完成所有操作

### 🌟 透明背景支持
本版本添加了完整的透明背景支持：
- 上传并处理带有Alpha通道的PNG图像
- 在整个视频生成过程中完整保留透明通道信息
- 输出带有透明背景的WebM格式视频
- 支持MOV格式的ProRes编码（专业级透明视频）

## 📋 使用说明

### 首次使用
1. 启动应用后，模型会自动从Hugging Face下载
2. 首次下载可能需要较长时间，请耐心等待
3. 模型文件存储在 `./hf_download/` 目录下

### FramePack视频生成流程
1. 上传一张静态图像
2. 输入描述视频内容的提示词
3. 调整生成参数（长度、质量等）
4. 点击开始生成
5. 等待处理完成并下载结果

### RMBG背景处理流程
1. 选择功能：背景移除、颜色替换或视频处理
2. 上传图像或视频文件
3. 设置相关参数
4. 开始处理并下载结果

## ⚠️ 注意事项

### 透明背景相关
- 要正确查看透明背景效果，需要使用支持WebM和VP9编码的播放器（如Chrome、Firefox、Edge等现代浏览器）
- Windows默认的"电影和电视"应用可能无法正确显示透明背景
- 如果系统中没有安装FFmpeg，程序将自动回退到MP4格式（不支持透明背景）
- 处理透明通道需要额外的计算资源，可能会使生成过程略微变慢

### 性能优化
- 首次运行时会下载大量模型文件，请确保网络连接稳定
- 建议在生成过程中不要运行其他GPU密集型程序
- 如果遇到显存不足，可以尝试降低视频分辨率或缩短视频长度
- 定期清理 `outputs/` 目录中的临时文件

### 故障排除
- **模型下载失败**: 检查网络连接，或手动下载模型到指定目录
- **CUDA错误**: 确认GPU驱动和CUDA版本兼容性
- **内存不足**: 关闭其他程序，或使用更小的模型参数
- **FFmpeg未找到**: 安装FFmpeg并确保在系统PATH中

## 📚 文档导航

### 📄 核心文档
- [**项目索引**](PROJECT_INDEX.md) - 完整项目文档和导航
- [**API参考**](API_REFERENCE.md) - 详细API文档和使用示例
- [**开发指南**](DEVELOPMENT_GUIDE.md) - 架构设计和开发规范
- [**RMBG-2.0增强文档**](RMBG-2.0/README_ENHANCED.md) - 背景处理功能详细说明

## 🛠️ 开发和贡献

### 项目结构
```
FramePack/
├── scripts/                    # 主要应用脚本
│   └── unified_app.py          # 统一应用平台（被 start_unified.py 启动）
├── RMBG-2.0/                  # 背景处理核心代码
├── diffusers_helper/          # 扩散模型辅助工具
├── hf_download/               # 模型文件存储目录
├── outputs/                   # 输出文件目录
├── requirements.txt           # 依赖列表
├── install.py                # 自动安装脚本
├── start_unified.py           # 统一应用启动脚本（唯一入口）
├── PROJECT_INDEX.md           # 项目文档索引
├── API_REFERENCE.md           # API参考文档
├── DEVELOPMENT_GUIDE.md       # 开发指南
└── README.md                 # 项目说明
```

### 贡献指南
1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目遵循原版FramePack的许可证。详情请参阅[LICENSE](LICENSE)文件。

## 🙏 致谢

- 感谢原版FramePack项目的作者和贡献者
- 感谢Hugging Face提供的模型托管服务
- 感谢开源社区的支持和贡献

---

**如果这个项目对您有帮助，请给我们一个⭐Star！**
