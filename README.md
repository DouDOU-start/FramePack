# FramePack

FramePack是一个强大的AI视频生成框架，可以将静态图像转换为生动的动态视频。这是[原版FramePack](https://lllyasviel.github.io/frame_pack_gitpage/)项目的官方实现和桌面软件。

链接: [**论文**](https://arxiv.org/abs/2504.12626), [**项目页面**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack是一种逐帧预测神经网络结构，可以逐步生成视频。它将输入上下文压缩为固定长度，使生成工作量与视频长度无关，能够使用13B模型处理大量帧，即使在笔记本GPU上也能运行。

**视频扩散，但感觉像图像扩散。**

## 系统要求

* NVIDIA GPU，支持fp16和bf16的RTX 30XX、40XX、50XX系列。GTX 10XX/20XX系列未经测试。
* Linux或Windows操作系统。
* 至少6GB GPU内存。

生成1分钟视频(60秒，30fps，1800帧)使用13B模型，最低需要6GB GPU内存。

## 安装与运行

推荐使用独立的Python 3.10环境：

```bash
# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装依赖
pip install -r requirements.txt
```

### 启动WebUI

```bash
python demo_gradio.py
```

支持的参数：
- `--share`：公开分享
- `--port`：指定端口
- `--server`：指定服务器地址
- `--inbrowser`：自动在浏览器中打开

Windows桌面推荐使用：`python demo_gradio.py --server 127.0.0.1 --inbrowser`
Linux服务器推荐使用：`python demo_gradio.py --server 127.0.0.1`

## 透明背景支持

本版本添加了对透明背景的支持，可以上传并处理带有Alpha通道的PNG图像，在整个视频生成过程中完整保留透明通道信息，并输出带有透明背景的WebM格式视频。

详细使用说明请参考[透明背景功能使用指南](docs/transparent_background.md)。

## 注意事项

- 要正确查看透明背景效果，需要使用支持WebM和VP9编码的播放器（如Chrome、Firefox、Edge等现代浏览器）
- Windows默认的"电影和电视"应用可能无法正确显示透明背景
- 如果系统中没有安装FFMPEG，程序将自动回退到MP4格式（不支持透明背景）
- 处理透明通道需要额外的计算资源，可能会使生成过程略微变慢

## 许可证

本项目遵循原版FramePack的许可证。详情请参阅[LICENSE](LICENSE)文件。
