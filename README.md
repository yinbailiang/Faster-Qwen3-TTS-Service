# Faster-Qwen3-TTS-Service

一个基于 Qwen3-Audio 和 FastAPI 的语音克隆合成服务，支持流式和非流式语音合成。

## 功能特性

- 🎙️ **语音克隆**: 基于参考音频进行语音克隆
- 🚀 **流式合成**: 支持实时流式语音输出，降低延迟
- 🎭 **多角色支持**: 支持多个预设角色，可轻松添加自定义角色
- 🌐 **RESTful API**: 提供简洁的 HTTP 接口
- 🤖 **模型自动下载**: 首次运行自动从 HuggingFace 下载模型
- 📦 **多种模型支持**: 支持 0.6B 和 1.7B 等不同规模的模型

## 项目结构

```text
Faster-Qwen3-TTS-Service/
├── characters/           # 角色配置目录
│   ├── 空空儿/
│   │   ├── config.json  # 角色配置文件
│   │   └── 空空儿.wav   # 参考音频
│   └── ...
├── models/               # 模型目录
│   ├── Qwen3-TTS-12Hz-0.6B-Base/
│   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   └── ...
├── src/
│   └── main.py          # 主程序
├── service_config.json  # 服务配置文件
└── pyproject.toml       # 项目依赖配置
```

## 安装

### 环境要求

- Python 3.12+
- CUDA (推荐，用于 GPU 加速)

### 安装依赖

```bash
pip install -e .
```

## 配置

### 服务配置 (service_config.json)

```json
{
    "models": "./models",
    "module": "Qwen3-TTS-12Hz-1.7B-Base",
    "characters": "./characters",
    "port": 8001
}
```

配置说明：

- models: 模型存储目录
- module: 使用的模型名称（支持 0.6B 和 1.7B 版本）
- characters: 角色配置目录
- port: 服务端口

### 添加新角色

1. 在 characters 目录下创建新角色文件夹
2. 添加参考音频文件（如 oice.wav）
3. 创建 config.json 配置文件：

```json
{
    "ref_voice": "voice.wav",
    "ref_text": "参考音频对应的文本内容"
}
```

## 使用方法

### 启动服务

```bash
cd src
python main.py
```

服务将在 <http://localhost:8001> 启动。

### API 接口

#### 1. 健康检查

```http
GET /status
```

**响应:**

```json
{
    "status": "ok"
}
```

#### 2. 列出可用角色

```http
GET /characters
```

**响应:**

```json
{
    "characters": [
        {
            "name": "空空儿",
            "description": ""
        }
    ]
}
```

#### 3. 非流式语音合成

```http
POST /tts
Content-Type: application/json

{
    "characters": "空空儿",
    "text": "你好，欢迎使用语音合成服务。",
    "language": "Chinese"
}
```

**参数说明:**

- characters: 角色名称（必填）
- Text: 要合成的文本（必填）
- language: 语言（可选，默认 "English"，支持 "Chinese", "English" 等）

**响应:**

- Content-Type: audio/wav
- 返回 WAV 格式的音频文件

#### 4. 流式语音合成

```http
POST /tts/stream
Content-Type: application/json

{
    "characters": "空空儿",
    "text": "你好，欢迎使用语音合成服务。",
    "language": "Chinese",
    "chunk_size": 8
}
```

**参数说明:**

- characters: 角色名称（必填）
- Text: 要合成的文本（必填）
- language: 语言（可选，默认 "English"）
- chunk_size: 每个音频块对应的步数（可选，默认 8）

**响应:**

- Content-Type: audio/raw
- 音频格式: 16-bit PCM, 24000 Hz, 单声道
- 使用 HTTP 分块传输（Chunked Transfer Encoding）

**响应头:**

```text
X-Sample-Rate: 24000
X-Channels: 1
X-Bits-Per-Sample: 16
Cache-Control: no-cache
```

## 使用示例

### Python 示例

```python
import requests

# 非流式合成
response = requests.post('http://localhost:8001/tts', json={
    'characters': '空空儿',
    'text': '你好，这是一个测试。',
    'language': 'Chinese'
})

with open('output.wav', 'wb') as f:
    f.write(response.content)

# 流式合成
response = requests.post('http://localhost:8001/tts/stream', json={
    'characters': '空空儿',
    'text': '你好，这是一个流式测试。',
    'language': 'Chinese',
    'chunk_size': 8
}, stream=True)

with open('output_stream.pcm', 'wb') as f:
    for chunk in response.iter_content(chunk_size=4096):
        f.write(chunk)
```

### cURL 示例

```bash
# 非流式合成
curl -X POST "http://localhost:8001/tts" \
     -H "Content-Type: application/json" \
     -d '{"characters": "空空儿", "text": "你好", "language": "Chinese"}' \
     --output output.wav

# 流式合成
curl -X POST "http://localhost:8001/tts/stream" \
     -H "Content-Type: application/json" \
     -d '{"characters": "空空儿", "text": "你好", "language": "Chinese"}' \
     --output output.pcm
```

## 支持的模型

- Qwen3-TTS-12Hz-0.6B-Base: 0.6B 参数基础模型
- Qwen3-TTS-12Hz-0.6B-CustomVoice: 0.6B 参数自定义语音模型
- Qwen3-TTS-12Hz-1.7B-Base: 1.7B 参数基础模型（默认）
- Qwen3-TTS-12Hz-1.7B-CustomVoice: 1.7B 参数自定义语音模型
- Qwen3-TTS-12Hz-1.7B-VoiceDesign: 1.7B 参数语音设计模型

## 技术栈

- [FastAPI](https://fastapi.tiangolo.com/): Web 框架
- [faster_qwen3_tts](https://github.com/QwenLM/Qwen3-Audio): Qwen3 TTS 加速库
- [PyTorch](https://pytorch.org/): 深度学习框架

## 许可证

[查看 LICENSE 文件](LICENSE)
