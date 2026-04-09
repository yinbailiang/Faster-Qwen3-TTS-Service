import io
import json
import os
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import soundfile as sf

from faster_qwen3_tts import FasterQwen3TTS

service_config_path = "./service_config.json"
service_config = json.load(open(service_config_path, "r"))

characters_folder = service_config.get("characters", "./characters")
model_folder = service_config.get("models","./models")
model_name = service_config.get("model", "Qwen3-TTS-12Hz-1.7B-Base")

model_path = f"{model_folder}/{model_name}"

if not Path(model_path).exists():
    print(f"未找到模型 {model_path}, 尝试自动下载 Qwen/{model_name}")
    if os.system(f"hf download Qwen/{model_name} --local-dir {model_path}") != 0:
        print(f"未找到模型 {model_name} 在 {model_path}, 且自动下载 Qwen/{model_name} 失败")
        exit(1)

print(f"加载模型: {model_path}")
model = FasterQwen3TTS.from_pretrained(
    model_path,
)
print(f"模型加载完成")

app = FastAPI(title="Streaming TTS Service (Voice Clone)")

class TTSRequest(BaseModel):
    characters: str = Field(..., description="目标角色名称")
    text: str = Field(..., description="要合成的文本")
    language: str = Field("English", description="语言，如 'Chinese', 'English'")
    chunk_size: int = Field(8, description="每个音频块对应的步数，仅在流式接口中使用，默认为8步")
    
@app.get("/status")
async def health():
    return {"status": "ok"}

@app.get("/characters")
async def list_characters():
    """列出可用的角色"""
    characters = []
    for char_dir in os.listdir(characters_folder):
        char_path = os.path.join(characters_folder, char_dir)
        if os.path.isdir(char_path):
            config_path = os.path.join(char_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                characters.append({
                    "name": config.get("name", char_dir),
                    "description": config.get("description", ""),
                })
    return {"characters": characters}

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """
    流式语音合成接口
    返回原始 PCM 数据（16-bit, 24000 Hz, 单声道），采用 HTTP 分块传输。
    """

    try:
        # 根据角色加载配置
        target_character = request.characters.strip()
        char_dir = os.path.join(characters_folder, target_character)
        if not os.path.isdir(char_dir):
            raise HTTPException(status_code=404, detail=f"Character '{target_character}' not found")
        config_path = os.path.join(char_dir, "config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=500, detail=f"Character config for '{target_character}' not found")
        with open(config_path, "r", encoding="utf-8") as f:
            ref_config = json.load(f)
        
        # 调用流式生成方法，得到生成器
        generator = model.generate_voice_clone_streaming(
            text=request.text,
            language=request.language,
            ref_text=ref_config['ref_text'],
            ref_audio=f"./{characters_folder}/{target_character}/{ref_config['ref_voice']}",
            chunk_size=request.chunk_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")

    def audio_generator():
        """将音频块转换为 PCM 字节并 yield，同时记录性能"""
        for audio_chunk, sr, timing in generator:
            # audio_chunk: numpy.ndarray, float32, shape (samples,)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            chunk_bytes = audio_int16.tobytes()
            yield chunk_bytes

    # 返回 StreamingResponse
    return StreamingResponse(
        audio_generator(),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Bits-Per-Sample": "16",
            "Cache-Control": "no-cache",
        }
    )


@app.post("/tts")
async def tts(request: TTSRequest):
    """非流式语音合成接口，返回完整 WAV 文件"""
    try:
        # 根据角色加载配置
        target_character = request.characters.strip()
        char_dir = os.path.join(characters_folder, target_character)
        if not os.path.isdir(char_dir):
            raise HTTPException(status_code=404, detail=f"Character '{target_character}' not found")
        config_path = os.path.join(char_dir, "config.json")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=500, detail=f"Character config for '{target_character}' not found")
        with open(config_path, "r", encoding="utf-8") as f:
            ref_config = json.load(f)

        # 非流式生成，返回完整音频列表
        audio_list, sr = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            ref_text=ref_config['ref_text'],
            ref_audio=f"./{characters_folder}/{target_character}/{ref_config['ref_voice']}",
        )
        wav = audio_list[0]  # 只有一个样本
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")

    # 将波形保存到内存中的 WAV 文件
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav, sr, format='WAV')
    wav_buffer.seek(0)

    return Response(
        content=wav_buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=service_config.get("port",8001))