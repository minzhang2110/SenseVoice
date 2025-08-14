# @Author: Bi Ying
# @Date:   2024-07-10 17:22:55
import shutil
import os
from pathlib import Path
from typing import Union

import torch
import torchaudio
import numpy as np
from funasr import AutoModel
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status


app = FastAPI()

TMP_DIR = "./tmp"

# 确保临时目录存在
os.makedirs(TMP_DIR, exist_ok=True)


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "SenseVoice OpenAI Compatible API Server", 
        "version": "1.0.0",
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}

# Initialize the model - 使用本地缓存路径避免联网下载
# 本地模型路径
local_model_path = "/Users/dulei/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
local_vad_model_path = "/Users/dulei/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 检查本地模型是否存在，如果不存在则使用原来的方式
if os.path.exists(local_model_path) and os.path.exists(local_vad_model_path):
    model = AutoModel(
        model=local_model_path,
        vad_model=local_vad_model_path,
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
        disable_update=True,  # 禁用自动更新检查
    )
    print(f"使用本地模型: {local_model_path}")
else:
    # 如果本地模型不存在，回退到在线下载
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
        disable_update=True,  # 禁用自动更新检查
    )
    print("本地模型不存在，使用在线下载")

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
}


def format_str_v2(text: str, show_emo=True, show_event=True):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")

    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    if show_emo:
        text = text + emo_dict[emo]

    for e in event_dict:
        if sptk_dict[e] > 0 and show_event:
            text = event_dict[e] + text

    for emoji in emo_set.union(event_set):
        text = text.replace(" " + emoji, emoji)
        text = text.replace(emoji + " ", emoji)

    return text.strip()


def format_str_v3(text: str, show_emo=True, show_event=True):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        text = text.replace(lang, "<|lang|>")
    parts = [format_str_v2(part, show_emo, show_event).strip(" ") for part in text.split("<|lang|>")]
    new_s = " " + parts[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(parts)):
        if len(parts[i]) == 0:
            continue
        if get_event(parts[i]) == cur_ent_event and get_event(parts[i]) is not None:
            parts[i] = parts[i][1:]
        cur_ent_event = get_event(parts[i])
        if get_emo(parts[i]) is not None and get_emo(parts[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += parts[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def model_inference(input_wav, language, fs=16000, show_emo=True, show_event=True):
    language = "auto" if len(language) < 1 else language

    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

    if len(input_wav) == 0:
        raise ValueError("The provided audio is empty.")

    merge_vad = True
    text = model.generate(
        input=input_wav,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=0,
        merge_vad=merge_vad,
    )

    text = text[0]["text"]
    text = format_str_v3(text, show_emo, show_event)

    return text


@app.get("/v1/models")
async def models():
    """返回可用模型列表，兼容OpenAI API格式"""
    return {
        "object": "list",
        "data": [
            {
                "id": "iic/SenseVoiceSmall",
                "object": "model",
                "created": 1677610602,
                "owned_by": "sensevoice",
                "root": "iic/SenseVoiceSmall",
                "parent": None,
                "permission": [
                    {
                        "id": "modelperm-123",
                        "object": "model_permission",
                        "created": 1677610602,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: Union[UploadFile, None] = File(default=None), language: str = Form(default="auto")):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request, no file provided")

    # 确保临时目录存在
    os.makedirs(TMP_DIR, exist_ok=True)
    
    filename = file.filename
    fileobj = file.file
    tmp_file = Path(TMP_DIR) / filename

    try:
        with open(tmp_file, "wb+") as upload_file:
            shutil.copyfileobj(fileobj, upload_file)
        
        # 确保音频数据保持为int32格式，并转换为一维数组
        waveform, sample_rate = torchaudio.load(tmp_file)
        waveform = (waveform * np.iinfo(np.int32).max).to(dtype=torch.int32).squeeze()
        if len(waveform.shape) > 1:
            waveform = waveform.float().mean(axis=0)  # 将多通道音频转换为单通道
        input_wav = (sample_rate, waveform.numpy())

        result = model_inference(input_wav=input_wav, language=language, show_emo=False)
        
        return {"text": result}
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                          detail=f"Error processing audio file: {str(e)}")
    finally:
        # 确保临时文件被删除
        if tmp_file.exists():
            tmp_file.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
