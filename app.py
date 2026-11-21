import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# 로컬: .env 파일을 읽음
# Render: .env 파일이 없으므로 대시보드 환경변수를 그대로 사용
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Render 환경변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = "당신은 친절한 도우미입니다. 한국어로 간결하게 답변합니다."


def to_text(content):
    """Gradio가 전달하는 content를 문자열로 변환 (문자열 또는 파트 리스트 모두 대응)"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        messages.append({"role": item["role"], "content": to_text(item["content"])})
    messages.append({"role": "user", "content": message})

    # 스트리밍: 답변이 한 글자씩 표시됨
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )
    answer = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            yield answer


demo = gr.ChatInterface(
    fn=chat,
    title="ABC 챗봇",
    description=f"OpenAI `{MODEL}` 모델을 사용하는 Gradio 챗봇입니다.",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",                       # 외부 접속 허용 (Render 필수)
        server_port=int(os.getenv("PORT", 7860)),    # Render가 지정한 포트 사용
        share=True
    )