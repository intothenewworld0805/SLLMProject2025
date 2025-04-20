from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

# FastAPI 앱 초기화
app = FastAPI()

# CORS 허용 설정 (프론트엔드 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 실제 도메인으로 제한하는 것이 좋음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디바이스 설정: GPU 우선 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로딩 (최초 1회)
MODEL_PATH = "app/google/gemma-2-2b-it"  # 로컬 저장 경로
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(device)  # GPU 또는 CPU로 모델 이동
model.eval()      # 평가 모드

@app.get("/")
def root():
    return {"message": "✅ AI 서버가 정상적으로 실행 중입니다."}

@app.post("/ask")
async def ask(request: Request):
    try:
        # JSON 데이터 수신
        data = await request.json()
        input_text = data.get("question", "").strip()

        # 입력 검증
        if not input_text:
            return {"answer": "❗ 질문을 입력해주세요."}

        # 입력 텐서 생성 및 디바이스 이동
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)

        # 모델로 응답 생성
        output = model.generate(
            **input_ids,
            max_length=300,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
        )

        # 텍스트 디코딩 및 응답
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"answer": answer}

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        return {"answer": "⚠️ 처리 중 오류가 발생했습니다."}

# 로컬 서버 실행 (개발 중일 때만 사용)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
