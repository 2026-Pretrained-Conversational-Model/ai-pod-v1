"""
start_server.py
---------------
RunPod에서 ai-orchestrator FastAPI 서버를 백그라운드로 띄우는 스크립트.
코랩 노트북(colab_smoke_test_v03.ipynb)과 동일한 순서로 동작.

사용법:
    cd /workspace
    git clone https://github.com/2026-Pretrained-Conversational-Model/ai-engine.git ai-orchestrator
    pip install -r ai-orchestrator/requirements.txt
    pip install transformers accelerate bitsandbytes
    python start_server.py

서버 확인:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"session_id":"test-1","user_text":"안녕하세요"}'
"""

import os
import sys
import logging

# ─── 0. 경로 & 기본 로깅 ────────────────────────────────────────────────────────
REPO_DIR = os.environ.get("REPO_DIR", "/workspace/ai-orchestrator")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("start_server")
logger.info("REPO_DIR: %s", REPO_DIR)

# ─── 1. 환경변수 설정 (코랩 셀 2와 동일) ────────────────────────────────────────
os.environ["LLM_BACKEND"]                  = "local"
os.environ["EMBEDDING_MODEL_NAME"]         = "jhgan/ko-sroberta-multitask"
os.environ["EMBEDDING_DIM"]               = "768"
os.environ["EMBEDDING_DEVICE"]            = "cuda"
os.environ["LOCAL_FILE_DIR"]              = "/workspace/orchestrator_files"
os.environ["SESSION_MAX_BYTES"]           = str(20 * 1024 * 1024)
os.environ["MEMORY_UPDATE_EVERY_N_TURNS"] = "3"
os.environ["MEMORY_UPDATE_WINDOW_TURNS"]  = "3"
os.environ["ANSWER_MAX_NEW_TOKENS"]       = "800"

os.makedirs(os.environ["LOCAL_FILE_DIR"], exist_ok=True)
os.makedirs("/workspace/logs", exist_ok=True)
logger.info("환경변수 설정 완료")

# ─── 2. 모델 설정 (코랩 셀 3과 동일) ────────────────────────────────────────────
ANSWER_MODEL_ID          = os.environ.get("ANSWER_MODEL_ID",  "Qwen/Qwen2.5-7B-Instruct")
ROUTER_MODEL_ID          = os.environ.get("ROUTER_MODEL_ID",  "Qwen/Qwen2.5-3B-Instruct")
SUMMARY_MODEL_ID         = os.environ.get("SUMMARY_MODEL_ID", "g34634/qwen2.5-3b-memory-summary-v1")
BASE_QWEN_TOKENIZER_ID   = "Qwen/Qwen2.5-3B-Instruct"
USE_4BIT                 = os.environ.get("USE_4BIT", "false").lower() == "true"

logger.info("ANSWER_MODEL_ID:  %s", ANSWER_MODEL_ID)
logger.info("ROUTER_MODEL_ID:  %s", ROUTER_MODEL_ID)
logger.info("SUMMARY_MODEL_ID: %s", SUMMARY_MODEL_ID)
logger.info("USE_4BIT:         %s", USE_4BIT)

# ─── 3. 모델 로드 (코랩 셀 4와 동일) ────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load(model_id: str | None, tokenizer_id: str | None = None):
    if model_id is None:
        return None, None

    tok_id = tokenizer_id or model_id
    logger.info("loading model=%s", model_id)
    logger.info("loading tokenizer=%s", tok_id)

    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    model.eval()
    device = next(model.parameters()).device
    logger.info("  -> ready on %s", device)
    return model, tok


answer_model,  answer_tok  = _load(ANSWER_MODEL_ID)
router_model,  router_tok  = _load(ROUTER_MODEL_ID)
# summary 모델은 base Qwen 토크나이저 사용 (코랩과 동일)
summary_model, summary_tok = _load(SUMMARY_MODEL_ID, tokenizer_id=BASE_QWEN_TOKENIZER_ID)

logger.info("모델 로드 완료")

# ─── 4. config 재생성 (ANSWER_MAX_NEW_TOKENS 반영) ──────────────────────────────
from app.core.config import settings
settings = settings.__class__()
logger.info("ANSWER_MAX_NEW_TOKENS: %s", settings.ANSWER_MAX_NEW_TOKENS)

# ─── 5. LocalModelRegistry 등록 (코랩 셀 5와 동일) ──────────────────────────────
from app.services.llm.local_registry import LocalModelRegistry

LocalModelRegistry.clear()

if answer_model is not None:
    LocalModelRegistry.register(
        "answer", answer_model, answer_tok,
        device="cuda", max_new_tokens=800,
    )

if router_model is not None:
    LocalModelRegistry.register(
        "router", router_model, router_tok,
        device="cuda", max_new_tokens=60,
    )

# memory role로 등록 (memory_state_generator가 memory → summary 순서로 탐색)
if summary_model is not None:
    LocalModelRegistry.register(
        "memory", summary_model, summary_tok,
        device="cuda", max_new_tokens=400,
    )

logger.info("registered roles: %s", LocalModelRegistry.list_roles())

# ─── 6. 임베딩 모델 워밍업 ──────────────────────────────────────────────────────
from app.services.embedding.embedding_singleton import EmbeddingSingleton
EmbeddingSingleton.warmup()
logger.info("임베딩 모델 워밍업 완료")

# ─── 7. FastAPI 앱 생성 & uvicorn 실행 ──────────────────────────────────────────
# main.py의 lifespan은 EmbeddingSingleton.warmup()을 다시 호출하지만
# 이미 로드된 경우 no-op이므로 안전.
from app.main import app
import uvicorn

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

logger.info("서버 시작: http://%s:%d", HOST, PORT)
logger.info("(Ctrl+C로 종료)")

uvicorn.run(
    app,
    host=HOST,
    port=PORT,
    log_level=LOG_LEVEL,
    # 모델은 이미 이 프로세스에 로드됐으므로 reload 비활성화
    reload=False,
    # 워커 1개 — 모델 메모리 공유 불가
    workers=1,
)
