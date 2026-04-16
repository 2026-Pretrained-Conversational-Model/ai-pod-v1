#!/bin/bash
# ============================================================
# run.sh — RunPod에서 start_server.py를 백그라운드로 실행
#
# 사용법:
#   bash run.sh          # 백그라운드로 시작
#   bash run.sh stop     # 종료
#   bash run.sh status   # 상태 확인
#   bash run.sh logs     # 로그 실시간 확인
# ============================================================

REPO_DIR="${REPO_DIR:-/workspace/ai-orchestrator}"
LOG_FILE="/workspace/logs/server.log"
PID_FILE="/workspace/server.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /workspace/logs

case "${1:-start}" in

# ── start ────────────────────────────────────────────────────
start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "[INFO] 이미 실행 중 (PID=$(cat $PID_FILE))"
        exit 0
    fi

    echo "[INFO] 서버 시작 중..."
    echo "[INFO] 로그: $LOG_FILE"
    echo "[INFO] 모델 로드에 수 분 걸릴 수 있습니다."

    # REPO_DIR을 PYTHONPATH에 추가하여 실행
    REPO_DIR="$REPO_DIR" \
    PYTHONPATH="$REPO_DIR:$PYTHONPATH" \
    nohup python "$SCRIPT_DIR/start_server.py" \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "[INFO] PID=$(cat $PID_FILE) — 백그라운드 실행 시작"
    echo ""
    echo "  로그 확인:    bash run.sh logs"
    echo "  상태 확인:    bash run.sh status"
    echo "  서버 헬스:    curl http://localhost:8000/health"
    ;;

# ── stop ─────────────────────────────────────────────────────
stop)
    if [ ! -f "$PID_FILE" ]; then
        echo "[INFO] PID 파일 없음. 실행 중이 아닐 수 있습니다."
        exit 0
    fi
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] 종료 중 (PID=$PID)..."
        kill "$PID"
        sleep 2
        kill -9 "$PID" 2>/dev/null || true
        echo "[INFO] 종료 완료"
    else
        echo "[INFO] 이미 종료된 프로세스입니다."
    fi
    rm -f "$PID_FILE"
    ;;

# ── status ───────────────────────────────────────────────────
status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "[INFO] 실행 중 (PID=$(cat $PID_FILE))"
        curl -s http://localhost:8000/health && echo "" || echo "[WARN] /health 응답 없음 (아직 로딩 중일 수 있음)"
    else
        echo "[INFO] 실행 중이 아닙니다."
    fi
    ;;

# ── logs ─────────────────────────────────────────────────────
logs)
    tail -f "$LOG_FILE"
    ;;

*)
    echo "사용법: bash run.sh [start|stop|status|logs]"
    ;;
esac
