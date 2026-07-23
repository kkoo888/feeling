"""
我是进化系统的 LLM 替代 — 读取 prompt 文件，写回复文件。

进化系统通过握手协议发请求，我截取后分析并回复。

用法:
    python3 i_am_the_llm.py

流程:
    1. 监控 pending_prompt.json
    2. 发现新请求 → 打印内容供我分析
    3. 我写回复到 llm_response.json
    4. 进化系统读取回复，继续进化
"""
import json, time, sys
from pathlib import Path

HANDSHAKE_DIR = Path(__file__).resolve().parent / ".openclaw" / "tmp" / "llm_handshake"
PENDING = HANDSHAKE_DIR / "pending_prompt.json"
RESPONSE = HANDSHAKE_DIR / "llm_response.json"

def read_pending():
    if not PENDING.exists():
        return None
    try:
        d = json.loads(PENDING.read_text())
        if d.get("status") == "pending":
            return d
    except:
        pass
    return None

def write_response(text, request_id=None):
    HANDSHAKE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "response": text,
        "request_id": request_id,
        "status": "completed",
    }
    RESPONSE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # 标记 pending 完成
    if PENDING.exists():
        try:
            p = json.loads(PENDING.read_text())
            p["status"] = "completed"
            PENDING.write_text(json.dumps(p, ensure_ascii=False, indent=2))
        except:
            pass
    print(f"[回复已写入] {len(text)} 字符")

def main():
    print("=== 我是进化系统的 LLM ===")
    print(f"监控目录: {HANDSHAKE_DIR}")
    print("等待进化系统的请求...")
    print()

    while True:
        pending = read_pending()
        if pending:
            rid = pending.get("request_id", "?")
            purpose = pending.get("purpose", "?")
            system = pending.get("system", "")
            prompt = pending.get("prompt", "")

            print("=" * 60)
            print(f"[请求] ID={rid} 用途={purpose}")
            print(f"[System] {system[:200]}")
            print(f"[Prompt] {prompt[:500]}")
            print("=" * 60)
            print()

            # 等待我输入回复
            print("请输入回复（输入 END 结束多行输入）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            response = "\n".join(lines)

            write_response(response, rid)
            print()
            print("等待下一个请求...")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
