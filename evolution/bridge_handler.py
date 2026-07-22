# -*- coding: utf-8 -*-
"""
Bridge Handler — 自动处理 LLM 评估调用
=====================================
轮询 .trae_llm_bridge/ 中的 pending 文件:
  - 有 func_spec → 评估调用 → 自动构造 JSON 响应 (从执行输出提取 score)
  - 无 func_spec → 代码生成调用 → 跳过 (由主代理的子代理处理)

用法: python bridge_handler.py
"""
import json
import os
import re
import time
import sys
from pathlib import Path

BRIDGE_DIR = Path(r"c:\Users\CC\Desktop\feeling\.trae_llm_bridge")


def handle_eval_call(pending_path, pending_data):
    """处理评估调用 — 从执行输出提取 fitness score, 构造 JSON 响应"""
    call_id = pending_data.get("call_id", "")
    system_msg = pending_data.get("system_message", "")

    # 从执行输出中提取 score
    score_match = re.search(r"Score:\s*([\d.]+)", system_msg)
    edges_match = re.search(r"Edges:\s*(\d+)", system_msg)
    paths_match = re.search(r"Paths:\s*(\d+)", system_msg)

    # 检查是否有错误
    has_error = "SyntaxError" in system_msg or "Error:" in system_msg or "FitnessError" in system_msg
    has_traceback = "Traceback" in system_msg

    if has_error or has_traceback:
        # 有 bug
        response_dict = {
            "is_bug": True,
            "summary": f"代码执行有错误。Score=0, 需要修复。",
            "metric": None,
            "lower_is_better": False,
        }
    else:
        score = float(score_match.group(1)) if score_match else 0.0
        edges = int(edges_match.group(1)) if edges_match else 0
        paths = int(paths_match.group(1)) if paths_match else 0

        response_dict = {
            "is_bug": False,
            "summary": f"B5 fitness: score={score:.3f}, edges={edges}, paths={paths}. 代码执行成功。",
            "metric": score,
            "lower_is_better": False,
        }

    # 写 response 文件
    resp_path = BRIDGE_DIR / f"response_{call_id}.json"
    resp_data = {
        "call_id": call_id,
        "output": response_dict,
        "content": response_dict,
        "in_tokens": 0,
        "out_tokens": 0,
        "model": "trae",
        "timestamp": time.time(),
    }
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(resp_data, f, ensure_ascii=False, indent=2)

    print(f"[eval] #{call_id} → score={response_dict['metric']}")
    return True


def main():
    print("Bridge Handler 启动 — 自动处理评估调用")
    print(f"监控目录: {BRIDGE_DIR}")

    handled = 0
    while True:
        # 扫描 pending 文件
        pending_files = list(BRIDGE_DIR.glob("pending_*.json"))

        for pending_path in pending_files:
            call_id = pending_path.stem.replace("pending_", "")

            # 检查是否已有 response (避免重复处理)
            resp_path = BRIDGE_DIR / f"response_{call_id}.json"
            if resp_path.exists():
                continue

            try:
                with open(pending_path, "r", encoding="utf-8") as f:
                    pending_data = json.load(f)
            except Exception:
                continue

            # 只处理有 func_spec 的评估调用
            func_spec = pending_data.get("func_spec")
            if func_spec is not None:
                handle_eval_call(pending_path, pending_data)
                handled += 1
            # 代码生成调用 (func_spec=null) 不处理 — 留给子代理

        time.sleep(0.5)  # 500ms 轮询


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nBridge Handler 停止 (共处理 {handled} 个评估调用)")
