"""
进化系统运行器 — 我来当 LLM

进化系统每步会调 3 次 LLM:
  1. 生成/改进/调试/探索策略 (agent.step → plan_and_strategy_query)
  2. 执行策略 (agentrunner.run)
  3. 评估执行结果 (agent.parse_exec_result)

这 3 次调用都经过 evolution/backend/query()。
我把 query() 替换成我的函数 — 请求进来，我分析后直接返回。

用法:
    .venv/bin/python3 evolution/run_with_me_as_llm.py --steps 10
"""
import json, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evolution"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("evolution.run_with_me")

# ═══════════════════════════════════════════════════════
# 我的 LLM 替代函数
# ═══════════════════════════════════════════════════════

# 保存原始 query 函数
_original_query = None
_call_count = 0

def my_llm_query(
    system_message, user_message,
    model="", temperature=None, max_tokens=None,
    func_spec=None, **kwargs
):
    """
    我替代 LLM 的 query 函数。
    
    进化系统调这个函数时:
    1. 我拿到 system_message 和 user_message
    2. 我深度分析请求内容
    3. 我生成高质量回复
    4. 直接返回给进化系统
    """
    global _call_count
    _call_count += 1
    
    # 构建完整 prompt
    full_prompt = ""
    if system_message:
        full_prompt += f"[System]\n{system_message}\n\n"
    if user_message:
        full_prompt += f"[User]\n{user_message}"
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"[LLM 请求 #{_call_count}]")
    logger.info(f"{'='*60}")
    logger.info(f"[用途] {kwargs.get('purpose', 'evolution')}")
    if func_spec:
        logger.info(f"[函数调用] {func_spec.name}")
    logger.info(f"[System] {str(system_message)[:200]}...")
    logger.info(f"[User] {str(user_message)[:300]}...")
    logger.info(f"{'='*60}")
    
    # 把请求写到文件，等我回复
    request_file = ROOT / ".openclaw" / "tmp" / "llm_request.json"
    request_file.parent.mkdir(parents=True, exist_ok=True)
    response_file = ROOT / ".openclaw" / "tmp" / "llm_response.json"
    
    request_file.write_text(json.dumps({
        "call_id": _call_count,
        "system": system_message or "",
        "user": user_message or "",
        "func_spec": func_spec.name if func_spec else None,
        "purpose": kwargs.get("purpose", "evolution"),
        "status": "pending",
    }, ensure_ascii=False, indent=2))
    
    logger.info(f"[等待我的回复...] (请求文件: {request_file})")
    
    # 轮询等待回复
    start = time.time()
    timeout = 600  # 10 分钟
    while time.time() - start < timeout:
        if response_file.exists():
            try:
                data = json.loads(response_file.read_text())
                if data.get("status") == "completed":
                    response = data.get("response", "")
                    logger.info(f"[收到我的回复] {len(response)} 字符")
                    # 清理
                    response_file.unlink()
                    request_file.unlink(missing_ok=True)
                    return response
            except:
                pass
        time.sleep(2)
    
    logger.error(f"[超时] 等待 {timeout} 秒无回复")
    return ""


def install_my_llm():
    """替换进化系统的 LLM query 函数。"""
    import evolution.backend as backend
    global _original_query
    _original_query = backend.query
    backend.query = my_llm_query
    logger.info("已替换 evolution/backend/query → 我的 LLM 函数")


def restore_original_llm():
    """恢复原始 LLM。"""
    if _original_query:
        import evolution.backend as backend
        backend.query = _original_query
        logger.info("已恢复原始 LLM")


# ═══════════════════════════════════════════════════════
# 进化主循环
# ═══════════════════════════════════════════════════════

def run_evolution_with_me(max_steps=10):
    """用我替代 LLM 来运行进化系统。"""
    from evolution.agent import EvolutionAgent
    from evolution.runner import AgentRunner
    from evolution.journal import EvolutionJournal
    from evolution.run import check_convergence
    
    # 安装我的 LLM
    install_my_llm()
    
    journal = EvolutionJournal()
    agent = EvolutionAgent(
        task_desc=(
            "优化小茜融合引擎。目标: 提升意图识别准确率、添加实体提取和情感分析、"
            "提升信息密度和推理链质量。评估维度: accuracy, entity_accuracy, "
            "sentiment_accuracy, fusion_quality, info_density, chain_quality。"
        ),
        journal=journal,
        model="me",  # 不会真正用到，因为 query 被我替换了
        temperature=0.7,
        num_drafts=3,
        debug_prob=0.2,
        explore_prob=0.15,
    )
    
    runner = AgentRunner(
        task_desc="优化融合引擎",
        model="me",
    )
    agent._last_runner = runner
    
    def exec_callback(strategy, reset=True):
        return runner.run(strategy, reset)
    
    logger.info(f"开始进化 (max_steps={max_steps})")
    logger.info(f"我会处理每步的 3 次 LLM 请求")
    
    step = 0
    while step < max_steps:
        step += 1
        logger.info(f"")
        logger.info(f"{'#'*60}")
        logger.info(f"进化步骤 {step}/{max_steps}")
        logger.info(f"{'#'*60}")
        
        try:
            agent.step(exec_callback=exec_callback)
        except Exception as e:
            logger.error(f"步骤失败: {e}")
            import traceback
            traceback.print_exc()
        
        best = journal.get_best_node()
        if best:
            logger.info(f"当前最佳: {best.metric.value:.4f} (节点 {best.id[:8]})")
        
        # 收敛检测
        if step >= 5 and check_convergence(journal, 5, 0.01):
            logger.info(f"在步骤 {step} 收敛")
            break
    
    restore_original_llm()
    
    # 报告
    logger.info("")
    logger.info("=" * 60)
    logger.info("进化完成")
    logger.info(f"总节点: {len(journal)}")
    logger.info(f"好节点: {len(journal.good_nodes)}")
    best = journal.get_best_node()
    if best:
        logger.info(f"最佳: {best.metric.value:.4f}")
        logger.info(f"策略: {best.strategy[:200]}")
    logger.info("=" * 60)
    
    return journal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run_evolution_with_me(args.steps)
