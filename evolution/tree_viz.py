"""
进化树可视化 — 参考 AIDEML tree_export
========================================
生成交互式 HTML 查看进化树和每个节点的详情。

参考: https://github.com/WecoAI/aideml/blob/main/aide/utils/tree_export.py
"""

import json
import logging
import time
from pathlib import Path
from string import Template
from typing import Any, Dict

logger = logging.getLogger("evolution.tree_viz")


def _node_to_dict(node) -> Dict[str, Any]:
    """将节点转为可序列化的字典"""
    return {
        "id": node.id[:8],
        "full_id": node.id,
        "plan": node.plan or "",
        "strategy_preview": (node.strategy or "")[:300],
        "analysis": (node.analysis or "")[:200],
        "metric": node.metric.value if node.metric else None,
        "is_buggy": node.is_buggy,
        "step": node.step,
        "stage": node.stage_name if hasattr(node, "stage_name") else "unknown",
        "children_count": len(node.children) if hasattr(node, "children") else 0,
        "parent_id": node.parent.id[:8] if hasattr(node, "parent") and node.parent else None,
    }


def _build_tree_data(journal) -> Dict[str, Any]:
    """构建树形数据"""
    nodes = []
    edges = []
    for node in journal.nodes:
        nodes.append(_node_to_dict(node))
        if hasattr(node, "parent") and node.parent:
            edges.append({"from": node.parent.id[:8], "to": node.id[:8]})
    best = journal.get_best_node()
    return {
        "nodes": nodes,
        "edges": edges,
        "best_node_id": best.id[:8] if best else None,
        "total_nodes": len(journal.nodes),
        "good_nodes": len(journal.good_nodes),
        "buggy_nodes": len(journal.buggy_nodes),
    }


_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>进化树可视化</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
  .container { display: flex; height: 100vh; }
  .tree-panel { width: 60%; padding: 20px; overflow-y: auto; }
  .detail-panel { width: 40%; padding: 20px; background: #161b22; border-left: 1px solid #30363d; overflow-y: auto; }
  h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.5em; }
  .stats { display: flex; gap: 20px; margin-bottom: 20px; }
  .stat { background: #161b22; padding: 10px 15px; border-radius: 8px; border: 1px solid #30363d; }
  .stat-value { font-size: 1.5em; font-weight: bold; color: #58a6ff; }
  .stat-label { font-size: 0.8em; color: #8b949e; }
  .node { padding: 12px; margin: 8px 0; border-radius: 8px; cursor: pointer; border: 2px solid transparent; transition: all 0.2s; }
  .node:hover { border-color: #58a6ff; }
  .node.good { background: #0d2818; border-left: 4px solid #3fb950; }
  .node.buggy { background: #2d1117; border-left: 4px solid #f85149; }
  .node.best { border-color: #f0883e; background: #2d1d10; }
  .node-id { font-family: monospace; color: #8b949e; font-size: 0.8em; }
  .node-plan { margin-top: 4px; font-size: 0.9em; }
  .node-metric { float: right; font-weight: bold; color: #3fb950; }
  .node-metric.buggy { color: #f85149; }
  .detail-title { color: #58a6ff; font-size: 1.2em; margin-bottom: 15px; }
  .detail-section { margin-bottom: 15px; }
  .detail-label { color: #8b949e; font-size: 0.8em; margin-bottom: 4px; }
  .detail-content { background: #0d1117; padding: 10px; border-radius: 6px; font-size: 0.85em; white-space: pre-wrap; word-break: break-all; max-height: 300px; overflow-y: auto; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: bold; }
  .badge.good { background: #238636; color: white; }
  .badge.buggy { background: #da3633; color: white; }
  .badge.draft { background: #1f6feb; color: white; }
  .badge.improve { background: #238636; color: white; }
  .badge.debug { background: #da3633; color: white; }
  .badge.explore { background: #a371f7; color: white; }
  .badge.best { background: #f0883e; color: white; }
</style>
</head>
<body>
<div class="container">
  <div class="tree-panel">
    <h1>🌳 进化树</h1>
    <div class="stats">
      <div class="stat"><div class="stat-value">$total_nodes</div><div class="stat-label">总节点</div></div>
      <div class="stat"><div class="stat-value">$good_nodes</div><div class="stat-label">好节点</div></div>
      <div class="stat"><div class="stat-value">$buggy_nodes</div><div class="stat-label">Bug节点</div></div>
    </div>
    <div id="tree">$tree_html</div>
  </div>
  <div class="detail-panel">
    <div id="detail"><p style="color:#8b949e">点击节点查看详情</p></div>
  </div>
</div>
<script>
const data = $tree_json;
function showDetail(id) {
  const node = data.nodes.find(n => n.id === id);
  if (!node) return;
  const d = document.getElementById('detail');
  d.innerHTML = '<div class="detail-title">节点 ' + node.id +
    ' <span class="badge ' + (node.is_buggy ? 'buggy' : 'good') + '">' +
    (node.is_buggy ? 'BUG' : 'OK') + '</span></div>' +
    '<div class="detail-section"><div class="detail-label">计划</div><div class="detail-content">' + (node.plan || '无') + '</div></div>' +
    '<div class="detail-section"><div class="detail-label">策略</div><div class="detail-content">' + (node.strategy_preview || '无') + '</div></div>' +
    '<div class="detail-section"><div class="detail-label">分析</div><div class="detail-content">' + (node.analysis || '无') + '</div></div>' +
    '<div class="detail-section"><div class="detail-label">指标</div><div class="detail-content">' + (node.metric !== null ? node.metric.toFixed(4) : '无') + '</div></div>' +
    '<div class="detail-section"><div class="detail-label">阶段</div><div class="detail-content">' + node.stage + '</div></div>';
}
</script>
</body>
</html>""")


def _build_node_html(node: Dict, is_best: bool) -> str:
    """生成单个节点的 HTML"""
    classes = "node"
    if node["is_buggy"]:
        classes += " buggy"
    else:
        classes += " good"
    if is_best:
        classes += " best"

    metric_str = ""
    if node["metric"] is not None:
        metric_class = "buggy" if node["is_buggy"] else ""
        metric_str = f'<span class="node-metric {metric_class}">{node["metric"]:.4f}</span>'

    plan_preview = (node["plan"] or "无")[:80]
    stage_badge = f'<span class="badge {node["stage"]}">{node["stage"]}</span>' if node["stage"] else ""

    return f'''<div class="{classes}" onclick="showDetail('{node["id"]}')">
  <span class="node-id">{node["id"]}</span> {stage_badge} {metric_str}
  <div class="node-plan">{plan_preview}</div>
</div>'''


def generate_html(journal, output_path=None) -> str:
    """生成进化树可视化 HTML"""
    tree_data = _build_tree_data(journal)
    best_id = tree_data["best_node_id"]
    nodes_html = "\n".join(
        _build_node_html(n, n["id"] == best_id) for n in tree_data["nodes"]
    )
    html = _HTML_TEMPLATE.safe_substitute(
        total_nodes=tree_data["total_nodes"],
        good_nodes=tree_data["good_nodes"],
        buggy_nodes=tree_data["buggy_nodes"],
        tree_html=nodes_html,
        tree_json=json.dumps(tree_data, ensure_ascii=False, indent=2),
    )
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        logger.info(f"进化树已保存: {p}")
    return html


def save_best_solution(journal, output_path=None) -> str | None:
    """保存最佳策略到文件"""
    best = journal.get_best_node()
    if not best:
        return None
    content = f"""# 最佳策略 (节点 {best.id[:8]})
# 指标: {best.metric.value if best.metric else 0:.4f}
# 阶段: {best.stage_name if hasattr(best, 'stage_name') else 'unknown'}
# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

# === 计划 ===
{best.plan or ''}

# === 策略 ===
{best.strategy or ''}

# === 分析 ===
{best.analysis or ''}
"""
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return content
