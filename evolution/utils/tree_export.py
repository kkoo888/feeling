"""导出进化日志为 HTML 树形可视化"""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import EvolutionJournal


def get_edges(journal: EvolutionJournal):
    """获取进化树的边"""
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """生成图的可视化布局"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """将布局归一化到 [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def strip_code_markers(code: str) -> str:
    """移除 Markdown 代码块标记"""
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def cfg_to_tree_struct(cfg, jou: EvolutionJournal):
    """将进化日志转换为树形结构数据"""
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    metrics = np.array([0 for n in jou])

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[textwrap.fill(n.plan, width=80) for n in jou.nodes],
        code=[strip_code_markers(n.strategy) for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=metrics.tolist(),
    )


def generate_html(tree_graph_str: str):
    """生成 HTML 可视化页面"""
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("<placeholder>", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: EvolutionJournal, out_path: Path):
    """生成进化树的 HTML 可视化并写入文件"""
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
    html = generate_html(tree_graph_str)
    with open(out_path, "w") as f:
        f.write(html)
