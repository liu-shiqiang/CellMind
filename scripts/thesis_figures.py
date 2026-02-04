#!/usr/bin/env python3
"""
Generate figures and analysis results for thesis chapter 4.
Creates:
1. System architecture diagrams
2. Analysis pipeline diagrams
3. Experimental results visualizations
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge
import seaborn as sns

# Setup
os.environ['MPLBACKEND'] = 'Agg'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
FIGURE_DIR = BASE_DIR / "thesis" / "figures" / "chapter4"
DATA_PATH = BASE_DIR / "outputs" / "test_l3_stratified_5pct" / "test_l3_stratified_5pct_annotated_sampled_30pct.h5ad"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# PART 1: System Architecture Diagrams
# ============================================

def draw_agent_architecture(figsize=(12, 8)):
    """Draw the multi-agent system architecture diagram (Figure 4.1)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.8, "图4.1 多智能体协同框架系统架构",
            ha='center', va='top', fontsize=16, weight='bold')

    # Define agent positions
    agents = {
        '意图识别': (2, 6),
        '规划智能体': (4, 6),
        '执行智能体': (6, 6),
        '反思智能体': (8, 6),
        '解读智能体': (10, 6),
    }

    # Draw agents
    agent_colors = {
        '意图识别': '#E1F5FE',
        '规划智能体': '#FFF9C4',
        '执行智能体': '#C8E6C9',
        '反思智能体': '#FFCCBC',
        '解读智能体': '#E1BEE7',
    }

    agent_boxes = {}
    for name, (x, y) in agents.items():
        box = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=agent_colors[name],
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=10, weight='bold')
        agent_boxes[name] = box

    # Draw control flow arrows
    arrows = [
        ('意图识别', '规划智能体', "分析意图"),
        ('规划智能体', '执行智能体', "生成计划"),
        ('执行智能体', '反思智能体', "执行工具"),
        ('反思智能体', '执行智能体', "继续执行", True),
        ('反思智能体', '规划智能体', "重新规划", True),
        ('反思智能体', '解读智能体', "完成任务"),
        ('规划智能体', '解读智能体', "无需执行", True),
        ('意图识别', '解读智能体', "直接响应", True),
    ]

    for src, dst, label, dashed in [(a[0], a[1], a[2], len(a) > 3) for a in arrows]:
        src_x, src_y = agents[src]
        dst_x, dst_y = agents[dst]

        # Calculate arrow positions
        if src_x < dst_x:
            start_x = src_x + 0.8
            end_x = dst_x - 0.8
        else:
            start_x = src_x - 0.8
            end_x = dst_x + 0.8

        start_y = src_y
        end_y = dst_y

        # Draw arrow
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                               arrowstyle='->', mutation_scale=20,
                               color='steelblue', linestyle='--' if dashed else '-',
                               linewidth=1.5 if not dashed else 1.0,
                               alpha=0.7 if dashed else 1.0,
                               zorder=0)
        ax.add_patch(arrow)

        # Label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2 + 0.15
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
                fontsize=8, color='steelblue')

    # Draw state management
    state_box = FancyBboxPatch((3.5, 4.2), 5, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='#F3E5F5',
                               edgecolor='purple', linewidth=1.5)
    ax.add_patch(state_box)
    ax.text(6, 4.6, "AgentState 状态管理", ha='center', va='center',
            fontsize=10, weight='bold', color='purple')

    # State connection lines
    for name in agents.keys():
        if name == '意图识别':
            continue
        x, y = agents[name]
        ax.plot([x, x], [y - 0.4, 4.2 + 0.4], '--',
                color='purple', alpha=0.3, linewidth=1)

    # Draw tool registry
    tool_box = FancyBboxPatch((1, 3), 4, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='#FFF3E0',
                              edgecolor='orange', linewidth=1.5)
    ax.add_patch(tool_box)
    ax.text(3, 3.3, "ToolRegistry 工具注册表", ha='center', va='center',
            fontsize=10, weight='bold', color='orange')

    # Draw memory
    memory_box = FancyBboxPatch((7, 3), 4, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor='#E0F2F1',
                               edgecolor='teal', linewidth=1.5)
    ax.add_patch(memory_box)
    ax.text(9, 3.3, "ConversationMemory 记忆管理", ha='center', va='center',
            fontsize=10, weight='bold', color='teal')

    # External system connection
    ax.plot([4, 4], [3, 2.5], '-', color='orange', alpha=0.5)
    ax.plot([8, 8], [3, 2.5], '-', color='teal', alpha=0.5)

    # Draw external tools
    external_box = FancyBboxPatch((2.5, 1.5), 7, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ECEFF1',
                                  edgecolor='gray', linewidth=1.5)
    ax.add_patch(external_box)
    ax.text(6, 1.8, "外部工具: Scanpy, CellPhoneDB, GSEA, DPT...",
            ha='center', va='center', fontsize=9)

    ax.plot([6, 6], [2.1, 1.8], '-', color='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_1_system_architecture.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_1_system_architecture.png'}")


def draw_coordination_flow(figsize=(14, 8)):
    """Draw the agent coordination workflow diagram (Figure 4.2)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.8, "图4.2 智能体协同工作流程",
            ha='center', va='top', fontsize=16, weight='bold')

    # Define flow steps
    steps = [
        ("用户输入", (2, 6), "#E3F2FD"),
        ("意图识别", (4, 6), "#FFF9C4"),
        ("分支决策", (6, 6), "#F3E5F5"),
        ("规划智能体", (3, 4.5), "#C8E6C9"),
        ("执行智能体", (6, 4.5), "#C8E6C9"),
        ("反思智能体", (9, 4.5), "#C8E6C9"),
        ("解读智能体", (7, 3), "#FFCCBC"),
        ("用户响应", (11, 3), "#E1BEE7"),
    ]

    # Draw step boxes
    boxes = {}
    for name, (x, y), color in steps:
        width = 1.8 if len(name) <= 4 else 2.2
        box = FancyBboxPatch((x - width/2, y - 0.35), width, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='black', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=9, weight='bold')
        boxes[name] = (box, x, y)

    # Draw main flow arrows
    main_flow = [
        ("用户输入", "意图识别"),
        ("意图识别", "分支决策"),
        ("分支决策", "规划智能体", True),
        ("分支决策", "解读智能体", True),
        ("规划智能体", "执行智能体"),
        ("执行智能体", "反思智能体"),
        ("反思智能体", "执行智能体", True),
        ("反思智能体", "规划智能体", True),
        ("反思智能体", "解读智能体"),
        ("解读智能体", "用户响应"),
    ]

    for item in main_flow:
        src = item[0]
        dst = item[1]
        dashed = len(item) > 2

        _, src_x, src_y = boxes[src]
        _, dst_x, dst_y = boxes[dst]

        # Calculate box width
        src_width = boxes[src][0].get_width()
        dst_width = boxes[dst][0].get_width()

        # Calculate endpoints
        if dst_x > src_x:
            start_x = src_x + src_width / 2
            end_x = dst_x - dst_width / 2
        elif dst_x < src_x:
            start_x = src_x - src_width / 2
            end_x = dst_x + dst_width / 2
        else:
            start_x = src_x
            end_x = dst_x

        if dst_y > src_y:
            start_y = src_y + 0.35
            end_y = dst_y - 0.35
        elif dst_y < src_y:
            start_y = src_y - 0.35
            end_y = dst_y + 0.35
        else:
            start_y = src_y
            end_y = dst_y

        # Skip if same position
        if abs(start_x - end_x) < 0.1 and abs(start_y - end_y) < 0.1:
            continue

        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                               arrowstyle='->', mutation_scale=18,
                               color='#1976D2' if not dashed else '#FF9800',
                               linestyle='--' if dashed else '-',
                               linewidth=1.8 if not dashed else 1.2)
        ax.add_patch(arrow)

    # Add labels for key decisions
    ax.text(4.5, 5.3, "需要分析?", ha='center', fontsize=7, color='#666')
    ax.text(7.5, 5.3, "直接响应", ha='center', fontsize=7, color='#666')
    ax.text(7.5, 4, "完成?", ha='center', fontsize=7, color='#666')
    ax.text(6, 3.8, "重新规划", ha='center', fontsize=7, color='#FF9800')
    ax.text(9, 3.8, "继续", ha='center', fontsize=7, color='#FF9800')

    # Add legend
    legend_y = 1.5
    ax.text(7, legend_y + 0.3, "图例", ha='center', fontsize=10, weight='bold')

    # Main flow
    ax.plot([3, 4], [legend_y, legend_y], '-', color='#1976D2', linewidth=2)
    ax.text(4.2, legend_y, "主流程", va='center', fontsize=9)

    # Feedback loop
    ax.plot([7, 8], [legend_y - 0.3, legend_y - 0.3], '--', color='#FF9800', linewidth=1.5)
    ax.text(8.2, legend_y - 0.3, "反馈循环", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_2_coordination_flow.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_2_coordination_flow.png'}")


def draw_state_structure(figsize=(12, 8)):
    """Draw the AgentState structure diagram (Figure 4.3)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.8, "图4.3 AgentState 状态结构",
            ha='center', va='top', fontsize=16, weight='bold')

    # Main state box
    main_box = FancyBboxPatch((2, 1), 8, 6,
                              boxstyle="round,pad=0.1",
                              facecolor='#F5F5F5',
                              edgecolor='black', linewidth=2)
    ax.add_patch(main_box)

    # Category boxes
    categories = [
        ("基础信息", (3, 6.5), ["objective", "messages", "input_files"], "#E3F2FD"),
        ("意图与计划", (5.5, 6.5), ["intents", "plan"], "#FFF9C4"),
        ("执行控制", (8, 6.5), ["next_step", "execution_status", "replan_attempts"], "#FFCCBC"),
        ("工作上下文", (3, 5), ["work_dir", "tool_history", "analysis_notes"], "#C8E6C9"),
        ("会话信息", (5.5, 5), ["thread_id", "session_id", "run_id"], "#E1BEE7"),
        ("内存管理", (8, 5), ["memory_summary", "memory_records", "project_state"], "#F3E5F5"),
    ]

    for name, (x, y), fields, color in categories:
        # Draw category box
        box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y + 0.15, name, ha='center', va='center',
                fontsize=8, weight='bold')

        # List fields below
        for i, field in enumerate(fields):
            ax.text(x, y - 0.2 - i*0.18, f"• {field}",
                    ha='center', va='top', fontsize=7, color='#333')

    # Add data type annotations
    ax.text(3, 4.2, "str\nList[BaseMessage]\nList[str]",
            ha='center', va='top', fontsize=6, color='#666', style='italic')
    ax.text(5.5, 4.2, "List[Intent]\nList[str]",
            ha='center', va='top', fontsize=6, color='#666', style='italic')
    ax.text(8, 4.2, "Optional[str]\nstr\nint",
            ha='center', va='top', fontsize=6, color='#666', style='italic')

    # Add persistence mechanism
    persist_box = FancyBboxPatch((4, 2), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9',
                                edgecolor='green', linewidth=1.5)
    ax.add_patch(persist_box)
    ax.text(6, 2.4, "持久化存储", ha='center', va='center',
            fontsize=9, weight='bold', color='green')
    ax.text(6, 2.1, "MemorySaver → JSON", ha='center', va='center',
            fontsize=7, color='green')

    # Draw arrows to persistence
    for i, (name, (x, y), _, _) in enumerate(categories):
        ax.annotate('', xy=(5, 2.8), xytext=(x, y - 0.8),
                   arrowprops=dict(arrowstyle='->', color='green',
                                linestyle=':', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_3_state_structure.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_3_state_structure.png'}")


# ============================================
# PART 2: Analysis Pipeline Diagrams
# ============================================

def draw_clustering_pipeline(figsize=(14, 6)):
    """Draw the clustering and marker analysis pipeline (Figure 4.4)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(7, 5.7, "图4.4 细胞聚类与标记基因分析流程",
            ha='center', va='top', fontsize=16, weight='bold')

    # Define pipeline steps
    steps = [
        ("数据加载\nload_h5ad_data", (1.5, 4), "#E3F2FD", ".h5ad文件"),
        ("质量控制\ncalculate_qc_metrics", (3.5, 4), "#FFF9C4", "过滤低质量细胞"),
        ("标准化\nnormalize_and_hvg", (5.5, 4), "#C8E6C9", "归一化+HVG"),
        ("PCA降维\npca_reduction", (7.5, 4), "#FFCCBC", "降维到50 PC"),
        ("聚类分析\ncluster_and_umap", (9.5, 4), "#E1BEE7", "Leiden+UMAP"),
        ("标记基因\nfind_marker_genes", (11.5, 4), "#F3E5F5", "Wilcoxon检验"),
    ]

    # Draw step boxes
    for name, (x, y), color, desc in steps:
        # Step box
        box = FancyBboxPatch((x - 0.6, y - 0.3), 1.2, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=7, weight='bold')

        # Description below
        ax.text(x, y - 0.6, desc, ha='center', va='top',
                fontsize=6, color='#555')

    # Draw flow arrows
    for i in range(len(steps) - 1):
        _, (x1, _), _, _ = steps[i]
        _, (x2, _), _, _ = steps[i + 1]

        arrow = FancyArrowPatch((x1 + 0.6, 4), (x2 - 0.6, 4),
                               arrowstyle='->', mutation_scale=15,
                               color='#555', linewidth=1.5)
        ax.add_patch(arrow)

    # Draw outputs
    outputs = [
        ("质控报告", (3.5, 3)),
        ("UMAP图", (9.5, 3)),
        ("聚类结果", (9.5, 2.3)),
        ("标记基因列表", (11.5, 3)),
    ]

    for name, (x, y) in outputs:
        rect = Rectangle((x - 0.5, y - 0.2), 1, 0.4,
                         facecolor='#ECEFF1', edgecolor='gray')
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=7, color='#333')

    # Output arrows
    ax.annotate('', xy=(3.5, 3.2), xytext=(3.5, 3.7),
               arrowprops=dict(arrowstyle='->', color='#888', ls=':'))
    ax.annotate('', xy=(9.5, 3.2), xytext=(9.5, 3.7),
               arrowprops=dict(arrowstyle='->', color='#888', ls=':'))
    ax.annotate('', xy=(11.5, 3.2), xytext=(11.5, 3.7),
               arrowprops=dict(arrowstyle='->', color='#888', ls=':'))

    # Add dependency info
    ax.text(7, 1.2, "工具依赖关系：", ha='center', fontsize=9, weight='bold')
    deps = [
        "cluster_and_umap → normalize_and_hvg, pca_reduction",
        "find_marker_genes → cluster_and_umap",
    ]
    for i, dep in enumerate(deps):
        ax.text(7, 0.8 - i*0.25, dep, ha='center', fontsize=7, color='#666')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_4_clustering_pipeline.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_4_clustering_pipeline.png'}")


def draw_annotation_pipeline(figsize=(12, 6)):
    """Draw the cell annotation pipeline (Figure 4.5)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.7, "图4.5 细胞类型自动注释流程",
            ha='center', va='top', fontsize=16, weight='bold')

    # Input
    input_box = FancyBboxPatch((1, 3.5), 1.5, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='#E3F2FD',
                               edgecolor='blue', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.75, 3.9, "聚类结果\n+标记基因", ha='center', va='center',
            fontsize=8, weight='bold')

    # Decision diamond
    from matplotlib.patches import Polygon
    diamond_coords = [
        (4, 3.9 + 0.5),   # top
        (4.5, 3.9),        # right
        (4, 3.9 - 0.5),    # bottom
        (3.5, 3.9),        # left
    ]
    diamond = Polygon(diamond_coords,
                      facecolor='#FFF9C4',
                      edgecolor='orange', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(4, 3.9, "组织\n类型?", ha='center', va='center',
            fontsize=7, weight='bold')

    # Method boxes
    methods = [
        ("CIMA标记", (2.5, 2.5), "#C8E6C9", "血液/免疫"),
        ("LLM+RAG", (4, 2.5), "#FFCCBC", "脑组织"),
        ("LLM", (5.5, 2.5), "#E1BEE7", "其他组织"),
    ]

    for name, (x, y), color, tissue in methods:
        box = FancyBboxPatch((x - 0.5, y - 0.3), 1, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color,
                              edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y + 0.1, name, ha='center', va='center',
                fontsize=8, weight='bold')
        ax.text(x, y - 0.25, tissue, ha='center', va='center',
                fontsize=6, color='#555')

    # Processing step
    process_box = FancyBboxPatch((7, 3.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#F3E5F5',
                                  edgecolor='purple', linewidth=1.5)
    ax.add_patch(process_box)
    ax.text(8, 3.9, "置信度评分\n+ 备选类型", ha='center', va='center',
            fontsize=8, weight='bold')

    # Output
    output_box = FancyBboxPatch((10, 3.5), 1.5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9',
                                edgecolor='green', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(10.75, 3.9, "注释结果\n.cell_type", ha='center', va='center',
            fontsize=8, weight='bold')

    # Arrows
    arrows = [
        ((1.75, 3.5), (3.5, 3.9)),  # Input to decision
        ((3.5, 3.9), (2.5, 2.8)),  # Decision to CIMA
        ((3.5, 3.9), (4, 2.8)),   # Decision to LLM+RAG
        ((3.5, 3.9), (5.5, 2.8)), # Decision to LLM
        ((2.5, 2.8), (6, 3.5)),   # CIMA to process
        ((4, 2.8), (6, 3.5)),     # LLM+RAG to process
        ((5.5, 2.8), (6, 3.5)),   # LLM to process
        ((6, 3.9), (7, 3.9)),     # Process to output
        ((8, 3.5), (9.2, 3.9)),   # Process to output
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15,
                               color='#555', linewidth=1.2)
        ax.add_patch(arrow)

    # Fallback mechanism
    ax.text(4, 1.8, "回退机制：", ha='center', fontsize=8, weight='bold')
    ax.text(4, 1.5, "CIMA覆盖率 < 60% → LLM补充", ha='center',
            fontsize=7, color='#E65100')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_5_annotation_pipeline.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_5_annotation_pipeline.png'}")


# ============================================
# PART 3: Experimental Result Figures
# ============================================

def generate_umap_cluster_plot(adata, figsize=(10, 4)):
    """Generate UMAP clustering visualization (Figure 4.6)"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Check for required data
    if 'X_umap' not in adata.obsm:
        logger.warning("No UMAP coordinates found, skipping UMAP plot")
        return None

    # Get cluster key
    cluster_key = None
    for key in ['leiden', 'louvain', 'clusters', 'cluster']:
        if key in adata.obs.columns:
            cluster_key = key
            break

    if cluster_key is None:
        logger.warning("No cluster column found, skipping cluster plot")
        return None

    # Plot 1: UMAP by cluster
    sc.pl.umap(adata, color=cluster_key, ax=axes[0], show=False,
               frameon=False, legend_loc='on data', title='UMAP - Clustering')

    # Plot 2: UMAP by cell type (if available)
    if 'cell_type' in adata.obs.columns or 'pred_celltype' in adata.obs.columns:
        ct_key = 'cell_type' if 'cell_type' in adata.obs.columns else 'pred_celltype'
        sc.pl.umap(adata, color=ct_key, ax=axes[1], show=False,
                   frameon=False, legend_loc='on data', title='UMAP - Cell Type')
    else:
        axes[1].text(0.5, 0.5, '细胞类型注释\n尚未完成',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('UMAP - Cell Type')

    plt.suptitle('图4.6 UMAP聚类可视化结果', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_6_umap_clustering.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_6_umap_clustering.png'}")


def generate_celltype_distribution(adata, figsize=(8, 6)):
    """Generate cell type distribution plot (Figure 4.7)"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get cell type column
    ct_key = None
    for key in ['cell_type', 'pred_celltype']:
        if key in adata.obs.columns:
            ct_key = key
            break

    if ct_key is None:
        logger.warning("No cell type column found")
        return None

    # Count cell types
    celltype_counts = adata.obs[ct_key].value_counts()

    # Plot 1: Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(celltype_counts)))
    celltype_counts.plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_xlabel('细胞类型')
    axes[0].set_ylabel('细胞数量')
    axes[0].set_title('细胞类型分布')
    axes[0].tick_params(axis='x', rotation=45)

    # Add count labels
    for i, v in enumerate(celltype_counts):
        axes[0].text(i, v + 0.01 * celltype_counts.max(),
                    str(v), ha='center', va='bottom', fontsize=7)

    # Plot 2: Pie chart
    axes[1].pie(celltype_counts, labels=celltype_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('细胞类型占比')

    plt.suptitle('图4.7 细胞类型注释结果分布', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_7_celltype_distribution.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_7_celltype_distribution.png'}")


def generate_marker_heatmap(adata, figsize=(10, 6)):
    """Generate marker gene heatmap (Figure 4.8)"""
    if "rank_genes_groups" not in adata.uns:
        logger.warning("No marker gene analysis found")
        return None

    # Get cluster key
    cluster_key = None
    for key in ['leiden', 'louvain', 'clusters', 'cluster']:
        if key in adata.obs.columns:
            cluster_key = key
            break

    if cluster_key is None:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    try:
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby=cluster_key,
                                        show=False, ax=ax, cmap='RdBu_r')
        ax.set_title('图4.8 标记基因表达热图', fontsize=14, weight='bold', pad=20)
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")
        return None

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_8_marker_heatmap.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_4_8_marker_heatmap.png'}")


def generate_enrichment_plot(enrichment_file, figsize=(10, 6)):
    """Generate enrichment analysis plot"""
    fig, ax = plt.subplots(figsize=figsize)

    # Try to load enrichment data
    if enrichment_file and enrichment_file.exists():
        try:
            df = pd.read_csv(enrichment_file, sep='\t')
            if 'Term' in df.columns and 'adjP' in df.columns:
                # Get top terms
                top_terms = df.nsmallest(10, 'adjP')

                # Plot
                y_pos = np.arange(len(top_terms))
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_terms)))

                bars = ax.barh(y_pos, -np.log10(top_terms['adjP'] + 1e-300), color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_terms['Term'], fontsize=8)
                ax.set_xlabel('-log10(Adjusted P-value)')
                ax.set_title('图4.9 功能富集分析结果', fontsize=14, weight='bold')
                ax.invert_yaxis()

                # Add p-value labels
                for i, (_, row) in enumerate(top_terms.iterrows()):
                    pval = row['adjP']
                    if pval < 0.001:
                        ax.text(-np.log10(pval + 1e-300), i, f"***",
                                va='center', fontsize=10)
                    elif pval < 0.01:
                        ax.text(-np.log10(pval + 1e-300), i, f"**",
                                va='center', fontsize=10)
                    elif pval < 0.05:
                        ax.text(-np.log10(pval + 1e-300), i, f"*",
                                va='center', fontsize=10)

                plt.tight_layout()
                plt.savefig(FIGURE_DIR / "fig_4_9_enrichment_analysis.png",
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                logger.info(f"Saved: {FIGURE_DIR / 'fig_4_9_enrichment_analysis.png'}")
                return
        except Exception as e:
            logger.warning(f"Could not load enrichment data: {e}")

    # Create placeholder if no data
    ax.text(0.5, 0.5, '功能富集分析结果\n(需要运行富集分析工具)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('图4.9 功能富集分析结果', fontsize=14, weight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_4_9_enrichment_analysis.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved placeholder: {FIGURE_DIR / 'fig_4_9_enrichment_analysis.png'}")


def generate_qc_summary(adata, figsize=(12, 4)):
    """Generate QC summary figure"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Check for QC metrics
    qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    available_metrics = [m for m in qc_metrics if m in adata.obs.columns]

    if not available_metrics:
        logger.warning("No QC metrics available")
        return None

    for i, metric in enumerate(available_metrics[:3]):
        ax = axes[i]

        # Violin plot
        data = adata.obs[metric].values
        parts = ax.violinplot([data], showmeans=True, showmedians=True)

        # Color
        for pc in parts['bodies']:
            pc.set_facecolor('#90CAF9')
            pc.set_alpha(0.7)

        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Value')
        ax.set_xticks([1])
        ax.set_xticklabels(['Cells'])

        # Add stats
        ax.text(0.95, 0.95, f"Median: {np.median(data):.1f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)

    plt.suptitle('质量控制指标分布', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_qc_metrics.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'fig_qc_metrics.png'}")


# ============================================
# Main Execution
# ============================================

def main():
    """Main function to generate all figures"""
    logger.info("Starting thesis chapter 4 figure generation")

    # Part 1: Architecture diagrams (always generate)
    logger.info("Generating architecture diagrams...")
    draw_agent_architecture()
    draw_coordination_flow()
    draw_state_structure()
    draw_clustering_pipeline()
    draw_annotation_pipeline()

    # Part 2: Experimental results (require data)
    if DATA_PATH.exists():
        logger.info(f"Loading data from {DATA_PATH}")
        try:
            adata = sc.read_h5ad(DATA_PATH)
            logger.info(f"Loaded data: {adata.n_obs} cells x {adata.n_vars} genes")

            # Generate experimental figures
            generate_umap_cluster_plot(adata)
            generate_celltype_distribution(adata)
            generate_marker_heatmap(adata)
            generate_qc_summary(adata)

            # Check for enrichment results
            enrichment_file = DATA_PATH.parent / "enrichment" / "enrichment_results.tsv"
            generate_enrichment_plot(enrichment_file)

        except Exception as e:
            logger.error(f"Error generating experimental figures: {e}")
    else:
        logger.warning(f"Data file not found: {DATA_PATH}")
        logger.info("Generating placeholder figures for experimental results...")

        # Generate empty plots with message
        for name in ["fig_4_6_umap_clustering", "fig_4_7_celltype_distribution",
                     "fig_4_8_marker_heatmap", "fig_4_9_enrichment_analysis"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'需要运行单细胞分析\n生成 {name}\n实验数据',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            plt.savefig(FIGURE_DIR / f"{name}.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved placeholder: {name}.png")

    # Generate summary
    summary = f"""
# Thesis Chapter 4 Figures - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Architecture Diagrams
- fig_4_1_system_architecture.png: Multi-agent system architecture
- fig_4_2_coordination_flow.png: Agent coordination workflow
- fig_4_3_state_structure.png: AgentState structure
- fig_4_4_clustering_pipeline.png: Clustering analysis pipeline
- fig_4_5_annotation_pipeline.png: Cell annotation pipeline

## Experimental Results
"""
    if DATA_PATH.exists():
        summary += """
- fig_4_6_umap_clustering.png: UMAP clustering visualization
- fig_4_7_celltype_distribution.png: Cell type distribution
- fig_4_8_marker_heatmap.png: Marker gene heatmap
- fig_4_9_enrichment_analysis.png: Enrichment analysis results
- fig_qc_metrics.png: QC metrics distribution
"""
    else:
        summary += "\n(Experimental data not available - placeholders generated)\n"

    (FIGURE_DIR / "README.md").write_text(summary, encoding='utf-8')
    logger.info(f"Saved summary to {FIGURE_DIR / 'README.md'}")
    logger.info("Figure generation complete!")


if __name__ == "__main__":
    main()
