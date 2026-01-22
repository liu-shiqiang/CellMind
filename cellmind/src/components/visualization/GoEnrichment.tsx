/**
 * GoEnrichment - GO富集分析可视化组件
 * 用于展示基因本体(Gene Ontology)富集分析结果
 */
import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Download, Maximize2, Filter, List } from 'lucide-react';

export type GoCategory = 'BP' | 'MF' | 'CC';

export interface GoTerm {
  id: string;
  name: string;
  category: GoCategory;
  pValue: number;
  negLog10PValue: number;
  adjustedPValue?: number;
  geneRatio: string;  // e.g., "15/100"
  geneCount: number;
  bgCount: number;
  genes: string[];
  description?: string;
}

export interface GoEnrichmentData {
  terms: GoTerm[];
  categories?: GoCategory[];
}

export interface GoEnrichmentProps {
  /** GO富集数据 */
  data: GoEnrichmentData;
  /** 画布宽度 */
  width?: number;
  /** 画布高度 */
  height?: number;
  /** 显示的术语数量 */
  maxTerms?: number;
  /** 默认选择的类别 */
  defaultCategory?: GoCategory | 'all';
  /** P值阈值 */
  pValueThreshold?: number;
  /** 条形高度 */
  barHeight?: number;
  /** 是否显示基因列表 */
  showGenes?: boolean;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 是否交互 */
  interactive?: boolean;
  /** 点击条的回调 */
  onBarClick?: (term: GoTerm) => void;
  /** 悬停条的回调 */
  onBarHover?: (term: GoTerm | null) => void;
  /** 自定义类名 */
  className?: string;
}

export interface GoEnrichmentRef {
  /** 导出为图片 */
  exportAsImage: (format?: 'png' | 'svg') => Promise<Blob>;
  /** 重置缩放 */
  resetZoom: () => void;
  /** 设置类别过滤器 */
  setCategory: (category: GoCategory | 'all') => void;
  /** 设置P值阈值 */
  setPValueThreshold: (threshold: number) => void;
}

/**
 * 类别颜色配置
 */
const CATEGORY_COLORS = {
  BP: '#3b82f6',  // 蓝色 - Biological Process
  MF: '#22c55e',  // 绿色 - Molecular Function
  CC: '#f59e0b',  // 橙色 - Cellular Component
};

const CATEGORY_LABELS = {
  BP: '生物过程 (BP)',
  MF: '分子功能 (MF)',
  CC: '细胞组分 (CC)',
};

/**
 * GoEnrichment组件
 */
export const GoEnrichment = React.forwardRef<
  GoEnrichmentRef,
  GoEnrichmentProps
>((props, ref) => {
  const {
    data,
    width = 700,
    height = 500,
    maxTerms = 20,
    defaultCategory = 'all',
    pValueThreshold: propPThreshold = 0.05,
    barHeight = 24,
    showGenes = true,
    showLegend = true,
    interactive = true,
    onBarClick,
    onBarHover,
    className = '',
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredTerm, setHoveredTerm] = useState<GoTerm | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<GoCategory | 'all'>(defaultCategory);
  const [pValueThreshold, setPValueThreshold] = useState(propPThreshold);
  const [showFilterMenu, setShowFilterMenu] = useState(false);

  // 变换状态（缩放/平移）
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 过滤和排序数据
  const filteredTerms = useMemo(() => {
    let terms = [...data.terms];

    // 按类别过滤
    if (selectedCategory !== 'all') {
      terms = terms.filter(t => t.category === selectedCategory);
    }

    // 按P值过滤
    terms = terms.filter(t => t.pValue <= pValueThreshold);

    // 按P值排序并限制数量
    terms.sort((a, b) => a.pValue - b.pValue);

    return terms.slice(0, maxTerms);
  }, [data.terms, selectedCategory, pValueThreshold, maxTerms]);

  // 计算数值范围
  const valueRange = useMemo(() => {
    if (filteredTerms.length === 0) {
      return { min: 0, max: 10 };
    }

    let max = 0;
    for (const term of filteredTerms) {
      max = Math.max(max, term.negLog10PValue);
    }

    return { min: 0, max: max * 1.1 };
  }, [filteredTerms]);

  // 计算布局
  const layout = useMemo(() => {
    const topMargin = 60;
    const leftMargin = 180;  // 为标签预留空间
    const rightMargin = 60;
    const bottomMargin = 40;

    const chartWidth = width - leftMargin - rightMargin;
    const chartHeight = height - topMargin - bottomMargin;

    return { topMargin, leftMargin, rightMargin, bottomMargin, chartWidth, chartHeight };
  }, [width, height]);

  // 坐标转换：数值 -> 画布X坐标
  const valueToCanvasX = useCallback(
    (value: number) => {
      const { leftMargin, chartWidth } = layout;
      const normalized = valueRange.max > 0 ? value / valueRange.max : 0;
      return leftMargin + normalized * chartWidth;
    },
    [layout, valueRange]
  );

  // 绘制函数
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    const { topMargin, leftMargin, chartWidth, chartHeight } = layout;

    // 绘制标题
    ctx.fillStyle = '#334155';
    ctx.font = 'bold 14px system-ui, sans-serif';
    ctx.textAlign = 'left';
    const title = selectedCategory === 'all'
      ? 'GO Enrichment Analysis'
      : `GO Enrichment - ${CATEGORY_LABELS[selectedCategory]}`;
    ctx.fillText(title, leftMargin, 25);

    // 绘制类别选择器提示
    ctx.fillStyle = '#64748b';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(`显示: ${filteredTerms.length} 个术语`, leftMargin, 45);

    // 绘制X轴
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(leftMargin, topMargin);
    ctx.lineTo(leftMargin + chartWidth, topMargin);
    ctx.stroke();

    // 绘制X轴刻度
    ctx.fillStyle = '#64748b';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const x = leftMargin + (i / xTicks) * chartWidth;
      const value = (i / xTicks) * valueRange.max;

      ctx.beginPath();
      ctx.moveTo(x, topMargin);
      ctx.lineTo(x, topMargin + 5);
      ctx.stroke();

      ctx.fillText(value.toFixed(1), x, topMargin + 16);
    }

    // X轴标签
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('-Log10 P-Value', leftMargin + chartWidth / 2, topMargin + 30);

    // 绘制条形
    const actualBarHeight = barHeight * transform.scale;
    const barGap = 4 * transform.scale;
    const startY = topMargin + 40;

    for (let i = 0; i < filteredTerms.length; i++) {
      const term = filteredTerms[i];
      const y = startY + i * (actualBarHeight + barGap);
      const barWidth = valueToCanvasX(term.negLog10PValue) - leftMargin;

      // 跳过超出画布的条
      if (y + actualBarHeight > height - 40) break;

      const color = CATEGORY_COLORS[term.category];

      // 绘制条形背景
      ctx.fillStyle = 'rgba(0, 0, 0, 0.03)';
      ctx.fillRect(leftMargin, y, chartWidth, actualBarHeight);

      // 绘制条形
      const gradient = ctx.createLinearGradient(leftMargin, y, leftMargin + barWidth, y);
      gradient.addColorStop(0, color + 'cc');
      gradient.addColorStop(1, color);
      ctx.fillStyle = gradient;
      ctx.fillRect(leftMargin, y, barWidth, actualBarHeight);

      // 高亮悬停条
      if (hoveredTerm && hoveredTerm.id === term.id) {
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 2;
        ctx.strokeRect(leftMargin - 2, y - 2, barWidth + 4, actualBarHeight + 4);
      }

      // 绘制术语标签
      ctx.fillStyle = '#334155';
      ctx.font = `${11 * transform.scale}px system-ui, sans-serif`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const maxWidth = leftMargin - 10;
      let displayName = term.name;
      const textWidth = ctx.measureText(displayName).width;

      if (textWidth > maxWidth) {
        // 简化名称
        const words = displayName.split(/\s+/);
        if (words.length > 3) {
          displayName = words.slice(0, 3).join(' ') + '...';
        } else {
          while (ctx.measureText(displayName + '...').width > maxWidth && displayName.length > 0) {
            displayName = displayName.slice(0, -1);
          }
          displayName += '...';
        }
      }

      ctx.fillText(displayName, leftMargin - 8, y + actualBarHeight / 2);

      // 绘制数值标签
      ctx.textAlign = 'left';
      ctx.fillStyle = '#64748b';
      ctx.font = `${10 * transform.scale}px system-ui, sans-serif`;
      ctx.fillText(
        term.negLog10PValue.toFixed(2),
        leftMargin + barWidth + 5,
        y + actualBarHeight / 2
      );
    }

    // 绘制P值阈值线
    const thresholdX = valueToCanvasX(-Math.log10(pValueThreshold));
    if (thresholdX > leftMargin) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(thresholdX, topMargin);
      ctx.lineTo(thresholdX, height - 40);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [data, filteredTerms, valueRange, layout, width, height, barHeight,
      selectedCategory, pValueThreshold, transform, valueToCanvasX, hoveredTerm]);

  // 处理鼠标事件
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!interactive) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // 处理拖拽
      if (isDragging) {
        const dx = mouseX - dragStart.x;
        const dy = mouseY - dragStart.y;
        setTransform(prev => ({
          ...prev,
          offsetX: prev.offsetX + dx,
          offsetY: prev.offsetY + dy,
        }));
        setDragStart({ x: mouseX, y: mouseY });
        return;
      }

      // 查找悬停的条
      const { topMargin, leftMargin } = layout;
      const actualBarHeight = barHeight * transform.scale;
      const barGap = 4 * transform.scale;
      const startY = topMargin + 40 + transform.offsetY;
      const adjustedMouseY = mouseY - transform.offsetY;

      const index = Math.floor((adjustedMouseY - startY) / (actualBarHeight + barGap));

      if (index >= 0 && index < filteredTerms.length) {
        const term = filteredTerms[index];
        const barEnd = valueToCanvasX(term.negLog10PValue);

        if (mouseX >= leftMargin && mouseX <= barEnd + 10) {
          setHoveredTerm(term);
          setTooltipPosition({ x: mouseX, y: mouseY });
          onBarHover?.(term);
          return;
        }
      }

      setHoveredTerm(null);
      setTooltipPosition(null);
      onBarHover?.(null);
    },
    [interactive, isDragging, dragStart, transform, layout, barHeight,
     filteredTerms, valueToCanvasX, onBarHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoveredTerm(null);
      setTooltipPosition(null);
    }
  }, [isDragging]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!interactive) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (hoveredTerm) {
      onBarClick?.(hoveredTerm);
    } else {
      setIsDragging(true);
      setDragStart({ x: mouseX, y: mouseY });
      canvas.style.cursor = 'grabbing';
    }
  }, [interactive, hoveredTerm, onBarClick]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.style.cursor = interactive ? 'grab' : 'default';
      }
    }
  }, [isDragging, interactive]);

  // 处理滚轮缩放
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    if (!interactive || !e.ctrlKey) return;
    e.preventDefault();

    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.5, Math.min(2, transform.scale * delta));

    setTransform(prev => ({ ...prev, scale: newScale }));
  }, [interactive, transform]);

  // 重置缩放
  const resetZoom = useCallback(() => {
    setTransform({ scale: 1, offsetX: 0, offsetY: 0 });
  }, []);

  // 设置类别
  const setCategory = useCallback((category: GoCategory | 'all') => {
    setSelectedCategory(category);
  }, []);

  // 导出图片
  const exportAsImage = useCallback(async (format: 'png' | 'svg' = 'png') => {
    const canvas = canvasRef.current;
    if (!canvas) {
      throw new Error('Canvas not available');
    }

    if (format === 'png') {
      return new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          resolve(blob!);
        }, 'image/png');
      });
    }

    throw new Error('SVG export not yet implemented');
  }, []);

  // 暴露ref方法
  React.useImperativeHandle(ref, () => ({
    exportAsImage,
    resetZoom,
    setCategory,
    setPValueThreshold,
  }));

  // 绘制效果
  useEffect(() => {
    draw();
  }, [draw]);

  // 同步外部P值阈值变化
  useEffect(() => {
    setPValueThreshold(propPThreshold);
  }, [propPThreshold]);

  // 统计每个类别的术语数量
  const categoryStats = useMemo(() => {
    const stats: Record<GoCategory, number> = { BP: 0, MF: 0, CC: 0 };
    for (const term of data.terms) {
      if (term.pValue <= pValueThreshold) {
        stats[term.category]++;
      }
    }
    return stats;
  }, [data.terms, pValueThreshold]);

  return (
    <div
      ref={containerRef}
      className={`relative bg-white rounded-xl border border-slate-200 ${className}`}
    >
      {/* 工具栏 */}
      <div className="absolute top-4 left-4 flex gap-2">
        <button
          onClick={() => setTransform(p => ({ ...p, scale: Math.min(2, p.scale * 1.2) }))}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="放大"
        >
          <ZoomIn size={16} />
        </button>
        <button
          onClick={() => setTransform(p => ({ ...p, scale: Math.max(0.5, p.scale / 1.2) }))}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="缩小"
        >
          <ZoomOut size={16} />
        </button>
        <button
          onClick={resetZoom}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="重置视图"
        >
          <Maximize2 size={16} />
        </button>

        {/* 类别过滤器 */}
        <div className="relative">
          <button
            onClick={() => setShowFilterMenu(!showFilterMenu)}
            className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm flex items-center gap-1"
            title="过滤类别"
          >
            <Filter size={16} />
            <span className="text-xs font-semibold text-slate-600">
              {selectedCategory === 'all' ? '全部' : selectedCategory}
            </span>
          </button>

          {showFilterMenu && (
            <div className="absolute top-full left-0 mt-2 bg-white border border-slate-200 rounded-lg shadow-lg py-1 z-20 min-w-[120px]">
              <button
                onClick={() => { setSelectedCategory('all'); setShowFilterMenu(false); }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-slate-50 flex items-center gap-2 ${
                  selectedCategory === 'all' ? 'bg-blue-50 text-blue-700' : 'text-slate-700'
                }`}
              >
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-blue-500 via-green-500 to-orange-500" />
                全部 ({data.terms.filter(t => t.pValue <= pValueThreshold).length})
              </button>
              <button
                onClick={() => { setSelectedCategory('BP'); setShowFilterMenu(false); }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-slate-50 flex items-center gap-2 ${
                  selectedCategory === 'BP' ? 'bg-blue-50 text-blue-700' : 'text-slate-700'
                }`}
              >
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                BP ({categoryStats.BP})
              </button>
              <button
                onClick={() => { setSelectedCategory('MF'); setShowFilterMenu(false); }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-slate-50 flex items-center gap-2 ${
                  selectedCategory === 'MF' ? 'bg-green-50 text-green-700' : 'text-slate-700'
                }`}
              >
                <div className="w-3 h-3 rounded-full bg-green-500" />
                MF ({categoryStats.MF})
              </button>
              <button
                onClick={() => { setSelectedCategory('CC'); setShowFilterMenu(false); }}
                className={`w-full px-3 py-2 text-left text-sm hover:bg-slate-50 flex items-center gap-2 ${
                  selectedCategory === 'CC' ? 'bg-orange-50 text-orange-700' : 'text-slate-700'
                }`}
              >
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                CC ({categoryStats.CC})
              </button>
            </div>
          )}
        </div>

        <button
          onClick={async () => {
            const blob = await exportAsImage();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `go_enrichment_${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="导出图片"
        >
          <Download size={16} />
        </button>
      </div>

      {/* 图例 */}
      {showLegend && (
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg p-3 shadow-lg">
          <div className="text-xs font-semibold text-slate-700 mb-2">GO类别</div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-slate-600">生物过程 (BP)</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-slate-600">分子功能 (MF)</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              <span className="text-slate-600">细胞组分 (CC)</span>
            </div>
          </div>
        </div>
      )}

      {/* 工具提示 */}
      {hoveredTerm && tooltipPosition && (
        <div
          className="absolute bg-slate-900/90 text-white text-xs p-3 rounded-lg pointer-events-none z-10 max-w-[280px]"
          style={{
            left: Math.min(tooltipPosition.x + 15, width - 290),
            top: Math.min(tooltipPosition.y - 10, height - 150),
          }}
        >
          <div className="font-semibold mb-1.5">{hoveredTerm.name}</div>
          <div className="space-y-1 text-slate-300">
            <div className="flex justify-between gap-4">
              <span>类别:</span>
              <span className="font-medium">{CATEGORY_LABELS[hoveredTerm.category]}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span>P值:</span>
              <span className="font-medium">{hoveredTerm.pValue.toExponential(2)}</span>
            </div>
            {hoveredTerm.adjustedPValue && (
              <div className="flex justify-between gap-4">
                <span>校正P值:</span>
                <span className="font-medium">{hoveredTerm.adjustedPValue.toExponential(2)}</span>
              </div>
            )}
            <div className="flex justify-between gap-4">
              <span>基因比:</span>
              <span className="font-medium">{hoveredTerm.geneRatio}</span>
            </div>
            {showGenes && hoveredTerm.genes && hoveredTerm.genes.length > 0 && (
              <div className="mt-2 pt-2 border-t border-slate-700">
                <div className="font-medium mb-1">相关基因:</div>
                <div className="flex flex-wrap gap-1">
                  {hoveredTerm.genes.slice(0, 8).map((g, i) => (
                    <span key={i} className="px-1.5 py-0.5 bg-slate-700 rounded text-[10px]">
                      {g}
                    </span>
                  ))}
                  {hoveredTerm.genes.length > 8 && (
                    <span className="text-[10px] text-slate-400">
                      +{hoveredTerm.genes.length - 8} 更多
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        style={{ cursor: interactive ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
        className="rounded-xl"
      />
    </div>
  );
});

GoEnrichment.displayName = 'GoEnrichment';
