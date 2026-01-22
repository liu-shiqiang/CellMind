/**
 * ClusterHeatmap - 聚类热图组件
 * 用于展示标记基因在各个聚类中的表达模式
 */
import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Download, Maximize2, RotateCw } from 'lucide-react';

export interface HeatmapDataPoint {
  gene: string;
  cluster: string | number;
  expression: number;
  avgExpression?: number;
  pctExpressed?: number;
}

export interface HeatmapData {
  points: HeatmapDataPoint[];
  genes: string[];
  clusters: string[] | number[];
  clusterLabels?: Record<string | number, string>;
}

export interface ClusterHeatmapProps {
  /** 热图数据 */
  data: HeatmapData;
  /** 画布宽度 */
  width?: number;
  /** 画布高度 */
  height?: number;
  /** 单元格宽度 */
  cellWidth?: number;
  /** 单元格高度 */
  cellHeight?: number;
  /** 颜色方案 */
  colorScheme?: 'blue' | 'red' | 'green' | 'purple';
  /** 是否显示聚类标签 */
  showClusterLabels?: boolean;
  /** 是否显示基因标签 */
  showGeneLabels?: true;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 是否交互 */
  interactive?: boolean;
  /** 点击单元格的回调 */
  onCellClick?: (gene: string, cluster: string | number, value: number) => void;
  /** 悬停单元格的回调 */
  onCellHover?: (gene: string, cluster: string | number, value: number | null) => void;
  /** 自定义类名 */
  className?: string;
}

export interface ClusterHeatmapRef {
  /** 导出为图片 */
  exportAsImage: (format?: 'png' | 'svg') => Promise<Blob>;
  /** 重置缩放 */
  resetZoom: () => void;
  /** 切换排序方式 */
  toggleSort: () => void;
}

/**
 * 颜色方案配置
 */
const COLOR_SCHEMES = {
  blue: { min: [243, 247, 255], max: [37, 99, 235] },   // 浅蓝 -> 深蓝
  red: { min: [254, 242, 242], max: [220, 38, 38] },    // 浅红 -> 深红
  green: { min: [240, 253, 244], max: [22, 163, 74] },  // 浅绿 -> 深绿
  purple: { min: [245, 243, 255], max: [139, 92, 246] }, // 浅紫 -> 深紫
};

/**
 * 根据表达值获取颜色
 */
function getExpressionColor(
  value: number,
  minVal: number,
  maxVal: number,
  colorScheme: keyof typeof COLOR_SCHEMES
): string {
  const scheme = COLOR_SCHEMES[colorScheme];
  const normalized = maxVal > minVal ? (value - minVal) / (maxVal - minVal) : 0;

  const r = Math.round(scheme.min[0] + (scheme.max[0] - scheme.min[0]) * normalized);
  const g = Math.round(scheme.min[1] + (scheme.max[1] - scheme.min[1]) * normalized);
  const b = Math.round(scheme.min[2] + (scheme.max[2] - scheme.min[2]) * normalized);

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * ClusterHeatmap组件
 */
export const ClusterHeatmap = React.forwardRef<
  ClusterHeatmapRef,
  ClusterHeatmapProps
>((props, ref) => {
  const {
    data,
    width = 800,
    height = 500,
    cellWidth = 40,
    cellHeight = 25,
    colorScheme = 'blue',
    showClusterLabels = true,
    showGeneLabels = true,
    showLegend = true,
    interactive = true,
    onCellClick,
    onCellHover,
    className = '',
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{ gene: string; cluster: string | number; value: number } | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [sortOrder, setSortOrder] = useState<'default' | 'expression'>('default');

  // 变换状态（缩放/平移）
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 计算布局
  const layout = useMemo(() => {
    const geneLabelWidth = showGeneLabels ? 120 : 20;
    const clusterLabelHeight = showClusterLabels ? 30 : 10;
    const legendWidth = 60;
    const topMargin = 40;
    const leftMargin = 20;

    const gridWidth = data.genes.length * cellWidth;
    const gridHeight = data.clusters.length * cellHeight;

    return {
      geneLabelWidth,
      clusterLabelHeight,
      legendWidth,
      topMargin,
      leftMargin,
      gridWidth,
      gridHeight,
    };
  }, [data.genes.length, data.clusters.length, cellWidth, cellHeight, showGeneLabels, showClusterLabels]);

  // 构建表达矩阵
  const expressionMatrix = useMemo(() => {
    const matrix: Record<string, Record<string | number, number>> = {};

    for (const point of data.points) {
      if (!matrix[point.gene]) {
        matrix[point.gene] = {};
      }
      matrix[point.gene][point.cluster] = point.expression;
    }

    return matrix;
  }, [data.points]);

  // 计算表达值范围
  const expressionRange = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;

    for (const point of data.points) {
      min = Math.min(min, point.expression);
      max = Math.max(max, point.expression);
    }

    return { min, max: max === -Infinity ? 1 : max };
  }, [data.points]);

  // 基因排序（可选）
  const sortedGenes = useMemo(() => {
    if (sortOrder === 'default') {
      return data.genes;
    }

    // 按平均表达量排序
    return [...data.genes].sort((a, b) => {
      const avgA = data.points
        .filter(p => p.gene === a)
        .reduce((sum, p) => sum + p.expression, 0) / data.clusters.length;
      const avgB = data.points
        .filter(p => p.gene === b)
        .reduce((sum, p) => sum + p.expression, 0) / data.clusters.length;
      return avgB - avgA;
    });
  }, [data.genes, data.points, data.clusters.length, sortOrder]);

  // 坐标转换：网格坐标 -> 画布坐标
  const gridToCanvas = useCallback(
    (geneIndex: number, clusterIndex: number) => {
      const { leftMargin, topMargin } = layout;

      const x = leftMargin + geneIndex * cellWidth;
      const y = topMargin + clusterIndex * cellHeight;

      return {
        x: x * transform.scale + transform.offsetX,
        y: y * transform.scale + transform.offsetY,
      };
    },
    [layout, cellWidth, cellHeight, transform]
  );

  // 绘制函数
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    const { leftMargin, topMargin, geneLabelWidth, clusterLabelHeight, legendWidth } = layout;

    // 绘制标题
    ctx.fillStyle = '#334155';
    ctx.font = 'bold 14px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Marker Gene Expression by Cluster', leftMargin, 20);

    // 绘制热图单元格
    for (let i = 0; i < sortedGenes.length; i++) {
      const gene = sortedGenes[i];
      const geneData = expressionMatrix[gene] || {};

      for (let j = 0; j < data.clusters.length; j++) {
        const cluster = data.clusters[j];
        const value = geneData[cluster] ?? 0;
        const pos = gridToCanvas(i, j);

        const color = getExpressionColor(
          value,
          expressionRange.min,
          expressionRange.max,
          colorScheme
        );

        // 绘制单元格
        ctx.fillStyle = color;
        ctx.fillRect(
          pos.x,
          pos.y,
          cellWidth * transform.scale,
          cellHeight * transform.scale
        );

        // 绘制边框
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(
          pos.x,
          pos.y,
          cellWidth * transform.scale,
          cellHeight * transform.scale
        );

        // 高亮悬停单元格
        if (hoveredCell && hoveredCell.gene === gene && hoveredCell.cluster === cluster) {
          ctx.strokeStyle = '#1e293b';
          ctx.lineWidth = 2;
          ctx.strokeRect(
            pos.x,
            pos.y,
            cellWidth * transform.scale,
            cellHeight * transform.scale
          );
        }
      }
    }

    // 绘制基因标签
    if (showGeneLabels && transform.scale > 0.5) {
      ctx.fillStyle = '#475569';
      ctx.font = `${11 * transform.scale}px system-ui, sans-serif`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      for (let i = 0; i < sortedGenes.length; i++) {
        const gene = sortedGenes[i];
        const y = topMargin + (i + 0.5) * cellHeight;

        const pos = gridToCanvas(i, 0);
        ctx.fillText(
          gene.length > 12 ? gene.substring(0, 10) + '...' : gene,
          pos.x - 5,
          pos.y + cellHeight * transform.scale / 2
        );
      }
    }

    // 绘制聚类标签
    if (showClusterLabels && transform.scale > 0.5) {
      ctx.fillStyle = '#475569';
      ctx.font = `${11 * transform.scale}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';

      for (let j = 0; j < data.clusters.length; j++) {
        const cluster = data.clusters[j];
        const label = data.clusterLabels?.[cluster] ?? String(cluster);
        const x = leftMargin + sortedGenes.length * cellWidth / 2;
        const y = topMargin + j * cellHeight;

        const pos = gridToCanvas(0, j);
        ctx.fillText(label, x * transform.scale + transform.offsetX, pos.y - 2);
      }
    }

    // 绘制颜色图例
    const legendX = width - legendWidth - 20;
    const legendY = topMargin;
    const legendHeight = layout.gridHeight * Math.min(1, transform.scale);

    for (let i = 0; i <= 50; i++) {
      const t = i / 50;
      const value = expressionRange.min + t * (expressionRange.max - expressionRange.min);
      const color = getExpressionColor(value, expressionRange.min, expressionRange.max, colorScheme);
      const y = legendY + (1 - t) * legendHeight;

      ctx.fillStyle = color;
      ctx.fillRect(legendX, y, 20, legendHeight / 50);
    }

    // 图例边框
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, 20, legendHeight);

    // 图例标签
    ctx.fillStyle = '#64748b';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(expressionRange.max.toFixed(2), legendX + 25, legendY + 5);
    ctx.fillText(expressionRange.min.toFixed(2), legendX + 25, legendY + legendHeight);
  }, [data, sortedGenes, expressionMatrix, expressionRange, layout, cellWidth, cellHeight,
      colorScheme, showGeneLabels, showClusterLabels, width, height, transform,
      gridToCanvas, hoveredCell]);

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

      // 查找悬停的单元格
      const { leftMargin, topMargin } = layout;
      const gridMouseX = (mouseX - transform.offsetX) / transform.scale - leftMargin;
      const gridMouseY = (mouseY - transform.offsetY) / transform.scale - topMargin;

      const geneIndex = Math.floor(gridMouseX / cellWidth);
      const clusterIndex = Math.floor(gridMouseY / cellHeight);

      if (geneIndex >= 0 && geneIndex < sortedGenes.length &&
          clusterIndex >= 0 && clusterIndex < data.clusters.length) {
        const gene = sortedGenes[geneIndex];
        const cluster = data.clusters[clusterIndex];
        const value = expressionMatrix[gene]?.[cluster] ?? 0;

        setHoveredCell({ gene, cluster, value });
        setTooltipPosition({ x: mouseX, y: mouseY });
        onCellHover?.(gene, cluster, value);
      } else {
        setHoveredCell(null);
        setTooltipPosition(null);
        onCellHover?.('', '', 0);
      }
    },
    [interactive, isDragging, dragStart, transform, layout, sortedGenes,
     data.clusters, cellWidth, cellHeight, expressionMatrix, onCellHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoveredCell(null);
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

    if (hoveredCell) {
      onCellClick?.(hoveredCell.gene, hoveredCell.cluster, hoveredCell.value);
    } else {
      setIsDragging(true);
      setDragStart({ x: mouseX, y: mouseY });
      canvas.style.cursor = 'grabbing';
    }
  }, [interactive, hoveredCell, onCellClick]);

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
    const newScale = Math.max(0.2, Math.min(3, transform.scale * delta));

    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setTransform(prev => {
        const scaleChange = newScale / prev.scale;
        return {
          ...prev,
          scale: newScale,
          offsetX: mouseX - (mouseX - prev.offsetX) * scaleChange,
          offsetY: mouseY - (mouseY - prev.offsetY) * scaleChange,
        };
      });
    }
  }, [interactive, transform]);

  // 重置缩放
  const resetZoom = useCallback(() => {
    setTransform({ scale: 1, offsetX: 0, offsetY: 0 });
  }, []);

  // 切换排序
  const toggleSort = useCallback(() => {
    setSortOrder(prev => prev === 'default' ? 'expression' : 'default');
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
    toggleSort,
  }));

  // 绘制效果
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div
      ref={containerRef}
      className={`relative bg-white rounded-xl border border-slate-200 ${className}`}
    >
      {/* 工具栏 */}
      <div className="absolute top-4 left-4 flex gap-2">
        <button
          onClick={() => setTransform(p => ({ ...p, scale: Math.min(3, p.scale * 1.2) }))}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="放大"
        >
          <ZoomIn size={16} />
        </button>
        <button
          onClick={() => setTransform(p => ({ ...p, scale: Math.max(0.2, p.scale / 1.2) }))}
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
        <button
          onClick={toggleSort}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="切换排序"
        >
          <RotateCw size={16} />
        </button>
        <button
          onClick={async () => {
            const blob = await exportAsImage();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `heatmap_${Date.now()}.png`;
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
          <div className="text-xs font-semibold text-slate-700 mb-2">表达量</div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-20 rounded" style={{
              background: `linear-gradient(to top, ${getExpressionColor(expressionRange.min, expressionRange.min, expressionRange.max, colorScheme)}, ${getExpressionColor(expressionRange.max, expressionRange.min, expressionRange.max, colorScheme)})`
            }} />
            <div className="text-xs text-slate-600 space-y-8">
              <div>{expressionRange.max.toFixed(2)}</div>
              <div>{expressionRange.min.toFixed(2)}</div>
            </div>
          </div>
        </div>
      )}

      {/* 工具提示 */}
      {hoveredCell && tooltipPosition && (
        <div
          className="absolute bg-slate-900/90 text-white text-xs p-2.5 rounded-lg pointer-events-none z-10"
          style={{
            left: tooltipPosition.x + 15,
            top: tooltipPosition.y - 10,
          }}
        >
          <div className="font-semibold mb-1">{hoveredCell.gene}</div>
          <div className="text-slate-300 space-y-0.5">
            <div>聚类: {data.clusterLabels?.[hoveredCell.cluster] ?? String(hoveredCell.cluster)}</div>
            <div>表达量: {hoveredCell.value.toFixed(3)}</div>
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

      {/* 数据信息 */}
      <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg px-3 py-1.5 text-xs text-slate-600">
        {data.genes.length} 基因 × {data.clusters.length} 聚类
      </div>
    </div>
  );
});

ClusterHeatmap.displayName = 'ClusterHeatmap';
