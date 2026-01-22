/**
 * DotPlot - 点图组件
 * 用于展示基因在不同聚类中的表达量和表达比例
 * 点的大小表示表达比例，点的颜色表示平均表达量
 */
import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Download, Maximize2, Grid3x3 } from 'lucide-react';

export interface DotPlotDataPoint {
  gene: string;
  cluster: string | number;
  avgExpression: number;
  pctExpressed: number;  // 0-1
}

export interface DotPlotData {
  points: DotPlotDataPoint[];
  genes: string[];
  clusters: string[] | number[];
  clusterLabels?: Record<string | number, string>;
}

export interface DotPlotProps {
  /** 点图数据 */
  data: DotPlotData;
  /** 画布宽度 */
  width?: number;
  /** 画布高度 */
  height?: number;
  /** 基因间距 */
  geneSpacing?: number;
  /** 聚类间距 */
  clusterSpacing?: number;
  /** 最大点大小 */
  maxDotSize?: number;
  /** 颜色方案 */
  colorScheme?: 'blue' | 'red' | 'green' | 'purple';
  /** 是否显示聚类标签 */
  showClusterLabels?: boolean;
  /** 是否显示基因标签 */
  showGeneLabels?: boolean;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 是否交互 */
  interactive?: boolean;
  /** 点击点的回调 */
  onDotClick?: (point: DotPlotDataPoint) => void;
  /** 悬停点的回调 */
  onDotHover?: (point: DotPlotDataPoint | null) => void;
  /** 自定义类名 */
  className?: string;
}

export interface DotPlotRef {
  /** 导出为图片 */
  exportAsImage: (format?: 'png' | 'svg') => Promise<Blob>;
  /** 重置缩放 */
  resetZoom: () => void;
}

/**
 * 颜色方案配置
 */
const COLOR_SCHEMES = {
  blue: { min: [248, 250, 252], max: [29, 78, 216] },
  red: { min: [254, 242, 242], max: [185, 28, 28] },
  green: { min: [240, 253, 244], max: [21, 128, 61] },
  purple: { min: [250, 245, 255], max: [107, 33, 168] },
};

/**
 * 根据表达量获取颜色
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
 * DotPlot组件
 */
export const DotPlot = React.forwardRef<
  DotPlotRef,
  DotPlotProps
>((props, ref) => {
  const {
    data,
    width = 800,
    height = 500,
    geneSpacing = 60,
    clusterSpacing = 50,
    maxDotSize = 18,
    colorScheme = 'blue',
    showClusterLabels = true,
    showGeneLabels = true,
    showLegend = true,
    interactive = true,
    onDotClick,
    onDotHover,
    className = '',
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredDot, setHoveredDot] = useState<DotPlotDataPoint | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);

  // 变换状态（缩放/平移）
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 计算布局
  const layout = useMemo(() => {
    const geneLabelWidth = showGeneLabels ? 100 : 20;
    const clusterLabelHeight = showClusterLabels ? 40 : 20;
    const topMargin = 50;
    const leftMargin = 20;

    const gridWidth = data.clusters.length * clusterSpacing;
    const gridHeight = data.genes.length * geneSpacing;

    return {
      geneLabelWidth,
      clusterLabelHeight,
      topMargin,
      leftMargin,
      gridWidth,
      gridHeight,
    };
  }, [data.genes.length, data.clusters.length, geneSpacing, clusterSpacing, showGeneLabels, showClusterLabels]);

  // 构建数据矩阵
  const dataMatrix = useMemo(() => {
    const matrix: Record<string, Record<string | number, DotPlotDataPoint>> = {};

    for (const point of data.points) {
      if (!matrix[point.gene]) {
        matrix[point.gene] = {};
      }
      matrix[point.gene][point.cluster] = point;
    }

    return matrix;
  }, [data.points]);

  // 计算表达量范围
  const expressionRange = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;

    for (const point of data.points) {
      min = Math.min(min, point.avgExpression);
      max = Math.max(max, point.avgExpression);
    }

    return { min, max: max === -Infinity ? 1 : max };
  }, [data.points]);

  // 计算表达比例范围
  const pctRange = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;

    for (const point of data.points) {
      min = Math.min(min, point.pctExpressed);
      max = Math.max(max, point.pctExpressed);
    }

    return { min, max: max === -Infinity ? 1 : max };
  }, [data.points]);

  // 坐标转换：网格坐标 -> 画布坐标
  const gridToCanvas = useCallback(
    (geneIndex: number, clusterIndex: number) => {
      const { leftMargin, topMargin } = layout;

      const x = leftMargin + clusterIndex * clusterSpacing;
      const y = topMargin + geneIndex * geneSpacing;

      return {
        x: x * transform.scale + transform.offsetX,
        y: y * transform.scale + transform.offsetY,
      };
    },
    [layout, clusterSpacing, geneSpacing, transform]
  );

  // 获取点的大小
  const getDotSize = useCallback((pctExpressed: number): number => {
    const normalized = pctRange.max > pctRange.min
      ? (pctExpressed - pctRange.min) / (pctRange.max - pctRange.min)
      : 0.5;
    return 2 + normalized * (maxDotSize - 2);
  }, [pctRange, maxDotSize]);

  // 绘制函数
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    const { leftMargin, topMargin } = layout;

    // 绘制标题
    ctx.fillStyle = '#334155';
    ctx.font = 'bold 14px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Gene Expression by Cluster', leftMargin, 25);

    // 绘制网格线（水平）
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= data.genes.length; i++) {
      const y = topMargin + i * geneSpacing;
      ctx.beginPath();
      ctx.moveTo(leftMargin * transform.scale + transform.offsetX, y * transform.scale + transform.offsetY);
      ctx.lineTo((leftMargin + data.clusters.length * clusterSpacing) * transform.scale + transform.offsetX,
                 y * transform.scale + transform.offsetY);
      ctx.stroke();
    }

    // 绘制数据点
    for (let i = 0; i < data.genes.length; i++) {
      const gene = data.genes[i];
      const geneData = dataMatrix[gene] || {};

      for (let j = 0; j < data.clusters.length; j++) {
        const cluster = data.clusters[j];
        const point = geneData[cluster];

        if (point) {
          const pos = gridToCanvas(i, j);
          const color = getExpressionColor(
            point.avgExpression,
            expressionRange.min,
            expressionRange.max,
            colorScheme
          );
          const dotSize = getDotSize(point.pctExpressed) * transform.scale;

          // 绘制点
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, Math.max(0, dotSize), 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();

          // 高亮悬停点
          if (hoveredDot && hoveredDot.gene === gene && hoveredDot.cluster === cluster) {
            ctx.strokeStyle = '#1e293b';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
      }
    }

    // 绘制基因标签
    if (showGeneLabels && transform.scale > 0.4) {
      ctx.fillStyle = '#475569';
      ctx.font = `${11 * transform.scale}px system-ui, sans-serif`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      for (let i = 0; i < data.genes.length; i++) {
        const gene = data.genes[i];
        const pos = gridToCanvas(i, 0);

        const truncatedGene = gene.length > 15 ? gene.substring(0, 13) + '...' : gene;
        ctx.fillText(
          truncatedGene,
          pos.x - 10,
          pos.y
        );
      }
    }

    // 绘制聚类标签
    if (showClusterLabels && transform.scale > 0.4) {
      ctx.fillStyle = '#475569';
      ctx.font = `bold ${11 * transform.scale}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      for (let j = 0; j < data.clusters.length; j++) {
        const cluster = data.clusters[j];
        const label = data.clusterLabels?.[cluster] ?? String(cluster);
        const pos = gridToCanvas(0, j);

        ctx.fillText(
          label.length > 10 ? label.substring(0, 8) + '...' : label,
          pos.x,
          pos.y - 10
        );
      }
    }
  }, [data, dataMatrix, expressionRange, pctRange, layout, clusterSpacing, geneSpacing,
      colorScheme, showGeneLabels, showClusterLabels, width, height, transform,
      gridToCanvas, getDotSize, hoveredDot]);

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

      // 查找悬停的点
      const { leftMargin, topMargin } = layout;
      const gridMouseX = (mouseX - transform.offsetX) / transform.scale - leftMargin;
      const gridMouseY = (mouseY - transform.offsetY) / transform.scale - topMargin;

      const clusterIndex = Math.round(gridMouseX / clusterSpacing);
      const geneIndex = Math.round(gridMouseY / geneSpacing);

      if (clusterIndex >= 0 && clusterIndex < data.clusters.length &&
          geneIndex >= 0 && geneIndex < data.genes.length) {
        const gene = data.genes[geneIndex];
        const cluster = data.clusters[clusterIndex];
        const point = dataMatrix[gene]?.[cluster];

        if (point) {
          setHoveredDot(point);
          setTooltipPosition({ x: mouseX, y: mouseY });
          onDotHover?.(point);
        } else {
          setHoveredDot(null);
          setTooltipPosition(null);
          onDotHover?.(null);
        }
      } else {
        setHoveredDot(null);
        setTooltipPosition(null);
        onDotHover?.(null);
      }
    },
    [interactive, isDragging, dragStart, transform, layout, clusterSpacing, geneSpacing,
     data.clusters, data.genes, dataMatrix, onDotHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoveredDot(null);
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

    if (hoveredDot) {
      onDotClick?.(hoveredDot);
    } else {
      setIsDragging(true);
      setDragStart({ x: mouseX, y: mouseY });
      canvas.style.cursor = 'grabbing';
    }
  }, [interactive, hoveredDot, onDotClick]);

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
          onClick={async () => {
            const blob = await exportAsImage();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dotplot_${Date.now()}.png`;
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
          <div className="text-xs font-semibold text-slate-700 mb-2">图例</div>

          {/* 大小图例 */}
          <div className="mb-2">
            <div className="text-xs text-slate-500 mb-1">表达比例</div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-slate-400" />
              <div className="w-3 h-3 rounded-full bg-slate-400" />
              <div className="w-5 h-5 rounded-full bg-slate-400" />
              <div className="w-7 h-7 rounded-full bg-slate-400" />
            </div>
          </div>

          {/* 颜色图例 */}
          <div>
            <div className="text-xs text-slate-500 mb-1">平均表达量</div>
            <div className="flex items-center gap-1">
              <div className="w-5 h-3 rounded" style={{
                backgroundColor: getExpressionColor(expressionRange.min, expressionRange.min, expressionRange.max, colorScheme)
              }} />
              <div className="w-5 h-3 rounded" style={{
                backgroundColor: getExpressionColor((expressionRange.min + expressionRange.max) / 2, expressionRange.min, expressionRange.max, colorScheme)
              }} />
              <div className="w-5 h-3 rounded" style={{
                backgroundColor: getExpressionColor(expressionRange.max, expressionRange.min, expressionRange.max, colorScheme)
              }} />
            </div>
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>{expressionRange.min.toFixed(2)}</span>
              <span>{expressionRange.max.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}

      {/* 工具提示 */}
      {hoveredDot && tooltipPosition && (
        <div
          className="absolute bg-slate-900/90 text-white text-xs p-2.5 rounded-lg pointer-events-none z-10"
          style={{
            left: tooltipPosition.x + 15,
            top: tooltipPosition.y - 10,
          }}
        >
          <div className="font-semibold mb-1">{hoveredDot.gene}</div>
          <div className="text-slate-300 space-y-0.5">
            <div>聚类: {data.clusterLabels?.[hoveredDot.cluster] ?? String(hoveredDot.cluster)}</div>
            <div>平均表达: {hoveredDot.avgExpression.toFixed(3)}</div>
            <div>表达比例: {(hoveredDot.pctExpressed * 100).toFixed(1)}%</div>
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

DotPlot.displayName = 'DotPlot';
