/**
 * VolcanoPlot - 火山图组件
 * 用于展示差异表达分析结果
 */
import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Download, Maximize2, Filter, Settings } from 'lucide-react';

export interface VolcanoDataPoint {
  gene: string;
  log2FoldChange: number;
  negLog10PValue: number;
  pValue?: number;
  padj?: number;
  isSignificant?: boolean;
  cluster?: number;
}

export interface VolcanoData {
  points: VolcanoDataPoint[];
  pValueThreshold?: number;
  foldChangeThreshold?: number;
  labels?: string[];
}

export interface VolcanoPlotProps {
  /** 火山图数据 */
  data: VolcanoData;
  /** 画布宽度 */
  width?: number;
  /** 画布高度 */
  height?: number;
  /** 点大小 */
  pointSize?: number;
  /** P值阈值 (-log10) */
  pValueThreshold?: number;
  /** Fold change阈值 (log2) */
  foldChangeThreshold?: number;
  /** 是否显示标签 */
  showLabels?: boolean;
  /** 显示的标签数量 */
  maxLabels?: number;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 是否交互 */
  interactive?: boolean;
  /** 点击点的回调 */
  onPointClick?: (point: VolcanoDataPoint) => void;
  /** 悬停点的回调 */
  onPointHover?: (point: VolcanoDataPoint | null) => void;
  /** 自定义类名 */
  className?: string;
}

export interface VolcanoPlotRef {
  /** 导出为图片 */
  exportAsImage: (format?: 'png' | 'svg') => Promise<Blob>;
  /** 重置缩放 */
  resetZoom: () => void;
  /** 设置阈值 */
  setThresholds: (pValue: number, foldChange: number): void;
}

/**
 * 计算显著性的颜色
 */
function getPointColor(
  point: VolcanoDataPoint,
  pThreshold: number,
  fcThreshold: number
): string {
  const isSignificant = point.negLog10PValue >= pThreshold &&
    Math.abs(point.log2FoldChange) >= fcThreshold;

  if (!isSignificant) return '#94a3b8'; // 灰色 - 不显著

  if (point.log2FoldChange > 0) return '#ef4444'; // 红色 - 上调
  return '#3b82f6'; // 蓝色 - 下调
}

/**
 * VolcanoPlot组件
 */
export const VolcanoPlot = React.forwardRef<
  VolcanoPlotRef,
  VolcanoPlotProps
>((props, ref) => {
  const {
    data,
    width = 700,
    height = 500,
    pointSize = 4,
    pValueThreshold: propPThreshold = -Math.log10(0.05),
    foldChangeThreshold: propFCThreshold = 1,
    showLabels = true,
    maxLabels = 20,
    showLegend = true,
    interactive = true,
    onPointClick,
    onPointHover,
    className = '',
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<VolcanoDataPoint | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // 阈值状态
  const [pThreshold, setPThreshold] = useState(propPThreshold);
  const [fcThreshold, setFCTreshold] = useState(propFCThreshold);

  // 变换状态（缩放/平移）
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 计算数据边界
  const bounds = useMemo(() => {
    if (!data.points || data.points.length === 0) {
      return {
        minX: -5, maxX: 5,
        minY: 0, maxY: 10
      };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = 0, maxY = -Infinity;

    for (const point of data.points) {
      minX = Math.min(minX, point.log2FoldChange);
      maxX = Math.max(maxX, point.log2FoldChange);
      maxY = Math.max(maxY, point.negLog10PValue);
    }

    // 添加边距
    const xMargin = (maxX - minX) * 0.1;
    const yMargin = maxY * 0.1;

    return {
      minX: minX - xMargin,
      maxX: maxX + xMargin,
      minY: minY,
      maxY: maxY + yMargin
    };
  }, [data.points]);

  // 获取显著基因（用于标签）
  const significantGenes = useMemo(() => {
    return data.points
      .filter(p => p.negLog10PValue >= pThreshold && Math.abs(p.log2FoldChange) >= fcThreshold)
      .sort((a, b) => b.negLog10PValue - a.negLog10PValue)
      .slice(0, maxLabels);
  }, [data.points, pThreshold, fcThreshold, maxLabels]);

  // 坐标转换：数据坐标 -> 画布坐标
  const dataToCanvas = useCallback(
    (x: number, y: number) => {
      const { minX, maxX, minY, maxY } = bounds;
      const padding = { left: 60, right: 20, top: 20, bottom: 50 };

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const canvasX = padding.left + ((x - minX) / rangeX) * (width - padding.left - padding.right);
      const canvasY = height - padding.bottom - ((y - minY) / rangeY) * (height - padding.top - padding.bottom);

      return {
        x: canvasX * transform.scale + transform.offsetX,
        y: canvasY * transform.scale + transform.offsetY,
      };
    },
    [bounds, width, height, transform]
  );

  // 画布坐标 -> 数据坐标
  const canvasToData = useCallback(
    (cx: number, cy: number) => {
      const { minX, maxX, minY, maxY } = bounds;
      const padding = { left: 60, right: 20, top: 20, bottom: 50 };

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const dataX = minX + ((cx - transform.offsetX) / transform.scale - padding.left) / (width - padding.left - padding.right) * rangeX;
      const dataY = minY + ((height - padding.bottom - (cy - transform.offsetY) / transform.scale) / (height - padding.top - padding.bottom)) * rangeY;

      return { x: dataX, y: dataY };
    },
    [bounds, width, height, transform]
  );

  // 绘制函数
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.points || data.points.length === 0) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    const padding = { left: 60, right: 20, top: 20, bottom: 50 };

    // 绘制背景区域（显著区）
    const upperLeft = dataToCanvas(-fcThreshold, pThreshold);
    const upperRight = dataToCanvas(fcThreshold, pThreshold);
    const bottomY = height - padding.bottom;

    // 上调显著区（右上）
    ctx.fillStyle = 'rgba(239, 68, 68, 0.05)';
    ctx.fillRect(upperRight.x, upperRight.y, width - padding.right - upperRight.x, bottomY - upperRight.y);

    // 下调显著区（左上）
    ctx.fillStyle = 'rgba(59, 130, 246, 0.05)';
    ctx.fillRect(padding.left, upperLeft.y, upperLeft.x - padding.left, bottomY - upperLeft.y);

    // 绘制阈值线
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;

    // P值阈值线
    ctx.beginPath();
    const pLineY = dataToCanvas(0, pThreshold).y;
    ctx.moveTo(padding.left, pLineY);
    ctx.lineTo(width - padding.right, pLineY);
    ctx.stroke();

    // Fold change阈值线
    ctx.strokeStyle = '#64748b';
    const fcLeftX = dataToCanvas(-fcThreshold, 0).x;
    const fcRightX = dataToCanvas(fcThreshold, 0).x;

    ctx.beginPath();
    ctx.moveTo(fcLeftX, padding.top);
    ctx.lineTo(fcLeftX, bottomY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(fcRightX, padding.top);
    ctx.lineTo(fcRightX, bottomY);
    ctx.stroke();

    ctx.setLineDash([]);

    // 绘制坐标轴
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, bottomY);
    ctx.lineTo(width - padding.right, bottomY);
    ctx.stroke();

    // 绘制刻度和标签
    ctx.fillStyle = '#64748b';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';

    // X轴刻度
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const x = padding.left + (i / xTicks) * (width - padding.left - padding.right);
      ctx.beginPath();
      ctx.moveTo(x, bottomY);
      ctx.lineTo(x, bottomY + 5);
      ctx.stroke();

      const dataX = canvasToData(x, bottomY).x;
      ctx.fillText(dataX.toFixed(1), x, bottomY + 18);
    }

    // Y轴刻度
    ctx.textAlign = 'right';
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = bottomY - (i / yTicks) * (bottomY - padding.top);
      ctx.beginPath();
      ctx.moveTo(padding.left - 5, y);
      ctx.lineTo(padding.left, y);
      ctx.stroke();

      const dataY = canvasToData(padding.left, y).y;
      ctx.fillText(dataY.toFixed(1), padding.left - 8, y + 4);
    }

    // 轴标签
    ctx.save();
    ctx.fillStyle = '#334155';
    ctx.font = 'bold 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Log2 Fold Change', width / 2, height - 8);

    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('-Log10 P-Value', 0, 0);
    ctx.restore();

    // 绘制数据点
    for (const point of data.points) {
      const pos = dataToCanvas(point.log2FoldChange, point.negLog10PValue);
      const color = getPointColor(point, pThreshold, fcThreshold);

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, pointSize, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // 绘制基因标签
    if (showLabels) {
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'left';

      for (const gene of significantGenes) {
        const pos = dataToCanvas(gene.log2FoldChange, gene.negLog10PValue);

        // 标签背景
        const textWidth = ctx.measureText(gene.gene).width;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillRect(pos.x + 5, pos.y - 8, textWidth + 4, 14);

        ctx.fillStyle = '#334155';
        ctx.fillText(gene.gene, pos.x + 7, pos.y + 3);
      }
    }

    // 绘制高亮边框
    if (hoveredPoint) {
      const pos = dataToCanvas(hoveredPoint.log2FoldChange, hoveredPoint.negLog10PValue);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, pointSize + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [data.points, width, height, pointSize, pThreshold, fcThreshold, showLabels, significantGenes, dataToCanvas, canvasToData, hoveredPoint, transform]);

  // 处理鼠标事件
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!interactive || !data.points) {
        return;
      }

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

      // 查找最近的点
      let minDist = Infinity;
      let nearestPoint: VolcanoDataPoint | null = null;

      const threshold = pointSize + 5;

      for (const point of data.points) {
        const pos = dataToCanvas(point.log2FoldChange, point.negLog10PValue);
        const dist = Math.sqrt((mouseX - pos.x) ** 2 + (mouseY - pos.y) ** 2);

        if (dist < minDist) {
          minDist = dist;
          nearestPoint = point;
        }
      }

      if (minDist <= threshold && nearestPoint) {
        setHoveredPoint(nearestPoint);
        setTooltipPosition({ x: mouseX, y: mouseY });
        onPointHover?.(nearestPoint);
      } else {
        setHoveredPoint(null);
        setTooltipPosition(null);
        onPointHover?.(null);
      }
    },
    [interactive, data.points, pointSize, dataToCanvas, isDragging, dragStart, onPointHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoveredPoint(null);
      setTooltipPosition(null);
      onPointHover?.(null);
    }
  }, [isDragging, onPointHover]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!interactive) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (hoveredPoint) {
      onPointClick?.(hoveredPoint);
    } else {
      setIsDragging(true);
      setDragStart({ x: mouseX, y: mouseY });
      canvas.style.cursor = 'grabbing';
    }
  }, [interactive, hoveredPoint, onPointClick]);

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
    const newScale = Math.max(0.5, Math.min(5, transform.scale * delta));

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

  // 设置阈值
  const setThresholds = useCallback((pValue: number, foldChange: number) => {
    setPThreshold(pValue);
    setFCTreshold(foldChange);
  }, []);

  // 暴露ref方法
  React.useImperativeHandle(ref, () => ({
    exportAsImage,
    resetZoom,
    setThresholds,
  }));

  // 绘制效果
  useEffect(() => {
    draw();
  }, [draw]);

  // 同步外部阈值变化
  useEffect(() => {
    setPThreshold(propPThreshold);
    setFCTreshold(propFCThreshold);
  }, [propPThreshold, propFCThreshold]);

  // 统计显著基因数量
  const stats = useMemo(() => {
    const up = data.points.filter(p =>
      p.negLog10PValue >= pThreshold && p.log2FoldChange >= fcThreshold
    ).length;
    const down = data.points.filter(p =>
      p.negLog10PValue >= pThreshold && p.log2FoldChange <= -fcThreshold
    ).length;
    const total = data.points.length;
    return { up, down, total };
  }, [data.points, pThreshold, fcThreshold]);

  return (
    <div
      ref={containerRef}
      className={`relative bg-white rounded-xl border border-slate-200 ${className}`}
    >
      {/* 工具栏 */}
      <div className="absolute top-4 left-4 flex gap-2">
        <button
          onClick={() => setTransform(p => ({ ...p, scale: Math.min(5, p.scale * 1.2) }))}
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
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="设置阈值"
        >
          <Settings size={16} />
        </button>
        <button
          onClick={async () => {
            const blob = await exportAsImage();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `volcano_${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="p-2 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg hover:bg-white shadow-sm"
          title="导出图片"
        >
          <Download size={16} />
        </button>
      </div>

      {/* 设置面板 */}
      {showSettings && (
        <div className="absolute top-14 left-4 bg-white/95 backdrop-blur-sm border border-slate-200 rounded-lg p-4 shadow-lg z-20 w-64">
          <div className="text-sm font-semibold text-slate-700 mb-3">阈值设置</div>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-600 block mb-1">
                P-Value阈值 (-log10): {pThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={10}
                step={0.1}
                value={pThreshold}
                onChange={(e) => setPThreshold(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-xs text-slate-600 block mb-1">
                Fold Change阈值 (log2): {fcThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={5}
                step={0.1}
                value={fcThreshold}
                onChange={(e) => setFCTreshold(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}

      {/* 图例 */}
      {showLegend && (
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg p-3 shadow-lg">
          <div className="text-xs font-semibold text-slate-700 mb-2">图例</div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-slate-600">上调 ({stats.up})</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-slate-600">下调 ({stats.down})</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="w-3 h-3 rounded-full bg-slate-400" />
              <span className="text-slate-600">不显著</span>
            </div>
            <div className="h-px bg-slate-200 my-1" />
            <div className="text-xs text-slate-500">
              总计: {stats.total.toLocaleString()} 基因
            </div>
          </div>
        </div>
      )}

      {/* 工具提示 */}
      {hoveredPoint && tooltipPosition && (
        <div
          className="absolute bg-slate-900/90 text-white text-xs p-2.5 rounded-lg pointer-events-none z-10 min-w-[150px]"
          style={{
            left: tooltipPosition.x + 15,
            top: tooltipPosition.y - 10,
          }}
        >
          <div className="font-semibold mb-1">{hoveredPoint.gene}</div>
          <div className="text-slate-300 space-y-0.5">
            <div>Log2FC: {hoveredPoint.log2FoldChange.toFixed(3)}</div>
            <div>-Log10P: {hoveredPoint.negLog10PValue.toFixed(3)}</div>
            {hoveredPoint.padj && (
              <div>Adj.P: {hoveredPoint.padj.toExponential(2)}</div>
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

VolcanoPlot.displayName = 'VolcanoPlot';
