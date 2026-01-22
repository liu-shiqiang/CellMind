/**
 * UmapVisualization - UMAP/t-SNE降维结果可视化组件
 * 使用Canvas实现高性能渲染，支持大量数据点
 */
import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Download, Maximize2, Info } from 'lucide-react';

export interface UmapDataPoint {
  index?: number;
  x: number;
  y: number;
  cluster?: number;
  cellType?: string;
  metadata?: Record<string, any>;
}

export interface UmapData {
  points: UmapDataPoint[];
  clusterLabels?: string[];
  cellTypeLabels?: string[];
  metadataFields?: string[];
}

export interface UmapVisualizationProps {
  /** UMAP数据 */
  data: UmapData;
  /** 画布宽度 */
  width?: number;
  /** 画布高度 */
  height?: number;
  /** 点大小 */
  pointSize?: number;
  /** 着色方式 */
  colorBy?: 'cluster' | 'cellType' | 'metadata';
  /** 元数据字段（当colorBy='metadata'时使用） */
  colorKey?: string;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 是否交互 */
  interactive?: boolean;
  /** 点击点的回调 */
  onPointClick?: (point: UmapDataPoint, index: number) => void;
  /** 悬停点的回调 */
  onPointHover?: (point: UmapDataPoint | null, index: number) => void;
  /** 自定义类名 */
  className?: string;
  /** 工具提示内容 */
  getTooltipContent?: (point: UmapDataPoint) => string;
}

export interface UmapVisualizationRef {
  /** 导出为图片 */
  exportAsImage: (format?: 'png' | 'svg') => Promise<Blob>;
  /** 重置缩放 */
  resetZoom: () => void;
  /** 高亮特定点 */
  highlightPoints: (indices: number[]) => void;
  /** 清除高亮 */
  clearHighlight: () => void;
}

/**
 * 颜色生成器 - 为聚类生成可区分的颜色
 */
function generateClusterColors(n: number): string[] {
  const colors: string[] = [];
  const hueStep = 360 / Math.max(n, 1);

  for (let i = 0; i < n; i++) {
    const hue = Math.round(i * hueStep);
    const saturation = 70;
    const lightness = 50;
    colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
  }

  return colors;
}

const DEFAULT_CLUSTER_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
];

/**
 * UmapVisualization组件
 */
export const UmapVisualization = React.forwardRef<
  UmapVisualizationRef,
  UmapVisualizationProps
>((props, ref) => {
  const {
    data,
    width = 600,
    height = 400,
    pointSize = 3,
    colorBy = 'cluster',
    colorKey,
    showLegend = true,
    interactive = true,
    onPointClick,
    onPointHover,
    className = '',
    getTooltipContent,
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<UmapDataPoint | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);

  // 变换状态（缩放/平移）
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 计算数据边界
  const bounds = useMemo(() => {
    if (!data.points || data.points.length === 0) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (const point of data.points) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    return { minX, maxX, minY, maxY };
  }, [data.points]);

  // 生成颜色映射
  const colorMap = useMemo(() => {
    const map = new Map<number | string, string>();
    const uniqueValues = new Set<number | string>();

    // 收集唯一值
    for (const point of data.points) {
      if (colorBy === 'cluster' && point.cluster !== undefined) {
        uniqueValues.add(point.cluster);
      } else if (colorBy === 'cellType' && point.cellType) {
        uniqueValues.add(point.cellType);
      } else if (colorBy === 'metadata' && colorKey && point.metadata) {
        const value = point.metadata[colorKey];
        if (value !== undefined) {
          uniqueValues.add(value);
        }
      }
    }

    // 分配颜色
    const colors = generateClusterColors(uniqueValues.size);
    let i = 0;
    for (const value of uniqueValues) {
      map.set(value, colors[i % colors.length]);
      i++;
    }

    return map;
  }, [data.points, colorBy, colorKey]);

  // 获取点的颜色
  const getPointColor = useCallback((point: UmapDataPoint): string => {
    let key: number | string | undefined;

    if (colorBy === 'cluster') {
      key = point.cluster;
    } else if (colorBy === 'cellType') {
      key = point.cellType;
    } else if (colorBy === 'metadata' && colorKey && point.metadata) {
      key = point.metadata[colorKey];
    }

    if (key !== undefined) {
      return colorMap.get(key) || '#94a3b8';
    }

    return '#94a3b8';
  }, [colorBy, colorKey, colorMap]);

  // 坐标转换：数据坐标 -> 画布坐标
  const dataToCanvas = useCallback(
    (x: number, y: number) => {
      const { minX, maxX, minY, maxY } = bounds;
      const padding = 40;

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const canvasX = padding + ((x - minX) / rangeX) * (width - 2 * padding);
      const canvasY = padding + ((y - minY) / rangeY) * (height - 2 * padding);

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
      const padding = 40;

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;

      const dataX = minX + ((cx - transform.offsetX) / transform.scale - padding) / (width - 2 * padding) * rangeX;
      const dataY = minY + ((cy - transform.offsetY) / transform.scale - padding) / (height - 2 * padding) * rangeY;

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

    // 绘制点
    for (const point of data.points) {
      const pos = dataToCanvas(point.x, point.y);
      const color = getPointColor(point);

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, pointSize, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    // 绘制高亮边框
    if (hoveredPoint) {
      const pos = dataToCanvas(hoveredPoint.x, hoveredPoint.y);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, pointSize + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [data.points, width, height, pointSize, getPointColor, dataToCanvas, hoveredPoint, transform]);

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
      let nearestPoint: UmapDataPoint | null = null;

      const threshold = pointSize + 5;

      for (const point of data.points) {
        const pos = dataToCanvas(point.x, point.y);
        const dist = Math.sqrt((mouseX - pos.x) ** 2 + (mouseY - pos.y) ** 2);

        if (dist < minDist) {
          minDist = dist;
          nearestPoint = point;
        }
      }

      if (minDist <= threshold && nearestPoint) {
        setHoveredPoint(nearestPoint);
        setTooltipPosition({ x: mouseX, y: mouseY });
        onPointHover?.(nearestPoint, nearestPoint.index ?? -1);
      } else {
        setHoveredPoint(null);
        setTooltipPosition(null);
        onPointHover?.(null, -1);
      }
    },
    [interactive, data.points, pointSize, dataToCanvas, transform, isDragging, dragStart, onPointHover]
  );

  const handleMouseLeave = useCallback(() => {
    if (!isDragging) {
      setHoveredPoint(null);
      setTooltipPosition(null);
      onPointHover?.(null, -1);
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
      onPointClick?.(hoveredPoint, hoveredPoint.index ?? -1);
    } else {
      // 开始拖拽
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

      // 以鼠标位置为中心缩放
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

  // 高亮点
  const highlightPoints = useCallback((indices: number[]) => {
    // TODO: 实现高亮逻辑
    console.log('Highlight points:', indices);
  }, []);

  // 清除高亮
  const clearHighlight = useCallback(() => {
    setHoveredPoint(null);
  }, []);

  // 暴露ref方法
  React.useImperativeHandle(ref, () => ({
    exportAsImage,
    resetZoom,
    highlightPoints,
    clearHighlight,
  }));

  // 绘制效果
  useEffect(() => {
    draw();
  }, [draw]);

  // 生成图例
  const legend = useMemo(() => {
    if (!showLegend) return null;

    const items = new Map<string, string>();

    for (const point of data.points) {
      let key: string;
      let label: string;

      if (colorBy === 'cluster') {
        key = String(point.cluster ?? 'unknown');
        label = data.clusterLabels?.[point.cluster ?? 0] ?? `Cluster ${point.cluster}`;
      } else if (colorBy === 'cellType' && point.cellType) {
        key = point.cellType;
        label = point.cellType;
      } else if (colorBy === 'metadata' && colorKey) {
        key = String(point.metadata?.[colorKey] ?? 'unknown');
        label = key;
      } else {
        continue;
      }

      if (!items.has(key)) {
        items.set(key, getPointColor(point));
      }
    }

    return (
      <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg p-3 shadow-lg">
        <div className="text-xs font-semibold text-slate-700 mb-2">
          {colorBy === 'cluster' ? '聚类' : colorBy === 'cellType' ? '细胞类型' : '元数据'}
        </div>
        <div className="space-y-1">
          {Array.from(items.entries()).map(([key, color]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-slate-600">{key}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }, [showLegend, data.points, data.clusterLabels, colorBy, colorKey, getPointColor]);

  return (
    <div
      ref={containerRef}
      className={`relative bg-slate-50 rounded-xl border border-slate-200 ${className}`}
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
          onClick={async () => {
            const blob = await exportAsImage();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `umap_${Date.now()}.png`;
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
      {legend}

      {/* 提示工具 */}
      {hoveredPoint && tooltipPosition && (
        <div
          className="absolute bg-slate-900/90 text-white text-xs p-2 rounded-lg pointer-events-none z-10"
          style={{
            left: tooltipPosition.x + 15,
            top: tooltipPosition.y - 10,
          }}
        >
          {getTooltipContent ? (
            getTooltipContent(hoveredPoint)
          ) : (
            <>
              <div className="font-semibold">
                {colorBy === 'cluster' && `Cluster: ${hoveredPoint.cluster}`}
                {colorBy === 'cellType' && hoveredPoint.point.cellType}
              </div>
              <div className="text-slate-300">
                ({hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)})
              </div>
            </>
          )}
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
      {data.points && (
        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm border border-slate-200 rounded-lg px-3 py-1.5 text-xs text-slate-600">
          {data.points.length.toLocaleString()} 个细胞
        </div>
      )}
    </div>
  );
});

UmapVisualization.displayName = 'UmapVisualization';
