import React, { useState } from 'react';
import { Maximize2, Download, Info } from 'lucide-react';

export interface PlotInterpretation {
  title: string;
  description: string;
  what_to_look: string[];
  biological_meaning?: string;
  technical_notes?: string;
}

export interface PlotItem {
  name: string;
  title: string;
  path: string;
  local_path?: string;
  type?: string;
  interpretation: PlotInterpretation;
  run_id?: string;
}

interface PlotCardProps {
  plot: PlotItem;
}

const PlotCard: React.FC<PlotCardProps> = ({ plot }) => {
  const [imageError, setImageError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showInterpretation, setShowInterpretation] = useState(false);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = plot.path;
    link.download = `${plot.name}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleImageLoad = () => {
    setIsLoading(false);
  };

  const handleImageError = () => {
    setImageError(true);
    setIsLoading(false);
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
        <h3 className="font-semibold text-slate-800 text-sm">{plot.title}</h3>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowInterpretation(!showInterpretation)}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
            title="显示解读"
          >
            <Info size={16} />
          </button>
          <button
            onClick={handleDownload}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
            title="下载图表"
          >
            <Download size={16} />
          </button>
        </div>
      </div>

      {/* Image */}
      <div className="relative bg-slate-50 aspect-video flex items-center justify-center">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-100">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        )}
        {imageError ? (
          <div className="text-center p-4">
            <p className="text-slate-500 text-sm">图表加载失败</p>
            <p className="text-slate-400 text-xs mt-1">{plot.name}</p>
          </div>
        ) : (
          <img
            src={plot.path}
            alt={plot.title}
            className="max-w-full max-h-full object-contain"
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        )}
      </div>

      {/* Interpretation */}
      {showInterpretation && plot.interpretation && (
        <div className="px-4 py-3 bg-slate-50 border-t border-slate-100">
          <p className="text-xs text-slate-600 mb-2">{plot.interpretation.description}</p>
          {plot.interpretation.what_to_look && plot.interpretation.what_to_look.length > 0 && (
            <div>
              <p className="text-xs font-medium text-slate-700 mb-1">观察要点：</p>
              <ul className="text-xs text-slate-600 space-y-0.5">
                {plot.interpretation.what_to_look.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-1">
                    <span className="text-blue-500">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface PlotGalleryProps {
  plots: PlotItem[];
  title?: string;
}

export const PlotGallery: React.FC<PlotGalleryProps> = ({ plots, title }) => {
  if (plots.length === 0) {
    return (
      <div className="text-center py-8 bg-slate-50 rounded-xl border border-slate-200">
        <p className="text-slate-500">暂无图表</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {title && (
        <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <span>{title}</span>
          <span className="text-sm font-normal text-slate-500">
            ({plots.length} 个图表)
          </span>
        </h3>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {plots.map((plot) => (
          <PlotCard key={plot.name} plot={plot} />
        ))}
      </div>
    </div>
  );
};

// Lightweight version for inline display in messages
interface InlinePlotProps {
  plot: PlotItem;
}

export const InlinePlot: React.FC<InlinePlotProps> = ({ plot }) => {
  const [imageError, setImageError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  return (
    <div className="my-4 rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-3 py-2 bg-slate-50 border-b border-slate-100 flex items-center justify-between">
        <span className="font-medium text-slate-700 text-sm">{plot.title}</span>
        <a
          href={plot.path}
          download={`${plot.name}.png`}
          className="text-xs text-blue-600 hover:text-blue-700"
        >
          下载
        </a>
      </div>
      <div className="relative bg-white p-2 min-h-[200px] flex items-center justify-center">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          </div>
        )}
        {imageError ? (
          <p className="text-slate-400 text-sm">图表加载失败</p>
        ) : (
          <img
            src={plot.path}
            alt={plot.title}
            className="max-w-full max-h-[400px] object-contain"
            onLoad={() => setIsLoading(false)}
            onError={() => {
              setImageError(true);
              setIsLoading(false);
            }}
          />
        )}
      </div>
      {plot.interpretation?.description && (
        <div className="px-3 py-2 bg-slate-50 text-xs text-slate-600">
          {plot.interpretation.description}
        </div>
      )}
    </div>
  );
};

export default PlotGallery;
