import React from 'react';
import { FileText, X, Eye } from 'lucide-react';
import { formatFileSize } from '@/utils/helpers';
import type { UploadedFile } from '@/types';

interface FilePillProps {
  file: UploadedFile;
  onRemove: () => void;
  onPreview?: () => void;
}

export const FilePill: React.FC<FilePillProps> = ({ file, onRemove, onPreview }) => {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg border border-blue-100 shadow-sm">
      <FileText size={14} />
      <span className="text-xs font-bold truncate max-w-[150px]">{file.name}</span>
      <span className="text-xs text-blue-500">({formatFileSize(file.size)})</span>
      {onPreview && (
        <button
          onClick={onPreview}
          className="hover:text-blue-900 transition-colors"
          title="预览数据"
        >
          <Eye size={14} />
        </button>
      )}
      <button
        onClick={onRemove}
        className="hover:text-red-500 transition-colors"
        title="移除文件"
      >
        <X size={14} />
      </button>
    </div>
  );
};
