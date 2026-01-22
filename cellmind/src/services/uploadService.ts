import api from './api';
import type { UploadedFile } from '@/types';

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  size: number;
  content_type: string;
  created_at: string;
  validation?: unknown;
}

export const uploadService = {
  /**
   * 上传文件
   */
  async uploadFile(file: File): Promise<UploadedFile> {
    const formData = new FormData();
    formData.append('file', file);

    // 对于 FormData，需要移除默认的 Content-Type，让 axios 自动设置 multipart/form-data
    const response: FileUploadResponse = await api.post('/upload/', formData, {
      headers: {
        'Content-Type': undefined, // 重要：让 axios 自动设置正确的 boundary
      },
      timeout: 300000, // 5分钟超时，用于大文件上传
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          console.log(`Upload progress: ${percentCompleted}%`);
        }
      },
    });

    return {
      id: response.file_id,
      name: response.filename,
      size: response.size,
      path: `/uploads/${response.file_id}`, // 根据后端规则构造路径
      uploadedAt: new Date(response.created_at),
    };
  },

  /**
   * 验证文件
   */
  validateFile(file: File): { valid: boolean; error?: string } {
    const validExtensions = ['.h5ad', '.h5ad'];
    const maxSize = 500 * 1024 * 1024; // 500MB

    const hasValidExtension = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExtension) {
      return { valid: false, error: '仅支持.h5ad格式的文件' };
    }

    if (file.size > maxSize) {
      return { valid: false, error: '文件大小不能超过500MB' };
    }

    return { valid: true };
  },

  /**
   * 格式化文件大小
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  },
};
