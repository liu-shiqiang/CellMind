import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore, getRefreshToken } from '@/stores';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器 - 自动添加 Authorization header
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = useAuthStore.getState().token;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器 - 处理 Token 刷新
api.interceptors.response.use(
  (response) => response.data,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    // 如果是 401 错误且不是登录/注册请求，尝试刷新 token
    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      !originalRequest.url?.includes('/auth/login') &&
      !originalRequest.url?.includes('/auth/register')
    ) {
      originalRequest._retry = true;

      const refreshToken = getRefreshToken();
      if (refreshToken) {
        try {
          const { authService } = await import('@/services');
          const { access_token } = await authService.refreshToken(refreshToken);

          // 更新 store 中的 token
          useAuthStore.setState({ token: access_token });

          // 重试原始请求
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          // 刷新失败，登出用户
          useAuthStore.getState().logout();
          window.location.href = '/';
          return Promise.reject(refreshError);
        }
      } else {
        // 没有 refresh token，直接登出
        useAuthStore.getState().logout();
      }
    }

    return Promise.reject(error);
  }
);

export default api;
