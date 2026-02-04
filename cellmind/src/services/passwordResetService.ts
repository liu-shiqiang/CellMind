const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SendCodeResponse {
  success: boolean;
  message: string;
  expires_in: number;
  _dev_code?: string; // Only available in development
}

interface VerifyCodeResponse {
  success: boolean;
  message: string;
}

interface ResetPasswordResponse {
  success: boolean;
  message: string;
}

class PasswordResetService {
  async sendCode(email: string): Promise<SendCodeResponse> {
    const response = await fetch(`${API_BASE_URL}/api/auth/password-reset/send-code`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to send verification code');
    }

    return response.json();
  }

  async verifyCode(email: string, code: string): Promise<VerifyCodeResponse> {
    const response = await fetch(`${API_BASE_URL}/api/auth/password-reset/verify-code`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, code })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to verify code');
    }

    return response.json();
  }

  async resetPassword(email: string, code: string, newPassword: string): Promise<ResetPasswordResponse> {
    const response = await fetch(`${API_BASE_URL}/api/auth/password-reset/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, code, new_password: newPassword })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to reset password');
    }

    return response.json();
  }
}

export const passwordResetService = new PasswordResetService();
