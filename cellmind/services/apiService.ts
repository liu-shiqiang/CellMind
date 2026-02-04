const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

class ApiService {
  private getAuthHeaders() {
    const token = localStorage.getItem('auth_token');
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    };
  }

  // Session APIs
  async updateSession(sessionId: string, title: string): Promise<Session> {
    const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      method: 'PUT',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({ title })
    });

    if (!response.ok) {
      throw new Error('Failed to update session');
    }

    return response.json();
  }

  async deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this.getAuthHeaders()
    });

    if (!response.ok) {
      throw new Error('Failed to delete session');
    }
  }

  // File upload API
  async uploadFile(file: File, sessionId: string): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      headers: {
        ...(localStorage.getItem('auth_token') && { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` })
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error('Failed to upload file');
    }

    return response.json();
  }
}

export const apiService = new ApiService();
