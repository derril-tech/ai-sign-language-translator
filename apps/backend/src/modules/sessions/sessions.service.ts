import { Injectable } from '@nestjs/common';

@Injectable()
export class SessionsService {
  async getSessions() {
    return {
      sessions: [],
      total: 0,
    };
  }

  async createSession() {
    return {
      id: 'session_' + Date.now(),
      status: 'created',
      createdAt: new Date().toISOString(),
    };
  }
}
