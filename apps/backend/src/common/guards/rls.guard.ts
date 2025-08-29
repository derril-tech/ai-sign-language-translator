import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';

export const RLS_KEY = 'rls';
export const RequireRLS = (enabled: boolean = true) => 
  (target: any, propertyKey?: string, descriptor?: PropertyDescriptor) => {
    Reflector.createDecorator<boolean>()(enabled)(target, propertyKey, descriptor);
  };

@Injectable()
export class RLSGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requireRLS = this.reflector.getAllAndOverride<boolean>(RLS_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    if (requireRLS === false) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    const user = request.user;

    if (!user) {
      return false;
    }

    // Add org_id filter to query context
    request.orgFilter = {
      orgId: user.orgId || null,
    };

    return true;
  }
}
