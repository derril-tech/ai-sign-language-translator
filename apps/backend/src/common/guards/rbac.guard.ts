import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { newEnforcer } from 'casbin';
import { ROLES_KEY } from '../decorators/roles.decorator';
import * as path from 'path';

@Injectable()
export class RBACGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredRoles) {
      return true;
    }

    const request = context.switchToHttp().getRequest();
    const user = request.user;

    if (!user) {
      return false;
    }

    // Initialize Casbin enforcer
    const enforcer = await newEnforcer(
      path.join(__dirname, '../../config/casbin.conf'),
    );

    // Check if user has required role in their organization domain
    const domain = user.orgId || 'default';
    
    for (const role of requiredRoles) {
      const hasPermission = await enforcer.enforce(
        user.id,
        request.route?.path || request.url,
        request.method,
        domain,
      );
      
      if (hasPermission) {
        return true;
      }
    }

    return false;
  }
}
