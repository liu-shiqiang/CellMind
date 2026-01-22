import React from 'react';
import { cn } from '@/utils/helpers';

interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

export const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  ({ className, size = 'md', children, ...props }, ref) => {
    const sizeClasses = {
      sm: 'w-8 h-8 text-xs',
      md: 'w-10 h-10 text-sm',
      lg: 'w-12 h-12 text-base',
    };

    return (
      <div
        ref={ref}
        className={cn(
          'rounded-full flex items-center justify-center font-bold',
          sizeClasses[size],
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Avatar.displayName = 'Avatar';
