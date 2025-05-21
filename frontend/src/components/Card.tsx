import React, { ReactNode } from 'react';

export interface CardProps {
  title: string;
  children: ReactNode;
  icon?: ReactNode;
  fullWidth?: boolean;
  className?: string;
}

const Card: React.FC<CardProps> = ({ 
  title, 
  children, 
  icon, 
  fullWidth = false,
  className = ''
}) => {
  return (
    <div className={`rounded-xl bg-white shadow-md p-4 md:p-6 ${fullWidth ? 'w-full' : ''} ${className || ''}`}>
      <div className="flex items-center mb-4 gap-2">
        {icon && (
          <div className="h-5 w-5 text-gray-500">
            {icon}
          </div>
        )}
        <h2 className="text-xl font-semibold">{title}</h2>
      </div>
      <div>
        {children}
      </div>
    </div>
  );
};

export default Card;
