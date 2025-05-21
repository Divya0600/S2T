import React from 'react';

export interface StatusCardProps {
  icon: React.ReactNode;
  title: string;
  value: string;
  className?: string;
}

const StatusCard: React.FC<StatusCardProps> = ({ 
  icon, 
  title, 
  value, 
  className 
}) => {
  return (
    <div className={`bg-white rounded-lg shadow-md p-4 ${className || ''}`}>
      <div className="flex items-center mb-2">
        <div className="w-5 h-5 text-[#5236ab] mr-2">
          {icon}
        </div>
        <h3 className="text-lg font-medium text-gray-800">{title}</h3>
      </div>
      <p className="text-gray-600">{value}</p>
    </div>
  );
};

export default StatusCard;
