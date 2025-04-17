// src/components/dashboard/RecentActivity.tsx
import React from 'react';
import { FiShoppingCart, FiEye, FiDollarSign, FiUser } from 'react-icons/fi';

interface Activity {
  id: string;
  type: 'purchase' | 'view' | 'payment' | 'signup';
  customer: string;
  policy: string;
  time: string;
}

interface RecentActivityProps {
  activities: Activity[];
}

const iconMap = {
  purchase: <FiShoppingCart className="text-green-500" />,
  view: <FiEye className="text-blue-500" />,
  payment: <FiDollarSign className="text-purple-500" />,
  signup: <FiUser className="text-orange-500" />
};

export const RecentActivity = ({ activities = [] }: RecentActivityProps) => {
  return (
    <div className="space-y-4">
      {activities.map((activity) => (
        <div key={activity.id} className="flex items-start pb-4 border-b border-gray-100">
          <div className="p-2 rounded-full bg-gray-50 mr-3">
            {iconMap[activity.type]}
          </div>
          <div className="flex-1">
            <div className="flex justify-between">
              <p className="font-medium">{activity.customer}</p>
              <p className="text-sm text-gray-500">{activity.time}</p>
            </div>
            <p className="text-sm text-gray-600">
              {activity.type === 'purchase' ? 'Purchased' : 
               activity.type === 'view' ? 'Viewed' :
               activity.type === 'payment' ? 'Made payment for' : 'Signed up for'} 
              <span className="font-medium ml-1">{activity.policy}</span>
            </p>
          </div>
        </div>
      ))}
      {activities.length === 0 && (
        <p className="text-center py-4 text-gray-500">No recent activity</p>
      )}
    </div>
  );
};