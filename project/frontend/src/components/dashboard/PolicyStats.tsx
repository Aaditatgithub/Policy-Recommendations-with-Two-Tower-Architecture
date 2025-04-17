// src/components/dashboard/PolicyStats.tsx
import React from 'react';
import { FiTrendingUp, FiTrendingDown, FiDollarSign } from 'react-icons/fi';

interface PolicyStatsProps {
  topPolicies: {
    id: string;
    name: string;
    views: number;
    conversions: number;
  }[];
}

export const PolicyStats = ({ topPolicies = [] }: PolicyStatsProps) => {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <h3 className="font-semibold mb-4">Top Performing Policies</h3>
      <div className="space-y-4">
        {topPolicies.map((policy) => (
          <div key={policy.id} className="flex items-center">
            <div className="flex-1">
              <p className="font-medium">{policy.name}</p>
              <div className="flex items-center text-sm text-gray-600">
                <span className="mr-2">{policy.views.toLocaleString()} views</span>
                <span className="flex items-center">
                  {policy.conversions > 0 ? (
                    <FiTrendingUp className="text-green-500 mr-1" />
                  ) : (
                    <FiTrendingDown className="text-red-500 mr-1" />
                  )}
                  {policy.conversions}%
                </span>
              </div>
            </div>
            <div className="bg-blue-50 p-2 rounded-lg">
              <FiDollarSign className="text-blue-500" />
            </div>
          </div>
        ))}
        {topPolicies.length === 0 && (
          <p className="text-center py-4 text-gray-500">No policy data available</p>
        )}
      </div>
    </div>
  );
};