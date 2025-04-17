// src/components/dashboard/CustomerSegments.tsx
import React from 'react';
import { FiUsers, FiPieChart } from 'react-icons/fi';

interface Segment {
  id: string;
  name: string;
  value: number;
  color: string;
}

interface CustomerSegmentsProps {
  segments: Segment[];
}

export const CustomerSegments = ({ segments = [] }: CustomerSegmentsProps) => {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0);

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <div className="flex items-center mb-4">
        <FiUsers className="text-gray-500 mr-2" />
        <h3 className="font-semibold">Customer Segments</h3>
      </div>
      
      <div className="space-y-3">
        {segments.map((segment) => (
          <div key={segment.id} className="flex items-center">
            <div className="w-8 h-8 rounded-full mr-3" style={{ backgroundColor: segment.color }} />
            <div className="flex-1">
              <div className="flex justify-between text-sm">
                <span>{segment.name}</span>
                <span>{Math.round((segment.value / total) * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div 
                  className="h-2 rounded-full" 
                  style={{ 
                    width: `${(segment.value / total) * 100}%`,
                    backgroundColor: segment.color
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {segments.length === 0 && (
        <div className="text-center py-6 text-gray-500">
          <FiPieChart className="mx-auto text-2xl mb-2" />
          <p>No segment data available</p>
        </div>
      )}
    </div>
  );
};