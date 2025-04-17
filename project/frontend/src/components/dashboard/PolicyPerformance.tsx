import React from 'react';
import { BarChart, TrendingUp } from 'lucide-react';
import { policyPerformance } from '../../data/analytics';

const PolicyPerformance: React.FC = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-2 mb-6">
        <BarChart className="text-blue-600" />
        <h2 className="text-xl font-semibold">Policy Performance</h2>
      </div>
      
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium mb-4">Top Performing Policies</h3>
          <div className="space-y-4">
            {policyPerformance.topPolicies.map((policy) => (
              <div key={policy.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium">{policy.name}</p>
                  <p className="text-sm text-gray-600">Revenue: {policy.revenue}</p>
                </div>
                <div className="flex items-center text-green-600">
                  <TrendingUp size={16} className="mr-1" />
                  {policy.growth}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PolicyPerformance;