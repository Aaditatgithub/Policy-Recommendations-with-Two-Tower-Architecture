import React from 'react';
import { Users, AlertTriangle } from 'lucide-react';
import { customerSegments } from '../../data/analytics';

const CustomerInsights: React.FC = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-2 mb-6">
        <Users className="text-blue-600" />
        <h2 className="text-xl font-semibold">Customer Insights</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-medium mb-4">Customer Segments</h3>
          <div className="space-y-3">
            {customerSegments.distribution.map((segment) => (
              <div key={segment.segment} className="flex justify-between items-center">
                <span>{segment.segment}</span>
                <div className="flex items-center">
                  <div className="w-32 h-2 bg-gray-200 rounded-full mr-2">
                    <div
                      className="h-full bg-blue-600 rounded-full"
                      style={{ width: `${segment.percentage}%` }}
                    />
                  </div>
                  <span className="text-sm">{segment.percentage}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-4">Churn Risk Analysis</h3>
          <div className="p-4 bg-red-50 rounded-lg">
            <div className="flex items-center gap-2 text-red-600 mb-3">
              <AlertTriangle size={20} />
              <span className="font-medium">High Risk Customers: {customerSegments.churnRisk.highRisk}%</span>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Top Reasons:</p>
              <ul className="text-sm text-gray-600 list-disc list-inside">
                {customerSegments.churnRisk.topReasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CustomerInsights;