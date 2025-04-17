import React from 'react';
import { LineChart, ArrowUp } from 'lucide-react';
import { conversionMetrics } from '../../data/analytics';

const ConversionMetrics: React.FC = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-2 mb-6">
        <LineChart className="text-blue-600" />
        <h2 className="text-xl font-semibold">Conversion Metrics</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-4 bg-green-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Recommendation Success</p>
          <div className="flex items-center">
            <span className="text-2xl font-bold text-green-600">
              {conversionMetrics.recommendationSuccess}%
            </span>
            <ArrowUp className="text-green-600 ml-2" size={20} />
          </div>
        </div>

        <div className="p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Upsell Revenue</p>
          <p className="text-2xl font-bold text-blue-600">
            ₹{conversionMetrics.upsellRevenue}
          </p>
        </div>

        <div className="p-4 bg-purple-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">New Customer Revenue</p>
          <p className="text-2xl font-bold text-purple-600">
            ₹{conversionMetrics.newCustomerRevenue}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConversionMetrics;