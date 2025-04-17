// src/components/ComparisonModal.tsx
import React from 'react';
import { FiX } from 'react-icons/fi';

const ComparisonModal = ({ policies, onClose }) => {
  const features = [
    'policy_type', 
    'premium_amount',
    'sum_assured',
    'policy_duration_years',
    'risk_category'
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-auto">
        <div className="p-4 border-b flex justify-between items-center sticky top-0 bg-white">
          <h3 className="text-xl font-semibold">Compare Policies</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <FiX size={24} />
          </button>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="p-4 text-left min-w-48">Feature</th>
                {policies.map(policy => (
                  <th key={policy.policy_id} className="p-4 text-center min-w-64">
                    {policy.policy_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {features.map(feature => (
                <tr key={feature} className="border-b">
                  <td className="p-4 font-medium capitalize">
                    {feature.replace('_', ' ')}
                  </td>
                  {policies.map(policy => (
                    <td key={`${policy.policy_id}-${feature}`} className="p-4 text-center">
                      {policy[feature] || '--'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="p-4 border-t flex justify-end">
          <button 
            onClick={onClose}
            className="px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700"
          >
            Close Comparison
          </button>
        </div>
      </div>
    </div>
  );
};

export default ComparisonModal;