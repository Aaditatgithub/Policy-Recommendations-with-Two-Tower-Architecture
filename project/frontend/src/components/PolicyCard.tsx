// src/components/PolicyCard.tsx
import React, { useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { trackPolicyInteraction, trackElementView } from '../utils/tracking';

interface Policy {
  policy_id: string;
  policy_name: string;
  policy_type: string;
  description: string;
  premium_amount?: number | string;
  sum_assured?: number | string;
  policy_duration_years?: number | string;
  keywords?: string;
  risk_category?: string;
  customer_target_group?: string;
  // Add CSV column names exactly as they appear
  'premium_amount (INR)'?: number | string;
  'sum_assured (INR)'?: number | string;
  [key: string]: any;
}

interface PolicyCardProps {
  policy: Policy;
  interactionContext?: string;
  className?: string;
}

const PolicyCard: React.FC<PolicyCardProps> = ({ 
  policy, 
  interactionContext = 'browse',
  className = ''
}) => {
  const { id: customerId } = useParams<{ id: string }>();
  const cardRef = useRef<HTMLDivElement>(null);
  const cardId = `policy-card-${policy?.policy_id || 'unknown'}`;

  // Get values from either direct or CSV-style properties
  const getPolicyValue = (key: string) => {
    return policy[key] || policy[`${key} (INR)`] || null;
  };

  const formatValue = (value?: number | string, isCurrency = true) => {
    if (value === undefined || value === null) return isCurrency ? '₹---' : '--';
    
    let num: number;
    if (typeof value === 'string') {
      const cleaned = value.replace(/,/g, '').replace(/[^\d.-]/g, '');
      num = parseFloat(cleaned);
    } else {
      num = value;
    }
    
    if (isNaN(num)) return isCurrency ? '₹---' : '--';
    
    return isCurrency 
      ? '₹' + num.toLocaleString('en-IN', { maximumFractionDigits: 0 })
      : num.toString() + ' yrs';
  };

  const getKeywords = () => {
    if (!policy?.keywords) return [];
    return policy.keywords.split(',').map(k => k.trim()).filter(k => k.length > 0);
  };

  useEffect(() => {
    if (!cardRef.current || !policy?.policy_id) return;
    const cleanup = trackElementView(cardId, customerId || 'guest', policy.policy_id);
    return cleanup;
  }, [cardId, customerId, policy?.policy_id]);

  const handleClick = () => {
    if (!policy?.policy_id) return;
    trackPolicyInteraction(
      customerId || 'guest',
      policy.policy_id,
      'click',
      undefined,
      { context: interactionContext }
    );
  };

  // Get actual values from either property naming convention
  const premiumAmount = getPolicyValue('premium_amount');
  const sumAssured = getPolicyValue('sum_assured');
  const duration = policy.policy_duration_years;

  if (!policy) {
    return <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>Loading...</div>;
  }

  return (
    <div 
      ref={cardRef}
      id={cardId}
      className={`bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-all cursor-pointer ${className}`}
      onClick={handleClick}
    >
      <div className={`px-6 py-3 ${
        policy.risk_category === 'High' ? 'bg-red-100 text-red-800' :
        policy.risk_category === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
        'bg-green-100 text-green-800'
      }`}>
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">
            {policy.risk_category || 'Standard'} Risk
          </span>
          {policy.customer_target_group && (
            <span className="text-xs bg-white px-2 py-1 rounded-full">
              {policy.customer_target_group}
            </span>
          )}
        </div>
      </div>

      <div className="p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-2 line-clamp-1">
          {policy.policy_name || 'Unnamed Policy'}
        </h3>
        <p className="text-sm text-violet-600 font-medium mb-3">
          {policy.policy_type || 'Insurance Policy'}
        </p>
        <p className="text-gray-700 mb-4 line-clamp-2">
          {policy.description || 'No description available.'}
        </p>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-xs text-gray-500">Annual Premium</p>
            <p className="font-bold text-lg">
              {formatValue(premiumAmount)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Sum Assured</p>
            <p className="font-bold text-lg">
              {formatValue(sumAssured)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Duration</p>
            <p className="font-bold text-lg">
              {formatValue(duration, false)}
            </p>
          </div>
          {/* <div>
            <p className="text-xs text-gray-500">Type</p>
            <p className="font-bold text-lg">
              {policy.policy_type || '--'}
            </p>
          </div> */}
        </div>

        {getKeywords().length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {getKeywords().map((keyword, index) => (
              <span key={index} className="text-xs bg-violet-100 text-violet-800 px-2 py-1 rounded-full">
                {keyword}
              </span>
            ))}
          </div>
        )}

        <div className="flex space-x-3 pt-2 border-t border-gray-100">
          <button 
            className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              trackPolicyInteraction(
                customerId || 'guest',
                policy.policy_id,
                'cart_add',
                undefined,
                { context: interactionContext }
              );
            }}
          >
            Add to Cart
          </button>
          <button 
            className="px-4 py-2 border border-violet-600 text-violet-600 rounded-lg hover:bg-violet-50 transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              trackPolicyInteraction(
                customerId || 'guest',
                policy.policy_id,
                'compare',
                undefined,
                { context: interactionContext }
              );
            }}
          >
            Compare
          </button>
        </div>
      </div>
    </div>
  );
};

export default PolicyCard;