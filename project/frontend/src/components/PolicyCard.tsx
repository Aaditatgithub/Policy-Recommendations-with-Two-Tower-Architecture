import React, { useEffect, useRef } from 'react';
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
  'premium_amount (INR)'?: number | string;
  'sum_assured (INR)'?: number | string;
  is_promoted?: boolean;
  promotion_tag?: string;
  [key: string]: any;
}

interface PolicyCardProps {
  policy: Policy;
  customerId?: string | null;
  onAddToCart: (policyId: string) => void;
  onCompare: (policyId: string) => void;
  interactionContext?: string;
  className?: string;
}

const PolicyCard: React.FC<PolicyCardProps> = ({ 
  policy, 
  customerId,
  onAddToCart,
  onCompare,
  interactionContext = 'browse',
  className = ''
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const cardId = `policy-card-${policy?.policy_id || 'unknown'}`;

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

  const handleAddToCart = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!policy?.policy_id) return;
    
    trackPolicyInteraction(
      customerId || 'guest',
      policy.policy_id,
      'cart_add',
      undefined,
      { context: interactionContext }
    );
    
    onAddToCart(policy.policy_id);
  };

  const handleCompare = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!policy?.policy_id) return;
    
    trackPolicyInteraction(
      customerId || 'guest',
      policy.policy_id,
      'compare',
      undefined,
      { context: interactionContext }
    );
    
    onCompare(policy.policy_id);
  };

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
      className={`bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-all cursor-pointer relative ${className}`}
      onClick={handleClick}
    >
      {/* Promotion Badge */}
      {policy.is_promoted && (
        <div className="absolute top-0 right-0 z-10">
          <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-3 py-1 rounded-bl-lg rounded-tr-lg shadow-lg">
            <div className="flex items-center space-x-1">
              <span className="text-xs font-bold">⭐</span>
              <span className="text-xs font-semibold uppercase tracking-wide">
                {policy.promotion_tag || 'Featured'}
              </span>
            </div>
          </div>
        </div>
      )}

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
        </div>

        {getKeywords().length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {getKeywords().slice(0, 3).map((keyword, index) => (
              <span key={index} className="text-xs bg-violet-100 text-violet-800 px-2 py-1 rounded-full">
                {keyword}
              </span>
            ))}
          </div>
        )}

        <div className="flex space-x-3 pt-2 border-t border-gray-100">
          <button 
            className={`flex-1 px-4 py-2 rounded-lg transition-colors ${
              policy.is_promoted 
                ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white hover:from-yellow-600 hover:to-orange-600' 
                : 'bg-violet-600 text-white hover:bg-violet-700'
            }`}
            onClick={handleAddToCart}
          >
            {policy.is_promoted ? '⭐ Add Featured' : 'Add to Cart'}
          </button>
          <button 
            className="px-4 py-2 border border-violet-600 text-violet-600 rounded-lg hover:bg-violet-50 transition-colors"
            onClick={handleCompare}
          >
            Compare
          </button>
        </div>
      </div>
    </div>
  );
};

export default PolicyCard;