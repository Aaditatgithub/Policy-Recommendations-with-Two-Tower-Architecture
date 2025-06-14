import React from 'react';
import { Star, Users, TrendingUp, Calendar, DollarSign } from 'lucide-react';
import { Policy } from '../../types/promotion';

interface PolicyPromotionCardProps {
  policy: Policy;
  selected: boolean;
  onSelect: (policy: Policy) => void;
  onPromote: (policy: Policy) => void;
  viewMode?: 'grid' | 'list';
}

const PolicyPromotionCard: React.FC<PolicyPromotionCardProps> = ({
  policy,
  selected,
  onSelect,
  onPromote,
  viewMode = 'grid'
}) => {
  
  const formatCurrency = (amount: any) => {
  // Convert to number if it's a string, handle empty strings or invalid values
  const numericValue = typeof amount === 'string' ? parseFloat(amount) : amount;
  
  // Check if the value is a valid number
  if (numericValue === null || numericValue === undefined || isNaN(numericValue)) {
    return '₹0';
  }
  
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0
  }).format(numericValue);
};
const getCoverageAmount = (policy: Policy) => {
  // Try all possible coverage fields
  let coverageValue = policy.sum_assured || policy['sum_assured (INR)'] || policy.coverage;
  
  // Handle string format similar to premium
  if (typeof coverageValue === 'string') {
    const matches = coverageValue.match(/[0-9,]+/);
    if (matches) {
      coverageValue = parseFloat(matches[0].replace(/,/g, ''));
    }
  }
  
  return coverageValue;
};
  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPremiumAmount = (policy: Policy) => {
  // Try all possible premium fields
  let premiumValue = policy.premium_amount || policy['premium_amount (INR)'] || policy.premium;
  
  // If it's a string like "₹5,100/year", extract just the numeric part
  if (typeof premiumValue === 'string') {
    // Remove currency symbol, commas, and any text after numbers
    const matches = premiumValue.match(/[0-9,]+/);
    if (matches) {
      // Remove commas and convert to number
      premiumValue = parseFloat(matches[0].replace(/,/g, ''));
    }
  }
  
  return premiumValue;
};

  if (viewMode === 'list') {
    return (
      <div className={`bg-white rounded-lg border-2 p-4 transition-all duration-200 hover:shadow-md ${
        selected ? 'border-violet-500 bg-violet-50' : 'border-gray-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 flex-1">
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onSelect(policy)}
              className="w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
            />
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">{policy.policy_name}</h3>
              <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                <span className="flex items-center">
                  <DollarSign size={14} className="mr-1" />
                  {formatCurrency(getPremiumAmount(policy))}
                </span>
                <span className="flex items-center">
                  <Calendar size={14} className="mr-1" />
                  {policy.policy_duration_years} years
                </span>
                <span className={`px-2 py-1 rounded-full text-xs ${getRiskColor(policy.risk_category)}`}>
                  {policy.risk_category} Risk
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {policy.score && (
              <div className="flex items-center text-sm text-gray-600">
                <Star size={14} className="mr-1 text-yellow-500" />
                {policy.score}
              </div>
            )}
            <button
              onClick={() => onPromote(policy)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                policy.selected_for_promotion
                  ? 'bg-green-100 text-green-800 hover:bg-green-200'
                  : 'bg-violet-100 text-violet-800 hover:bg-violet-200'
              }`}
            >
              {policy.selected_for_promotion ? 'Promoting' : 'Promote'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-xl border-2 p-6 transition-all duration-200 hover:shadow-lg ${
      selected ? 'border-violet-500 bg-violet-50' : 'border-gray-200'
    }`}>
      <div className="flex items-start justify-between mb-4">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onSelect(policy)}
          className="w-4 h-4 text-violet-600 rounded focus:ring-violet-500 mt-1"
        />
        <div className="flex items-center space-x-2">
          {policy.score && (
            <div className="flex items-center bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs">
              <Star size={12} className="mr-1" />
              {policy.score}
            </div>
          )}
          <span className={`px-2 py-1 rounded-full text-xs ${getRiskColor(policy.risk_category)}`}>
            {policy.risk_category} Risk
          </span>
        </div>
      </div>

      <h3 className="font-bold text-lg text-gray-900 mb-2 line-clamp-2">
        {policy.policy_name}
      </h3>

      <p className="text-sm text-violet-600 font-medium mb-3">
        {policy.policy_type}
      </p>

      <p className="text-gray-600 text-sm mb-4 line-clamp-2">
        {policy.description}
      </p>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Premium</p>
          <p className="font-bold text-sm">{formatCurrency(getPremiumAmount(policy))}</p>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="text-xs text-gray-500 mb-1">Coverage</p>
          <p className="font-bold text-sm">{formatCurrency(getCoverageAmount(policy))}</p>
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center text-sm text-gray-600">
          <Users size={14} className="mr-1" />
          <span>{Array.isArray(policy.customer_target_group) ? policy.customer_target_group.join(', ') : policy.customer_target_group}</span>
        </div>
        <div className="flex items-center text-sm text-gray-600">
          <Calendar size={14} className="mr-1" />
          <span>{policy.policy_duration_years}y</span>
        </div>
      </div>

      <div className="flex space-x-2">
        <button
          onClick={() => onSelect(policy)}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            selected
              ? 'bg-violet-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {selected ? 'Selected' : 'Select'}
        </button>
        <button
          onClick={() => onPromote(policy)}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            policy.selected_for_promotion
              ? 'bg-green-100 text-green-800 hover:bg-green-200'
              : 'bg-violet-100 text-violet-800 hover:bg-violet-200'
          }`}
        >
          {policy.selected_for_promotion ? 'Promoting' : 'Promote'}
        </button>
      </div>
    </div>
  );
};

export default PolicyPromotionCard;