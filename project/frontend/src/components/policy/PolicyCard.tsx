import React from 'react';
import { Policy } from '../../data/policies';
import PolicyDetails from './PolicyDetails';
import PolicyTags from './PolicyTags';

interface PolicyCardProps {
  policy: Policy;
}

const PolicyCard: React.FC<PolicyCardProps> = ({ policy }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
      <h3 className="text-xl font-semibold text-sbi-purple mb-4">{policy.name}</h3>
      <PolicyDetails policy={policy} />
      <p className="text-gray-600 mb-4">{policy.description}</p>
      <PolicyTags tags={policy.tags} />
      <button className="w-full bg-sbi-purple text-white py-2 rounded-lg hover:bg-sbi-pink transition-colors duration-200">
        View Details
      </button>
    </div>
  );
};

export default PolicyCard;