import React from 'react';
import { Shield, Clock, IndianRupee } from 'lucide-react';
import { Policy } from '../../data/policies';

interface PolicyDetailsProps {
  policy: Policy;
}

const PolicyDetails: React.FC<PolicyDetailsProps> = ({ policy }) => {
  return (
    <div className="space-y-3 mb-4">
      <div className="flex items-center text-gray-700">
        <IndianRupee size={18} className="mr-2 text-sbi-pink" />
        <span>Premium: {policy.premium}</span>
      </div>
      <div className="flex items-center text-gray-700">
        <Shield size={18} className="mr-2 text-sbi-blue" />
        <span>Coverage: {policy.coverage}</span>
      </div>
      <div className="flex items-center text-gray-700">
        <Clock size={18} className="mr-2 text-sbi-purple" />
        <span>Duration: {policy.duration}</span>
      </div>
    </div>
  );
};

export default PolicyDetails;