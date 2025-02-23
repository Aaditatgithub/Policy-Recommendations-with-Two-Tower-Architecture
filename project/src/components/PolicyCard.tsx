import React from "react";
import { Shield, Clock, IndianRupee } from "lucide-react";

interface PolicyCardProps {
  policy: {
    policy_id: number;
    policy_name: string;
    policy_type: string;
    "premium_amount (INR)"?: number; // Correct key mapping
    "sum_assured (INR)"?: number; // Correct key mapping
    policy_duration_years: number;
    description: string;
    risk_category: string;
    customer_target_group: string;
  };
}

const PolicyCard: React.FC<PolicyCardProps> = ({ policy }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
      <h3 className="text-xl font-semibold text-sbi-purple mb-4">
        {policy.policy_name}
      </h3>

      <div className="space-y-3 mb-4">
        <div className="flex items-center text-gray-700">
          <Shield size={18} className="mr-2 text-sbi-blue" />
          <span>Type: {policy.policy_type}</span>
        </div>
        <div className="flex items-center text-gray-700">
          <IndianRupee size={18} className="mr-2 text-sbi-pink" />
          <span>
            Premium: ₹
            {policy["premium_amount (INR)"]
              ? policy["premium_amount (INR)"].toLocaleString()
              : "N/A"}
          </span>
        </div>
        <div className="flex items-center text-gray-700">
          <Shield size={18} className="mr-2 text-sbi-green" />
          <span>
            Sum Assured: ₹
            {policy["sum_assured (INR)"]
              ? policy["sum_assured (INR)"].toLocaleString()
              : "N/A"}
          </span>
        </div>
        <div className="flex items-center text-gray-700">
          <Clock size={18} className="mr-2 text-sbi-purple" />
          <span>Duration: {policy.policy_duration_years} years</span>
        </div>
      </div>

      <p className="text-gray-600 mb-4">{policy.description}</p>

      <div className="flex flex-wrap gap-2 mb-4">
        {policy.risk_category && (
          <span className="px-3 py-1 bg-sbi-light-purple text-sbi-purple rounded-full text-sm">
            {policy.risk_category}
          </span>
        )}
        {policy.customer_target_group && (
          <span className="px-3 py-1 bg-sbi-light-pink text-sbi-pink rounded-full text-sm">
            {policy.customer_target_group}
          </span>
        )}
      </div>

      <button className="w-full bg-sbi-purple text-white py-2 rounded-lg hover:bg-sbi-pink transition-colors duration-200">
        View Details
      </button>
    </div>
  );
};

export default PolicyCard;
