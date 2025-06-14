import React, { useState } from 'react';
import { X } from 'lucide-react';

interface PolicyCreationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (policyData: any) => void;
}

const PolicyCreationModal: React.FC<PolicyCreationModalProps> = ({
  isOpen,
  onClose,
  onSave
}) => {
  const [formData, setFormData] = useState({
    policy_name: '',
    policy_type: 'Term',
    sum_assured: '',
    premium_amount: '',
    policy_duration_years: '',
    risk_category: 'Low',
    customer_target_group: '',
    description: '',
    keywords: ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSave = () => {
    // Validate required fields
    const requiredFields = ['policy_name', 'policy_type', 'sum_assured', 'premium_amount', 'policy_duration_years', 'description'];
    const missingFields = requiredFields.filter(field => !formData[field as keyof typeof formData]);

    if (missingFields.length > 0) {
      alert(`Please fill in the following required fields: ${missingFields.join(', ')}`);
      return;
    }

    // Validate numeric fields
    if (isNaN(Number(formData.sum_assured)) || Number(formData.sum_assured) <= 0) {
      alert('Sum Assured must be a valid positive number');
      return;
    }

    if (isNaN(Number(formData.premium_amount)) || Number(formData.premium_amount) <= 0) {
      alert('Premium Amount must be a valid positive number');
      return;
    }

    if (isNaN(Number(formData.policy_duration_years)) || Number(formData.policy_duration_years) <= 0) {
      alert('Policy Duration must be a valid positive number');
      return;
    }

    onSave(formData);

    // Reset form
    setFormData({
      policy_name: '',
      policy_type: 'Term',
      sum_assured: '',
      premium_amount: '',
      policy_duration_years: '',
      risk_category: 'Low',
      customer_target_group: '',
      description: '',
      keywords: ''
    });
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold">Create New Policy</h3>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
              <X size={24} />
            </button>
          </div>
        </div>

        <div className="p-6 space-y-4">
          {/* Policy Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Policy Name *
            </label>
            <input
              type="text"
              name="policy_name"
              value={formData.policy_name}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="e.g., SBI Life Smart Protection Plus"
            />
          </div>

          {/* Policy Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Policy Type *
            </label>
            <select
              name="policy_type"
              value={formData.policy_type}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value="Term">Term Insurance</option>
              <option value="ULIP">ULIP</option>
              <option value="Health">Health Insurance</option>
              <option value="Endowment">Endowment</option>
              <option value="Money Back">Money Back</option>
              <option value="Pension">Pension</option>
              <option value="Child Plan">Child Plan</option>
              <option value="Annuity">Annuity</option>
            </select>
          </div>

          {/* Sum Assured and Premium Amount */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sum Assured (INR) *
              </label>
              <input
                type="number"
                name="sum_assured"
                value={formData.sum_assured}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                placeholder="e.g., 1000000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Premium Amount (INR) *
              </label>
              <input
                type="number"
                name="premium_amount"
                value={formData.premium_amount}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                placeholder="e.g., 12000"
              />
            </div>
          </div>

          {/* Duration and Risk Category */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Policy Duration (Years) *
              </label>
              <input
                type="number"
                name="policy_duration_years"
                value={formData.policy_duration_years}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                placeholder="e.g., 20"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Risk Category *
              </label>
              <select
                name="risk_category"
                value={formData.risk_category}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>
          </div>

          {/* Customer Target Group */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Customer Target Group
            </label>
            <input
              type="text"
              name="customer_target_group"
              value={formData.customer_target_group}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="e.g., Young Professionals, Families, Retirees"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description *
            </label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              rows={4}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="Detailed description of the policy features and benefits..."
            />
          </div>

          {/* Keywords */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Keywords
            </label>
            <input
              type="text"
              name="keywords"
              value={formData.keywords}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="e.g., term, protection, affordable, family (comma-separated)"
            />
            <p className="text-xs text-gray-500 mt-1">
              Enter keywords separated by commas for better searchability
            </p>
          </div>
        </div>

        <div className="p-6 border-t border-gray-200 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700"
          >
            Create Policy
          </button>
        </div>
      </div>
    </div>
  );
};

export default PolicyCreationModal;