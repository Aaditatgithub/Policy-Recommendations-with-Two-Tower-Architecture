import React from 'react';
import { Edit, Trash2, Users, Calendar, Target } from 'lucide-react';
import { FilterPreset } from '../../types/promotion';

interface FilterPresetCardProps {
  preset: FilterPreset;
  selected: boolean;
  onSelect: (preset: FilterPreset) => void;
  onEdit: (preset: FilterPreset) => void;
  onDelete: (presetId: string) => void;
}

const FilterPresetCard: React.FC<FilterPresetCardProps> = ({
  preset,
  selected,
  onSelect,
  onEdit,
  onDelete
}) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getActiveFiltersCount = () => {
    let count = 0;
    const filters = preset.filters;
    
    // Count demographic filters
    if (filters.demographic.age.enabled) count++;
    if (filters.demographic.gender.length > 0) count++;
    if (filters.demographic.income_bracket.length > 0) count++;
    if (filters.demographic.employment_status.length > 0) count++;
    if (filters.demographic.marital_status.length > 0) count++;
    if (filters.demographic.location_city.length > 0) count++;
    
    // Count policy filters
    if (filters.policy.policy_ownership_count.enabled) count++;
    if (filters.policy.last_policy_purchase.enabled) count++;
    if (filters.policy.credit_score.enabled) count++;
    if (filters.policy.preferred_policy_type.length > 0) count++;
    
    // Count interaction filters
    if (filters.interaction.clicked !== null) count++;
    if (filters.interaction.purchased !== null) count++;
    if (filters.interaction.abandoned_cart !== null) count++;
    if (filters.interaction.viewed_duration.enabled) count++;
    if (filters.interaction.comparison_count.enabled) count++;
    
    // Count persona filters
    if (filters.persona.length > 0) count++;
    
    return count;
  };

  return (
    <div className={`bg-white rounded-xl border-2 p-6 transition-all duration-200 hover:shadow-lg cursor-pointer ${
      selected ? 'border-violet-500 bg-violet-50' : 'border-gray-200'
    }`} onClick={() => onSelect(preset)}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="font-bold text-lg text-gray-900 mb-1">{preset.name}</h3>
          {preset.description && (
            <p className="text-sm text-gray-600 mb-2">{preset.description}</p>
          )}
        </div>
        <div className="flex space-x-2 ml-4">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit(preset);
            }}
            className="p-2 text-gray-400 hover:text-violet-600 hover:bg-violet-100 rounded-lg transition-colors"
            title="Edit preset"
          >
            <Edit size={16} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(preset.id);
            }}
            className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-100 rounded-lg transition-colors"
            title="Delete preset"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center text-gray-600 mb-1">
            <Users size={14} className="mr-1" />
            <span className="text-xs">Target Customers</span>
          </div>
          <p className="font-bold text-lg text-gray-900">
            {preset.target_customer_count?.toLocaleString() || '0'}
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center text-gray-600 mb-1">
            <Target size={14} className="mr-1" />
            <span className="text-xs">Active Filters</span>
          </div>
          <p className="font-bold text-lg text-gray-900">
            {getActiveFiltersCount()}
          </p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center text-sm text-gray-600">
          <Calendar size={14} className="mr-1" />
          <span>Created {formatDate(preset.created_at)}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
            {preset.selected_policies.length} policies
          </span>
          {preset.is_active && (
            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
              Active
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default FilterPresetCard;