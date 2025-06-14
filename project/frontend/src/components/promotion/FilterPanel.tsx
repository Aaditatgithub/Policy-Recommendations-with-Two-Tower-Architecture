import React, { useState, useEffect } from 'react';
import { Save, Filter, X, Plus, Users, Target, Activity, BarChart3, TrendingUp, Eye, MousePointer, ShoppingCart, Clock, RotateCcw, Settings } from 'lucide-react';
import { FilterState, FilterPreset } from '../../types/promotion';
import CustomFilterModal from './CustomFilterModal';

interface FilterPanelProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
  presets: FilterPreset[];
  onSavePreset: (name: string, filters: FilterState) => void;
  onLoadPreset: (preset: FilterPreset) => void;
  onDeletePreset: (presetId: string) => void;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  onFilterChange,
  presets,
  onSavePreset,
  onLoadPreset,
  onDeletePreset
}) => {
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [customFilters, setCustomFilters] = useState<any[]>([]);
  const [showCustomFilterModal, setShowCustomFilterModal] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    demographic: true,
    policy: true,
    interaction: true,
    persona: true,
    custom: true
  });

  // Load custom filters on component mount
  useEffect(() => {
    fetchCustomFilters();
  }, []);

  const fetchCustomFilters = async () => {
    try {
      const response = await fetch('http://localhost:5000/custom_filters');
      const data = await response.json();
      setCustomFilters(data);
    } catch (error) {
      console.error('Error fetching custom filters:', error);
    }
  };

  const handleCreateCustomFilter = async (filterData: any) => {
    try {
      const response = await fetch('http://localhost:5000/custom_filters', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filterData),
      });

      if (response.ok) {
        fetchCustomFilters(); // Refresh the list
        alert('Custom filter created successfully!');
      } else {
        alert('Failed to create custom filter');
      }
    } catch (error) {
      console.error('Error creating custom filter:', error);
      alert('Error creating custom filter');
    }
  };

  const handleDeleteCustomFilter = async (filterId: number) => {
    if (!confirm('Are you sure you want to delete this custom filter?')) return;

    try {
      const response = await fetch(`http://localhost:5000/custom_filters/${filterId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        fetchCustomFilters(); // Refresh the list
        alert('Custom filter deleted successfully!');
      } else {
        alert('Failed to delete custom filter');
      }
    } catch (error) {
      console.error('Error deleting custom filter:', error);
      alert('Error deleting custom filter');
    }
  };

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const updateFilters = (section: keyof FilterState, field: string, value: any) => {
    const newFilters = {
      ...filters,
      [section]: {
        ...filters[section],
        [field]: value
      }
    };
    onFilterChange(newFilters);
  };

  const handleSavePreset = () => {
    if (presetName.trim()) {
      onSavePreset(presetName.trim(), filters);
      setPresetName('');
      setShowSaveDialog(false);
    }
  };

  const clearAllFilters = () => {
    const clearedFilters: FilterState = {
      demographic: {
        age: { min: 18, max: 65, enabled: false },
        gender: [],
        income_bracket: [],
        employment_status: [],
        marital_status: [],
        location_city: [],
      },
      policy: {
        policy_ownership_count: { min: 0, max: 5, enabled: false },
        last_policy_purchase: { from: '', to: '', enabled: false },
        credit_score: { min: 300, max: 900, enabled: false },
        preferred_policy_type: [],
      },
      interaction: {
        clicked: null,
        purchased: null,
        abandoned_cart: null,
        viewed_duration: { min: 0, max: 120, enabled: false },
        comparison_count: { min: 0, max: 10, enabled: false },
      },
      persona: [],
    };
    onFilterChange(clearedFilters);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center">
          <Filter className="mr-2 text-violet-600" size={20} />
          Customer Filters
        </h2>
        <button
          onClick={clearAllFilters}
          className="text-sm text-gray-500 hover:text-red-600 flex items-center transition-colors"
        >
          <RotateCcw size={16} className="mr-1" />
          Clear All
        </button>
      </div>

      {/* Demographics Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection('demographic')}
          className="w-full p-4 text-left font-medium bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
        >
          <div className="flex items-center">
            <Users className="mr-2 text-blue-600" size={18} />
            <span>Demographics</span>
          </div>
          <span className={`transform transition-transform ${expandedSections.demographic ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {expandedSections.demographic && (
          <div className="p-4 space-y-6 bg-white">
            {/* Age Range */}
            <div>
              <label className="flex items-center mb-3">
                <input
                  type="checkbox"
                  checked={filters.demographic.age.enabled}
                  onChange={(e) => updateFilters('demographic', 'age', {
                    ...filters.demographic.age,
                    enabled: e.target.checked
                  })}
                  className="mr-2 w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
                />
                <span className="text-sm font-medium text-gray-700">Age Range</span>
              </label>
              {filters.demographic.age.enabled && (
                <div className="ml-6">
                  <div className="flex items-center space-x-3 mb-2">
                    <input
                      type="number"
                      placeholder="18"
                      value={filters.demographic.age.min}
                      onChange={(e) => updateFilters('demographic', 'age', {
                        ...filters.demographic.age,
                        min: parseInt(e.target.value) || 18
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">to</span>
                    <input
                      type="number"
                      placeholder="65"
                      value={filters.demographic.age.max}
                      onChange={(e) => updateFilters('demographic', 'age', {
                        ...filters.demographic.age,
                        max: parseInt(e.target.value) || 65
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">years</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-violet-600 h-2 rounded-full" 
                      style={{ width: `${((filters.demographic.age.max - filters.demographic.age.min) / 47) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Gender */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Gender</label>
              <div className="flex flex-wrap gap-2 mt-2">
                {['Male', 'Female', 'Other'].map(gender => (
                  <button
                    key={gender}
                    type="button"
                    onClick={() => {
                      const isSelected = filters.demographic.gender.includes(gender);
                      const newGenders = isSelected
                        ? filters.demographic.gender.filter(g => g !== gender)
                        : [...filters.demographic.gender, gender];
                      updateFilters('demographic', 'gender', newGenders);
                    }}
                    className={`px-3 py-2 rounded-lg text-sm border ${
                      filters.demographic.gender.includes(gender)
                        ? 'bg-violet-100 border-violet-500 text-violet-800'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {gender}
                  </button>
                ))}
              </div>
              {filters.demographic.gender.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.demographic.gender.map(gender => (
                    <span key={gender} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-violet-100 text-violet-800">
                      {gender}
                      <button
                        onClick={() => {
                          const newGenders = filters.demographic.gender.filter(g => g !== gender);
                          updateFilters('demographic', 'gender', newGenders);
                        }}
                        className="ml-1 text-violet-600 hover:text-violet-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Income Bracket */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Income Bracket</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newBrackets = [...filters.demographic.income_bracket];
                      if (!newBrackets.includes(e.target.value)) {
                        newBrackets.push(e.target.value);
                        updateFilters('demographic', 'income_bracket', newBrackets);
                      }
                    }
                  }}
                >
                  <option value="">Select income range</option>
                  <option value="0-250000">₹0 - ₹2.5L</option>
                  <option value="250001-750000">₹2.5L - ₹7.5L</option>
                  <option value="750001-1500000">₹7.5L - ₹15L</option>
                  <option value="1500001+">₹15L+</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.demographic.income_bracket.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.demographic.income_bracket.map(bracket => (
                    <span key={bracket} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                      ₹{bracket}
                      <button
                        onClick={() => {
                          const newBrackets = filters.demographic.income_bracket.filter(b => b !== bracket);
                          updateFilters('demographic', 'income_bracket', newBrackets);
                        }}
                        className="ml-1 text-green-600 hover:text-green-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Employment Status */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Employment Status</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newStatuses = [...filters.demographic.employment_status];
                      if (!newStatuses.includes(e.target.value)) {
                        newStatuses.push(e.target.value);
                        updateFilters('demographic', 'employment_status', newStatuses);
                      }
                    }
                  }}
                >
                  <option value="">Select employment status</option>
                  <option value="Salaried">Salaried</option>
                  <option value="Self-Employed">Self-Employed</option>
                  <option value="Business Owner">Business Owner</option>
                  <option value="Retired">Retired</option>
                  <option value="Freelancer">Freelancer</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.demographic.employment_status.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.demographic.employment_status.map(status => (
                    <span key={status} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                      {status}
                      <button
                        onClick={() => {
                          const newStatuses = filters.demographic.employment_status.filter(s => s !== status);
                          updateFilters('demographic', 'employment_status', newStatuses);
                        }}
                        className="ml-1 text-blue-600 hover:text-blue-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Marital Status */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Marital Status</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newStatuses = [...filters.demographic.marital_status];
                      if (!newStatuses.includes(e.target.value)) {
                        newStatuses.push(e.target.value);
                        updateFilters('demographic', 'marital_status', newStatuses);
                      }
                    }
                  }}
                >
                  <option value="">Select marital status</option>
                  <option value="Single">Single</option>
                  <option value="Married">Married</option>
                  <option value="Divorced">Divorced</option>
                  <option value="Widowed">Widowed</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.demographic.marital_status.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.demographic.marital_status.map(status => (
                    <span key={status} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-800">
                      {status}
                      <button
                        onClick={() => {
                          const newStatuses = filters.demographic.marital_status.filter(s => s !== status);
                          updateFilters('demographic', 'marital_status', newStatuses);
                        }}
                        className="ml-1 text-purple-600 hover:text-purple-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Location */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Location</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newCities = [...filters.demographic.location_city];
                      if (!newCities.includes(e.target.value)) {
                        newCities.push(e.target.value);
                        updateFilters('demographic', 'location_city', newCities);
                      }
                    }
                  }}
                >
                  <option value="">Select cities</option>
                  <option value="Mumbai">Mumbai</option>
                  <option value="Delhi">Delhi</option>
                  <option value="Bengaluru">Bengaluru</option>
                  <option value="Chennai">Chennai</option>
                  <option value="Hyderabad">Hyderabad</option>
                  <option value="Pune">Pune</option>
                  <option value="Kolkata">Kolkata</option>
                  <option value="Ahmedabad">Ahmedabad</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.demographic.location_city.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.demographic.location_city.map(city => (
                    <span key={city} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-indigo-100 text-indigo-800">
                      {city}
                      <button
                        onClick={() => {
                          const newCities = filters.demographic.location_city.filter(c => c !== city);
                          updateFilters('demographic', 'location_city', newCities);
                        }}
                        className="ml-1 text-indigo-600 hover:text-indigo-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Policy Preferences Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection('policy')}
          className="w-full p-4 text-left font-medium bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
        >
          <div className="flex items-center">
            <Target className="mr-2 text-green-600" size={18} />
            <span>Policy Preferences</span>
          </div>
          <span className={`transform transition-transform ${expandedSections.policy ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {expandedSections.policy && (
          <div className="p-4 space-y-6 bg-white">
            {/* Policy Ownership Count */}
            <div>
              <label className="flex items-center mb-3">
                <input
                  type="checkbox"
                  checked={filters.policy.policy_ownership_count.enabled}
                  onChange={(e) => updateFilters('policy', 'policy_ownership_count', {
                    ...filters.policy.policy_ownership_count,
                    enabled: e.target.checked
                  })}
                  className="mr-2 w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
                />
                <span className="text-sm font-medium text-gray-700">Policy Ownership Count</span>
              </label>
              {filters.policy.policy_ownership_count.enabled && (
                <div className="ml-6">
                  <div className="flex items-center space-x-3 mb-2">
                    <input
                      type="number"
                      placeholder="0"
                      value={filters.policy.policy_ownership_count.min}
                      onChange={(e) => updateFilters('policy', 'policy_ownership_count', {
                        ...filters.policy.policy_ownership_count,
                        min: parseInt(e.target.value) || 0
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">to</span>
                    <input
                      type="number"
                      placeholder="5"
                      value={filters.policy.policy_ownership_count.max}
                      onChange={(e) => updateFilters('policy', 'policy_ownership_count', {
                        ...filters.policy.policy_ownership_count,
                        max: parseInt(e.target.value) || 5
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${(filters.policy.policy_ownership_count.max / 5) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Credit Score */}
            <div>
              <label className="flex items-center mb-3">
                <input
                  type="checkbox"
                  checked={filters.policy.credit_score.enabled}
                  onChange={(e) => updateFilters('policy', 'credit_score', {
                    ...filters.policy.credit_score,
                    enabled: e.target.checked
                  })}
                  className="mr-2 w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
                />
                <span className="text-sm font-medium text-gray-700">Credit Score</span>
              </label>
              {filters.policy.credit_score.enabled && (
                <div className="ml-6">
                  <div className="flex items-center space-x-3 mb-2">
                    <input
                      type="number"
                      placeholder="300"
                      value={filters.policy.credit_score.min}
                      onChange={(e) => updateFilters('policy', 'credit_score', {
                        ...filters.policy.credit_score,
                        min: parseInt(e.target.value) || 300
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">to</span>
                    <input
                      type="number"
                      placeholder="900"
                      value={filters.policy.credit_score.max}
                      onChange={(e) => updateFilters('policy', 'credit_score', {
                        ...filters.policy.credit_score,
                        max: parseInt(e.target.value) || 900
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-600 h-2 rounded-full" 
                      style={{ width: `${((filters.policy.credit_score.max - 300) / 600) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Preferred Policy Type */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Preferred Policy Type</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newTypes = [...filters.policy.preferred_policy_type];
                      if (!newTypes.includes(e.target.value)) {
                        newTypes.push(e.target.value);
                        updateFilters('policy', 'preferred_policy_type', newTypes);
                      }
                    }
                  }}
                >
                  <option value="">Select policy types</option>
                  <option value="Term">Term Insurance</option>
                  <option value="ULIP">ULIP</option>
                  <option value="Health">Health Insurance</option>
                  <option value="Endowment">Endowment</option>
                  <option value="Money Back">Money Back</option>
                  <option value="Pension">Pension</option>
                  <option value="Child Plan">Child Plan</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.policy.preferred_policy_type.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.policy.preferred_policy_type.map(type => (
                    <span key={type} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-orange-100 text-orange-800">
                      {type}
                      <button
                        onClick={() => {
                          const newTypes = filters.policy.preferred_policy_type.filter(t => t !== type);
                          updateFilters('policy', 'preferred_policy_type', newTypes);
                        }}
                        className="ml-1 text-orange-600 hover:text-orange-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Interaction Behavior Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection('interaction')}
          className="w-full p-4 text-left font-medium bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
        >
          <div className="flex items-center">
            <BarChart3 className="mr-2 text-purple-600" size={18} />
            <span>Interaction Behavior</span>
          </div>
          <span className={`transform transition-transform ${expandedSections.interaction ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {expandedSections.interaction && (
          <div className="p-4 space-y-6 bg-white">
            {/* Clicked Policies */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 flex items-center">
                <MousePointer className="mr-2 text-blue-500" size={16} />
                Clicked Policies
              </label>
              <select
                value={filters.interaction.clicked === null ? '' : filters.interaction.clicked.toString()}
                onChange={(e) => {
                  const value = e.target.value === '' ? null : e.target.value === 'true';
                  updateFilters('interaction', 'clicked', value);
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value="">Any</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>

            {/* Purchased */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 flex items-center">
                <ShoppingCart className="mr-2 text-green-500" size={16} />
                Purchased
              </label>
              <select
                value={filters.interaction.purchased === null ? '' : filters.interaction.purchased.toString()}
                onChange={(e) => {
                  const value = e.target.value === '' ? null : e.target.value === 'true';
                  updateFilters('interaction', 'purchased', value);
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value="">Any</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>

            {/* Abandoned Cart */}
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 flex items-center">
                <X className="mr-2 text-red-500" size={16} />
                Abandoned Cart
              </label>
              <select
                value={filters.interaction.abandoned_cart === null ? '' : filters.interaction.abandoned_cart.toString()}
                onChange={(e) => {
                  const value = e.target.value === '' ? null : e.target.value === 'true';
                  updateFilters('interaction', 'abandoned_cart', value);
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value="">Any</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>

            {/* Viewing Duration */}
            <div>
              <label className="flex items-center mb-3">
                <input
                  type="checkbox"
                  checked={filters.interaction.viewed_duration.enabled}
                  onChange={(e) => updateFilters('interaction', 'viewed_duration', {
                    ...filters.interaction.viewed_duration,
                    enabled: e.target.checked
                  })}
                  className="mr-2 w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
                />
                <Clock className="mr-2 text-yellow-500" size={16} />
                <span className="text-sm font-medium text-gray-700">Viewing Duration (seconds)</span>
              </label>
              {filters.interaction.viewed_duration.enabled && (
                <div className="ml-6">
                  <div className="flex items-center space-x-3 mb-2">
                    <input
                      type="number"
                      placeholder="0"
                      value={filters.interaction.viewed_duration.min}
                      onChange={(e) => updateFilters('interaction', 'viewed_duration', {
                        ...filters.interaction.viewed_duration,
                        min: parseInt(e.target.value) || 0
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">to</span>
                    <input
                      type="number"
                      placeholder="120"
                      value={filters.interaction.viewed_duration.max}
                      onChange={(e) => updateFilters('interaction', 'viewed_duration', {
                        ...filters.interaction.viewed_duration,
                        max: parseInt(e.target.value) || 120
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">sec</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-600 h-2 rounded-full" 
                      style={{ width: `${(filters.interaction.viewed_duration.max / 120) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            {/* Comparison Count */}
            <div>
              <label className="flex items-center mb-3">
                <input
                  type="checkbox"
                  checked={filters.interaction.comparison_count.enabled}
                  onChange={(e) => updateFilters('interaction', 'comparison_count', {
                    ...filters.interaction.comparison_count,
                    enabled: e.target.checked
                  })}
                  className="mr-2 w-4 h-4 text-violet-600 rounded focus:ring-violet-500"
                />
                <TrendingUp className="mr-2 text-indigo-500" size={16} />
                <span className="text-sm font-medium text-gray-700">Comparison Count</span>
              </label>
              {filters.interaction.comparison_count.enabled && (
                <div className="ml-6">
                  <div className="flex items-center space-x-3 mb-2">
                    <input
                      type="number"
                      placeholder="0"
                      value={filters.interaction.comparison_count.min}
                      onChange={(e) => updateFilters('interaction', 'comparison_count', {
                        ...filters.interaction.comparison_count,
                        min: parseInt(e.target.value) || 0
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                    <span className="text-gray-500 text-sm">to</span>
                    <input
                      type="number"
                      placeholder="10"
                      value={filters.interaction.comparison_count.max}
                      onChange={(e) => updateFilters('interaction', 'comparison_count', {
                        ...filters.interaction.comparison_count,
                        max: parseInt(e.target.value) || 10
                      })}
                      className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-indigo-600 h-2 rounded-full" 
                      style={{ width: `${(filters.interaction.comparison_count.max / 10) * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Customer Persona Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection('persona')}
          className="w-full p-4 text-left font-medium bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
        >
          <div className="flex items-center">
            <Activity className="mr-2 text-pink-600" size={18} />
            <span>Customer Persona</span>
          </div>
          <span className={`transform transition-transform ${expandedSections.persona ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {expandedSections.persona && (
          <div className="p-4 space-y-4 bg-white">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-3 block">Customer Persona Types</label>
              <div className="relative">
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none bg-white"
                  onChange={(e) => {
                    if (e.target.value) {
                      const newPersonas = [...filters.persona];
                      if (!newPersonas.includes(e.target.value)) {
                        newPersonas.push(e.target.value);
                        updateFilters('persona', '', newPersonas);
                      }
                    }
                  }}
                >
                  <option value="">Select customer personas</option>
                  <option value="Tech-Savvy">Tech-Savvy</option>
                  <option value="Family-Oriented">Family-Oriented</option>
                  <option value="Investment-Savvy">Investment-Savvy</option>
                  <option value="Security-Conscious">Security-Conscious</option>
                  <option value="Budget-Conscious">Budget-Conscious</option>
                  <option value="Career-Focused">Career-Focused</option>
                  <option value="Health-Conscious">Health-Conscious</option>
                  <option value="Wealth-Builder">Wealth-Builder</option>
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
              {filters.persona.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {filters.persona.map(persona => (
                    <span key={persona} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-pink-100 text-pink-800">
                      {persona}
                      <button
                        onClick={() => {
                          const newPersonas = filters.persona.filter(p => p !== persona);
                          updateFilters('persona', '', newPersonas);
                        }}
                        className="ml-1 text-pink-600 hover:text-pink-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Custom Filters Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleSection('custom')}
          className="w-full p-4 text-left font-medium bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
        >
          <div className="flex items-center">
            <Settings className="mr-2 text-orange-600" size={18} />
            <span>Custom Filters</span>
          </div>
          <span className={`transform transition-transform ${expandedSections.custom ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>
        {expandedSections.custom && (
          <div className="p-4 space-y-4 bg-white">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-gray-700">Available Custom Filters</h4>
              <button
                onClick={() => setShowCustomFilterModal(true)}
                className="flex items-center text-sm text-violet-600 hover:text-violet-700"
              >
                <Plus size={16} className="mr-1" />
                Create Filter
              </button>
            </div>

            {customFilters.length === 0 ? (
              <div className="text-center py-6 text-gray-500">
                <Settings className="mx-auto mb-2" size={32} />
                <p>No custom filters created yet</p>
                <button
                  onClick={() => setShowCustomFilterModal(true)}
                  className="mt-2 text-violet-600 hover:text-violet-700 text-sm"
                >
                  Create your first custom filter
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {customFilters.map((filter) => (
                  <div key={filter.id} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-gray-900">{filter.filter_name}</h5>
                      <button
                        onClick={() => handleDeleteCustomFilter(filter.id)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <X size={16} />
                      </button>
                    </div>
                    <p className="text-xs text-gray-600 mb-2">Type: {filter.filter_type}</p>
                    {filter.filter_options.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {filter.filter_options.slice(0, 3).map((option: string, index: number) => (
                          <span key={index} className="text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full">
                            {option}
                          </span>
                        ))}
                        {filter.filter_options.length > 3 && (
                          <span className="text-xs text-gray-500">+{filter.filter_options.length - 3} more</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Save Preset */}
      <div className="pt-4 border-t border-gray-200">
        <button
          onClick={() => setShowSaveDialog(true)}
          className="w-full bg-violet-600 text-white py-3 px-4 rounded-lg hover:bg-violet-700 flex items-center justify-center transition-colors font-medium"
        >
          <Save size={16} className="mr-2" />
          Save as Preset
        </button>
      </div>

      {/* Save Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96">
            <h3 className="text-lg font-semibold mb-4">Save Filter Preset</h3>
            <input
              type="text"
              placeholder="Enter preset name..."
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              className="w-full px-3 py-2 border rounded mb-4"
              autoFocus
            />
            <div className="flex space-x-3">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSavePreset}
                disabled={!presetName.trim()}
                className="flex-1 px-4 py-2 bg-violet-600 text-white rounded hover:bg-violet-700 disabled:opacity-50"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Custom Filter Modal */}
      <CustomFilterModal
        isOpen={showCustomFilterModal}
        onClose={() => setShowCustomFilterModal(false)}
        onSave={handleCreateCustomFilter}
      />
    </div>
  );
};

export default FilterPanel;