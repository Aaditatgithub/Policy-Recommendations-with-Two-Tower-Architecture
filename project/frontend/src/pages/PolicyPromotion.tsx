import React, { useState, useEffect } from 'react';
import { Grid, List, Filter, Users, TrendingUp, Target, Plus, Edit, Trash2, X, FileText } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

import PolicyPromotionCard from '../components/promotion/PolicyPromotionCard';
import FilterPanel from '../components/promotion/FilterPanel';
import PolicyCreationModal from '../components/promotion/PolicyCreationModal';
import Navbar from '../components/NavBar';
import { Policy, FilterPreset, FilterState } from '../types/promotion';

const STORAGE_KEY = 'policy_promotion_presets';

const PolicyPromotion: React.FC = () => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [selectedPolicies, setSelectedPolicies] = useState<Policy[]>([]);
  const [filterPresets, setFilterPresets] = useState<FilterPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<FilterPreset | null>(null);
  const [showCreatePreset, setShowCreatePreset] = useState(false);
  const [showCreatePolicy, setShowCreatePolicy] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [loading, setLoading] = useState(true);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');
  const [newPresetFilters, setNewPresetFilters] = useState<FilterState>({
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
  });

  useEffect(() => {
    fetchPolicies();
    loadPresets();
  }, []);

  useEffect(() => {
    if (filterPresets.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filterPresets));
    }
  }, [filterPresets]);

  const fetchPolicies = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/search_policies?q=');
      const data = await response.json();
      
      const transformedPolicies: Policy[] = data.map((policy: any) => ({
        policy_id: policy.policy_id,
        policy_name: policy.policy_name,
        policy_type: policy.policy_type,
        sum_assured: policy['sum_assured (INR)'] || 0,
        premium_amount: policy['premium_amount (INR)'] || 0,
        policy_duration_years: policy.policy_duration_years || 0,
        risk_category: policy.risk_category || 'Medium',
        customer_target_group: policy.customer_target_group ? 
          policy.customer_target_group.split(',').map((s: string) => s.trim()) : 
          ['General'],
        description: policy.description || '',
        keywords: policy.keywords ? 
          policy.keywords.split(',').map((s: string) => s.trim()) : 
          [],
        score: Math.floor(Math.random() * 40) + 60,
        selected_for_promotion: false,
        is_promoted: policy.is_promoted || false,
        promotion_tag: policy.promotion_tag || ''
      }));

      setPolicies(transformedPolicies);
    } catch (error) {
      console.error('Error fetching policies:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadSamplePresets = () => {
    const samplePresets: FilterPreset[] = [
      {
        id: uuidv4(),
        name: 'Young Professionals',
        description: 'Target young working professionals aged 25-35',
        filters: {
          demographic: {
            age: { min: 25, max: 35, enabled: true },
            gender: [],
            income_bracket: ['250001-750000', '750001-1500000'],
            employment_status: ['Salaried'],
            marital_status: [],
            location_city: [],
          },
          policy: {
            policy_ownership_count: { min: 0, max: 2, enabled: true },
            last_policy_purchase: { from: '', to: '', enabled: false },
            credit_score: { min: 650, max: 900, enabled: true },
            preferred_policy_type: ['Term', 'ULIP'],
          },
          interaction: {
            clicked: true,
            purchased: false,
            abandoned_cart: null,
            viewed_duration: { min: 30, max: 120, enabled: true },
            comparison_count: { min: 1, max: 10, enabled: false },
          },
          persona: ['Tech-Savvy', 'Career-Focused'],
        },
        selected_policies: [],
        created_at: new Date().toISOString(),
        target_customer_count: 15420,
        is_active: true
      },
      {
        id: uuidv4(),
        name: 'Family Protectors',
        description: 'Married individuals with families looking for comprehensive coverage',
        filters: {
          demographic: {
            age: { min: 30, max: 50, enabled: true },
            gender: [],
            income_bracket: ['750001-1500000', '1500001+'],
            employment_status: ['Salaried', 'Business Owner'],
            marital_status: ['Married'],
            location_city: [],
          },
          policy: {
            policy_ownership_count: { min: 1, max: 5, enabled: true },
            last_policy_purchase: { from: '', to: '', enabled: false },
            credit_score: { min: 700, max: 900, enabled: true },
            preferred_policy_type: ['Health', 'Term'],
          },
          interaction: {
            clicked: null,
            purchased: null,
            abandoned_cart: false,
            viewed_duration: { min: 60, max: 120, enabled: true },
            comparison_count: { min: 2, max: 10, enabled: true },
          },
          persona: ['Family-Oriented', 'Security-Conscious'],
        },
        selected_policies: [],
        created_at: new Date(Date.now() - 86400000).toISOString(),
        target_customer_count: 8750,
        is_active: false
      },
      {
        id: uuidv4(),
        name: 'High Net Worth Investors',
        description: 'Affluent customers interested in investment-linked policies',
        filters: {
          demographic: {
            age: { min: 35, max: 60, enabled: true },
            gender: [],
            income_bracket: ['1500001+'],
            employment_status: ['Business Owner', 'Self-Employed'],
            marital_status: [],
            location_city: ['Mumbai', 'Delhi', 'Bengaluru'],
          },
          policy: {
            policy_ownership_count: { min: 2, max: 5, enabled: true },
            last_policy_purchase: { from: '', to: '', enabled: false },
            credit_score: { min: 750, max: 900, enabled: true },
            preferred_policy_type: ['ULIP', 'Endowment'],
          },
          interaction: {
            clicked: true,
            purchased: true,
            abandoned_cart: false,
            viewed_duration: { min: 90, max: 120, enabled: true },
            comparison_count: { min: 3, max: 10, enabled: true },
          },
          persona: ['Investment-Savvy', 'Wealth-Builder'],
        },
        selected_policies: [],
        created_at: new Date(Date.now() - 172800000).toISOString(),
        target_customer_count: 3240,
        is_active: true
      }
    ];

    setFilterPresets(samplePresets);
  };

  const loadPresets = () => {
    try {
      const savedPresets = localStorage.getItem(STORAGE_KEY);
      
      if (savedPresets && savedPresets !== 'undefined') {
        const parsedPresets = JSON.parse(savedPresets);
        if (Array.isArray(parsedPresets) && parsedPresets.length > 0) {
          setFilterPresets(parsedPresets);
          console.log('Loaded presets from localStorage:', parsedPresets.length);
          return;
        }
      }
      
      loadSamplePresets();
    } catch (error) {
      console.error('Error loading presets from localStorage:', error);
      loadSamplePresets();
    }
  };

  const handlePolicySelect = (policy: Policy) => {
    setSelectedPolicies(prev => {
      const isSelected = prev.some(p => p.policy_id === policy.policy_id);
      if (isSelected) {
        return prev.filter(p => p.policy_id !== policy.policy_id);
      }
      return [...prev, policy];
    });
  };

  const handlePolicyPromote = async (policy: Policy) => {
    try {
      const response = await fetch('http://localhost:5000/promote_policies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          policy_ids: [policy.policy_id],
          preset_id: selectedPreset?.id || 'manual',
          promotion_name: selectedPreset?.name || 'Manual Promotion',
          priority: 1
        }),
      });

      if (response.ok) {
        setPolicies(prev => 
          prev.map(p => 
            p.policy_id === policy.policy_id 
              ? { ...p, selected_for_promotion: !p.selected_for_promotion, is_promoted: !p.selected_for_promotion }
              : p
          )
        );
        alert('Policy promotion status updated successfully!');
      } else {
        alert('Failed to update policy promotion status');
      }
    } catch (error) {
      console.error('Error promoting policy:', error);
      alert('Error updating policy promotion status');
    }
  };

  const handleCreatePolicy = async (policyData: any) => {
    try {
      const response = await fetch('http://localhost:5000/policies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(policyData),
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Policy created successfully! Policy ID: ${result.policy_id}`);
        fetchPolicies(); // Refresh the policies list
      } else {
        const error = await response.json();
        alert(`Failed to create policy: ${error.error}`);
      }
    } catch (error) {
      console.error('Error creating policy:', error);
      alert('Error creating policy');
    }
  };

  const handlePresetSelect = (preset: FilterPreset) => {
    setSelectedPreset(prev => prev?.id === preset.id ? null : preset);
  };

  const handlePresetEdit = (preset: FilterPreset) => {
    setSelectedPreset(preset);
    setNewPresetName(preset.name);
    setNewPresetDescription(preset.description || '');
    setNewPresetFilters(preset.filters);
    setShowCreatePreset(true);
  };

  const handleSaveEditedPreset = () => {
    if (!newPresetName.trim() || !selectedPreset) return;

    const updatedPreset: FilterPreset = {
      ...selectedPreset,
      name: newPresetName.trim(),
      description: newPresetDescription.trim(),
      filters: newPresetFilters,
      selected_policies: selectedPolicies,
      created_at: new Date().toISOString(),
    };

    setFilterPresets(prev => {
      const updated = prev.map(p => 
        p.id === selectedPreset.id ? updatedPreset : p
      );
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
    
    setSelectedPreset(updatedPreset);
    setShowCreatePreset(false);
    setNewPresetName('');
    setNewPresetDescription('');
  };

  const handlePresetDelete = (presetId: string) => {
    setFilterPresets(prev => {
      const updated = prev.filter(p => p.id !== presetId);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
    
    if (selectedPreset?.id === presetId) {
      setSelectedPreset(null);
    }
  };

  const handleCreatePreset = () => {
    if (!newPresetName.trim()) return;

    const newPreset: FilterPreset = {
      id: uuidv4(),
      name: newPresetName.trim(),
      description: newPresetDescription.trim(),
      filters: newPresetFilters,
      selected_policies: selectedPolicies,
      created_at: new Date().toISOString(),
      target_customer_count: Math.floor(Math.random() * 10000) + 1000,
      is_active: false
    };

    setFilterPresets(prev => {
      const updated = [...prev, newPreset];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
    
    setShowCreatePreset(false);
    setNewPresetName('');
    setNewPresetDescription('');
    setNewPresetFilters({
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
    });
  };

  const getFilteredPolicies = () => {
    if (!selectedPreset) return policies;
    
    return policies.filter(policy => {
      const filters = selectedPreset.filters;
      
      if (filters.policy.preferred_policy_type.length > 0) {
        return filters.policy.preferred_policy_type.includes(policy.policy_type);
      }
      
      return true;
    });
  };

  const filteredPolicies = getFilteredPolicies();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-200">
      <Navbar userType="admin" />
      
      <div className="flex h-screen pt-16">
        {/* Left Sidebar - Presets */}
        <div className="w-80 bg-white shadow-lg border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">Filter Presets</h2>
              <button
                onClick={() => setShowCreatePreset(true)}
                className="bg-violet-600 text-white p-2 rounded-lg hover:bg-violet-700 transition-colors"
                title="Create New Preset"
              >
                <Plus size={16} />
              </button>
            </div>
            
            {selectedPreset && (
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 mb-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-violet-800">
                      Active: {selectedPreset.name}
                    </h3>
                    <p className="text-xs text-violet-600">
                      {filteredPolicies.length} policies match
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedPreset(null)}
                    className="text-violet-600 hover:text-violet-800"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {filterPresets.map(preset => (
              <div
                key={preset.id}
                className={`border rounded-lg p-3 cursor-pointer transition-all ${
                  selectedPreset?.id === preset.id
                    ? 'border-violet-500 bg-violet-50'
                    : 'border-gray-200 hover:border-violet-300 hover:bg-gray-50'
                }`}
                onClick={() => handlePresetSelect(preset)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900 text-sm">{preset.name}</h3>
                    {preset.description && (
                      <p className="text-xs text-gray-600 mt-1">{preset.description}</p>
                    )}
                  </div>
                  <div className="flex space-x-1 ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePresetEdit(preset);
                      }}
                      className="p-1 text-gray-400 hover:text-violet-600 rounded"
                    >
                      <Edit size={12} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePresetDelete(preset.id);
                      }}
                      className="p-1 text-gray-400 hover:text-red-600 rounded"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">
                    {preset.target_customer_count?.toLocaleString()} customers
                  </span>
                  {preset.is_active && (
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full">
                      Active
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="bg-white border-b border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="text-2xl font-bold text-violet-700">Policy Promotion Management</h1>
                <p className="text-gray-600">Create targeted campaigns and manage policy promotions</p>
              </div>
              
              <div className="flex items-center space-x-3">
                {/* Add Policy Button */}
                <button
                  onClick={() => setShowCreatePolicy(true)}
                  className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <FileText size={16} className="mr-2" />
                  Add Policy
                </button>
                
                <div className="flex rounded-lg overflow-hidden shadow-sm border border-gray-200">
                  <button
                    className={`px-3 py-2 flex items-center text-sm ${
                      viewMode === 'grid' 
                        ? 'bg-violet-600 text-white' 
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                    onClick={() => setViewMode('grid')}
                  >
                    <Grid size={16} className="mr-1" /> Grid
                  </button>
                  <button
                    className={`px-3 py-2 flex items-center text-sm ${
                      viewMode === 'list' 
                        ? 'bg-violet-600 text-white' 
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                    onClick={() => setViewMode('list')}
                  >
                    <List size={16} className="mr-1" /> List
                  </button>
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm text-blue-600 mb-1">Total Policies</h3>
                    <p className="text-2xl font-bold text-blue-800">{policies.length}</p>
                  </div>
                  <Target className="text-blue-500" size={24} />
                </div>
              </div>
              <div className="bg-gradient-to-r from-violet-50 to-violet-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm text-violet-600 mb-1">Promoted</h3>
                    <p className="text-2xl font-bold text-violet-800">
                      {policies.filter(p => p.selected_for_promotion).length}
                    </p>
                  </div>
                  <TrendingUp className="text-violet-500" size={24} />
                </div>
              </div>
              <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm text-green-600 mb-1">Active Presets</h3>
                    <p className="text-2xl font-bold text-green-800">
                      {filterPresets.filter(p => p.is_active).length}
                    </p>
                  </div>
                  <Filter className="text-green-500" size={24} />
                </div>
              </div>
              <div className="bg-gradient-to-r from-amber-50 to-amber-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm text-amber-600 mb-1">Target Customers</h3>
                    <p className="text-2xl font-bold text-amber-800">
                      {selectedPreset?.target_customer_count?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <Users className="text-amber-500" size={24} />
                </div>
              </div>
            </div>
          </div>

          {/* Policies Grid */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">
                Available Policies
                {selectedPreset && (
                  <span className="text-sm font-normal text-gray-600 ml-2">
                    ({filteredPolicies.length} matching "{selectedPreset.name}")
                  </span>
                )}
              </h2>
              {selectedPolicies.length > 0 && (
                <div className="text-sm text-gray-600">
                  {selectedPolicies.length} selected
                </div>
              )}
            </div>
            
            {loading ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-700 mx-auto mb-4"></div>
                <p className="text-gray-500">Loading policies...</p>
              </div>
            ) : (
              <div className={`grid ${
                viewMode === 'grid' 
                  ? 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3' 
                  : 'grid-cols-1'
              } gap-4`}>
                {filteredPolicies.map(policy => (
                  <PolicyPromotionCard
                    key={policy.policy_id}
                    policy={policy}
                    selected={selectedPolicies.some(p => p.policy_id === policy.policy_id)}
                    onSelect={handlePolicySelect}
                    onPromote={handlePolicyPromote}
                    viewMode={viewMode}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Create Preset Modal */}
      {showCreatePreset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold">
                  {selectedPreset ? 'Edit Preset' : 'Create New Preset'}
                </h3>
                <button
                  onClick={() => setShowCreatePreset(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X size={24} />
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
              <div className="space-y-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Preset Name *
                  </label>
                  <input
                    type="text"
                    value={newPresetName}
                    onChange={(e) => setNewPresetName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                    placeholder="Enter preset name..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={newPresetDescription}
                    onChange={(e) => setNewPresetDescription(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                    rows={3}
                    placeholder="Enter preset description..."
                  />
                </div>
              </div>

              <FilterPanel
                filters={newPresetFilters}
                onFilterChange={setNewPresetFilters}
                presets={[]}
                onSavePreset={() => {}}
                onLoadPreset={() => {}}
                onDeletePreset={() => {}}
              />
            </div>
            
            <div className="p-6 border-t border-gray-200 flex justify-end space-x-3">
              <button
                onClick={() => setShowCreatePreset(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={selectedPreset ? handleSaveEditedPreset : handleCreatePreset}
                disabled={!newPresetName.trim()}
                className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50"
              >
                {selectedPreset ? 'Save Changes' : 'Create Preset'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Policy Modal */}
      <PolicyCreationModal
        isOpen={showCreatePolicy}
        onClose={() => setShowCreatePolicy(false)}
        onSave={handleCreatePolicy}
      />
    </div>
  );
};

export default PolicyPromotion;