import React, { useState } from 'react';
import { X, Plus, Trash2 } from 'lucide-react';

interface CustomFilterModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (filterData: any) => void;
}

const CustomFilterModal: React.FC<CustomFilterModalProps> = ({
  isOpen,
  onClose,
  onSave
}) => {
  const [filterName, setFilterName] = useState('');
  const [filterType, setFilterType] = useState('dropdown');
  const [filterOptions, setFilterOptions] = useState<string[]>(['']);

  const handleAddOption = () => {
    setFilterOptions([...filterOptions, '']);
  };

  const handleRemoveOption = (index: number) => {
    setFilterOptions(filterOptions.filter((_, i) => i !== index));
  };

  const handleOptionChange = (index: number, value: string) => {
    const newOptions = [...filterOptions];
    newOptions[index] = value;
    setFilterOptions(newOptions);
  };

  const handleSave = () => {
    if (!filterName.trim()) {
      alert('Please enter a filter name');
      return;
    }

    const validOptions = filterOptions.filter(opt => opt.trim() !== '');
    
    if (filterType === 'dropdown' && validOptions.length === 0) {
      alert('Please add at least one option for dropdown filter');
      return;
    }

    onSave({
      filter_name: filterName.trim(),
      filter_type: filterType,
      filter_options: validOptions
    });

    // Reset form
    setFilterName('');
    setFilterType('dropdown');
    setFilterOptions(['']);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold">Create Custom Filter</h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X size={24} />
          </button>
        </div>

        <div className="space-y-4">
          {/* Filter Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter Name *
            </label>
            <input
              type="text"
              value={filterName}
              onChange={(e) => setFilterName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
              placeholder="e.g., Occupation Type"
            />
          </div>

          {/* Filter Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter Type *
            </label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value="dropdown">Dropdown</option>
              <option value="checkbox">Checkbox</option>
              <option value="range">Range</option>
            </select>
          </div>

          {/* Filter Options (for dropdown and checkbox) */}
          {(filterType === 'dropdown' || filterType === 'checkbox') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Options *
              </label>
              <div className="space-y-2">
                {filterOptions.map((option, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={option}
                      onChange={(e) => handleOptionChange(index, e.target.value)}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                      placeholder={`Option ${index + 1}`}
                    />
                    {filterOptions.length > 1 && (
                      <button
                        onClick={() => handleRemoveOption(index)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                ))}
                <button
                  onClick={handleAddOption}
                  className="flex items-center text-violet-600 hover:text-violet-700 text-sm"
                >
                  <Plus size={16} className="mr-1" />
                  Add Option
                </button>
              </div>
            </div>
          )}

          {/* Range Filter Info */}
          {filterType === 'range' && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-sm text-blue-700">
                Range filters will allow users to set minimum and maximum values.
                No additional options needed.
              </p>
            </div>
          )}
        </div>

        <div className="flex space-x-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700"
          >
            Create Filter
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomFilterModal;