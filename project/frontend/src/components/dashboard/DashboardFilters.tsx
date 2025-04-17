import React from 'react';
import { Filter, Search } from 'lucide-react';

const DashboardFilters: React.FC = () => {
  return (
    <div className="flex flex-wrap gap-4 items-center mb-6">
      <div className="flex-1 min-w-[200px]">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Search policies or segments..."
            className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
      
      <div className="flex gap-4">
        <select className="px-4 py-2 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All Regions</option>
          <option value="north">North</option>
          <option value="south">South</option>
          <option value="east">East</option>
          <option value="west">West</option>
        </select>

        <select className="px-4 py-2 rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="">All Policy Types</option>
          <option value="life">Life Insurance</option>
          <option value="health">Health Insurance</option>
          <option value="term">Term Insurance</option>
        </select>

        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Filter size={20} />
          More Filters
        </button>
      </div>
    </div>
  );
}

export default DashboardFilters;