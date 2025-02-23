import React, { useState } from 'react';
import { allPolicies } from '../data/policies';
import PolicyCard from './policy/PolicyCard';

const PolicyList: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const policiesPerPage = 6;

  const indexOfLastPolicy = currentPage * policiesPerPage;
  const indexOfFirstPolicy = indexOfLastPolicy - policiesPerPage;
  const currentPolicies = allPolicies.slice(indexOfFirstPolicy, indexOfLastPolicy);

  const totalPages = Math.ceil(allPolicies.length / policiesPerPage);

  return (
    <div className="min-h-screen bg-sbi-light-purple p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-sbi-purple mb-8">All Insurance Plans</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {currentPolicies.map((policy) => (
            <PolicyCard key={policy.id} policy={policy} />
          ))}
        </div>
        
        {/* Pagination Controls */}
        <div className="flex justify-center mt-8 space-x-4">
          <button 
            onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))} 
            disabled={currentPage === 1} 
            className="px-4 py-2 bg-sbi-purple text-white rounded disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sbi-purple font-semibold">Page {currentPage} of {totalPages}</span>
          <button 
            onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))} 
            disabled={currentPage === totalPages} 
            className="px-4 py-2 bg-sbi-purple text-white rounded disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};

export default PolicyList;
