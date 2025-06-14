// src/pages/ComparePage.tsx
import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from '../components/NavBar';
import { trackEvent } from '../utils/tracking';

const ComparePage: React.FC = () => {
  const location = useLocation();
  const policyId = new URLSearchParams(location.search).get('policy_id');
  const [policy, setPolicy] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPolicy = async () => {
      if (!policyId) return;

      try {
        setLoading(true);
        const response = await fetch(`http://localhost:5000/search_policies?q=`);
        const allPolicies = await response.json();
        const matched = allPolicies.find((p: any) => String(p.policy_id) === policyId);

        
        if (!matched) throw new Error("Policy not found");
        setPolicy(matched);

        trackEvent('compare_viewed', {
          policy_id: policyId,
          context: 'compare_page'
        });
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicy();
  }, [policyId]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <Navbar />
      <div className="p-8 max-w-4xl mx-auto">
        {loading ? (
          <div className="text-center text-gray-600">Loading...</div>
        ) : error ? (
          <div className="text-center text-red-600">{error}</div>
        ) : policy ? (
          <>
            <h1 className="text-3xl font-bold text-gray-800 mb-4">
              Compare Policy
            </h1>
            <div className="bg-white rounded-xl shadow-md p-6 space-y-4">
              <div>
                <strong className="text-gray-700">Name:</strong> {policy.policy_name}
              </div>
              <div>
                <strong className="text-gray-700">Type:</strong> {policy.policy_type}
              </div>
              <div>
                <strong className="text-gray-700">Premium:</strong> ₹{policy['premium_amount (INR)']}
              </div>
              <div>
                <strong className="text-gray-700">Sum Assured:</strong> ₹{policy['sum_assured (INR)']}
              </div>
              <div>
                <strong className="text-gray-700">Description:</strong> {policy.description}
              </div>
              <div>
                <strong className="text-gray-700">Keywords:</strong> {policy.keywords}
              </div>
            </div>
          </>
        ) : (
          <div className="text-center text-gray-500">No policy data available</div>
        )}
      </div>
    </div>
  );
};

export default ComparePage;
