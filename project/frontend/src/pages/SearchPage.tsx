import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import PolicyCard from '../components/PolicyCard';
import Navbar from '../components/NavBar';
import { trackEvent } from '../utils/tracking';
import { addToCart } from '../services/cartService';

const SearchPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const query = new URLSearchParams(location.search).get('q') || '';
  const [policies, setPolicies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [customerId, setCustomerId] = useState<string | null>(null);

  useEffect(() => {
    const storedCustomerId = localStorage.getItem('customerId');
    setCustomerId(storedCustomerId);
  }, []);

  useEffect(() => {
    if (!query) {
      setLoading(false);
      return;
    }

    const fetchPolicies = async () => {
      try {
        setLoading(true);
        trackEvent('search_initiated', { query });

        const response = await fetch(`http://localhost:5000/search_policies?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        setPolicies(data);
        trackEvent('search_completed', {
          query,
          resultCount: data.length
        });
      } catch (error) {
        trackEvent('search_error', {
          query,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        console.error('Error fetching policies:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPolicies();
  }, [query]);

  const handleSearch = (searchQuery: string) => {
    navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
  };

  const handleAddToCart = async (policyId: string) => {
    try {
      await addToCart(customerId || 'guest', policyId);
      alert('Policy added to cart successfully!');
    } catch (error) {
      console.error('Add to cart failed:', error);
      alert('Failed to add policy to cart.');
    }
  };

  const handleCompare = (policyId: string) => {
    navigate(`/compare?policy_id=${policyId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <Navbar onSearch={handleSearch} initialQuery={query} />

      <div className="p-8 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">
          Search Results for "{query}"
        </h1>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-violet-600"></div>
          </div>
        ) : policies.length === 0 ? (
          <p className="text-gray-500">No policies found matching your search</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {policies.map((policy) => (
              <PolicyCard 
                key={policy.policy_id} 
                policy={policy}
                customerId={customerId}
                onAddToCart={handleAddToCart}
                onCompare={handleCompare}
                interactionContext="search"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchPage;
