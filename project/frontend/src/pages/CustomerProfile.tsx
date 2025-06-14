import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import PolicyCard from "../components/PolicyCard";
import Navbar from "../components/NavBar";
import { customers } from "../data/customers";
import { addToCart } from "../services/cartService";
import { trackEvent } from "../utils/tracking";

const CustomerProfile: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [recommendedPolicies, setRecommendedPolicies] = useState<any[]>([]);
  const [allPolicies, setAllPolicies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const customer = customers.find((customer) => customer.id === id);
  const customerName = customer ? customer.name : "Unknown Customer";

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch recommended policies
        const recResponse = await fetch(
          `http://localhost:5000/recommend_policies?customer_id=${id}`
        );
        if (!recResponse.ok) throw new Error("Failed to fetch recommended policies");
        const recData = await recResponse.json();
        setRecommendedPolicies(recData);

        // Fetch all policies
        const allResponse = await fetch(
          `http://localhost:5000/search_policies?q=`
        );
        if (!allResponse.ok) throw new Error("Failed to fetch all policies");
        const allData = await allResponse.json();
        setAllPolicies(allData);

      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchData();
  }, [id]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    navigate(`/search?q=${encodeURIComponent(query)}`);
  };

  const handleAddToCart = async (policyId: string) => {
    try {
      await addToCart(id || 'guest', policyId);
      trackEvent('cart_add', {
        customer_id: id,
        policy_id: policyId,
        context: 'customer_profile'
      });
      alert('Policy added to cart successfully!');
    } catch (error) {
      console.error('Failed to add to cart:', error);
      alert('Failed to add policy to cart');
    }
  };

  const handleCompare = (policyId: string) => {
    trackEvent('compare_initiated', {
      customer_id: id,
      policy_id: policyId,
      context: 'customer_profile'
    });
    navigate(`/compare?policy_id=${policyId}`);
  };

  const filteredAllPolicies = allPolicies.filter(policy => 
    !recommendedPolicies.some(recPolicy => recPolicy.policy_id === policy.policy_id)
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white p-8 flex justify-center items-center">
        <span className="text-gray-600 text-lg">Loading...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white p-8">
        <div className="text-center text-gray-600">{error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <Navbar userType="customer" onSearch={handleSearch} />
      
      <div className="p-8 max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Policies for {customerName}
          </h1>
          <p className="text-gray-600">Customer ID: {id}</p>
        </header>

        {recommendedPolicies.length > 0 && (
          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">
              Recommended For You
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendedPolicies.map((policy) => (
                <PolicyCard 
                  key={policy.policy_id} 
                  policy={policy} 
                  customerId={id}
                  onAddToCart={handleAddToCart}
                  onCompare={handleCompare}
                  interactionContext="recommendation"
                />
              ))}
            </div>
          </section>
        )}

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">
            All Available Policies
          </h2>
          {filteredAllPolicies.length === 0 ? (
            <p className="text-gray-500">No policies available</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredAllPolicies.map((policy) => (
                <PolicyCard 
                  key={policy.policy_id} 
                  policy={policy} 
                  customerId={id}
                  onAddToCart={handleAddToCart}
                  onCompare={handleCompare}
                  interactionContext="browse"
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default CustomerProfile;