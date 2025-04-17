import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import PolicyCard from "../components/PolicyCard";
import { customers } from "../data/customers";

const CustomerProfile: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [policies, setPolicies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const customer = customers.find((customer) => customer.id === id);
  const customerName = customer ? customer.name : "Unknown Customer";

  useEffect(() => {
    const fetchPolicies = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `http://localhost:5000/recommend_policies?customer_id=${id}`
        );

        if (!response.ok) throw new Error("Failed to fetch policies");
        const data = await response.json();
        setPolicies(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (id) fetchPolicies();
  }, [id]);

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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Policies for {customerName}
          </h1>
        </header>

        <section>
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">
            Recommended Policies
          </h2>
          {policies.length === 0 ? (
            <p className="text-gray-500">No policies available</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {policies.map((policy) => (
                <PolicyCard key={policy.policy_id} policy={policy} />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default CustomerProfile;
