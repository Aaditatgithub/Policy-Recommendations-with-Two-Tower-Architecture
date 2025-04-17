import React, { useEffect, useState } from 'react';
import { BarChart, PieChart, LineChart } from '../components/dashboard/Charts';
import { RecentActivity } from '../components/dashboard/RecentActivity';
import { FiRefreshCw, FiAlertCircle } from 'react-icons/fi';

interface DashboardData {
  metrics?: {
    total_policies: number;
    total_customers: number;
    conversion_rate: number;
    avg_premium: number;
  };
  policy_performance?: any;
  customer_segments?: any;
  sales_trend?: any;
  recent_activities?: any;
  top_policies?: any;
}

const Dashboard = () => {
  const [data, setData] = useState<DashboardData>({});
  const [timeRange, setTimeRange] = useState('7d');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`http://localhost:5000/analytics?range=${timeRange}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const formatNumber = (value?: number, fallback = '--') => {
    return value?.toLocaleString() ?? fallback;
  };

  if (loading) {
    return (
      <div className="p-8 flex justify-center items-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-violet-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center text-red-500">
        <FiAlertCircle className="inline-block text-2xl mb-2" />
        <p>Error loading dashboard: {error}</p>
        <button 
          onClick={fetchData}
          className="mt-4 px-4 py-2 bg-violet-600 text-white rounded-md"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <div className="flex space-x-4">
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-white border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
            <button 
              className="bg-violet-600 text-white px-4 py-2 rounded-md flex items-center"
              onClick={fetchData}
            >
              <FiRefreshCw className="mr-2" /> Refresh
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-violet-500">
            <h3 className="text-gray-500 text-sm font-medium">Total Policies</h3>
            <p className="text-3xl font-bold mt-2">{formatNumber(data.metrics?.total_policies)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-blue-500">
            <h3 className="text-gray-500 text-sm font-medium">Active Customers</h3>
            <p className="text-3xl font-bold mt-2">{formatNumber(data.metrics?.total_customers)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-green-500">
            <h3 className="text-gray-500 text-sm font-medium">Conversion Rate</h3>
            <p className="text-3xl font-bold mt-2">
              {data.metrics?.conversion_rate ? `${data.metrics.conversion_rate.toFixed(1)}%` : '--'}
            </p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border-l-4 border-yellow-500">
            <h3 className="text-gray-500 text-sm font-medium">Avg. Premium</h3>
            <p className="text-3xl font-bold mt-2">
              {data.metrics?.avg_premium ? `₹${formatNumber(data.metrics.avg_premium)}` : '--'}
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Policy Performance</h2>
            {data.policy_performance ? (
              <BarChart data={data.policy_performance} />
            ) : (
              <p className="text-gray-500 text-center py-10">No policy performance data</p>
            )}
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Customer Segments</h2>
            {data.customer_segments ? (
              <PieChart data={data.customer_segments} />
            ) : (
              <p className="text-gray-500 text-center py-10">No customer segment data</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Sales Trend</h2>
            {data.sales_trend ? (
              <LineChart data={data.sales_trend} />
            ) : (
              <p className="text-gray-500 text-center py-10">No sales trend data</p>
            )}
          </div>
          <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
            {data.recent_activities ? (
              <RecentActivity activities={data.recent_activities} />
            ) : (
              <p className="text-gray-500 text-center py-10">No recent activity</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;