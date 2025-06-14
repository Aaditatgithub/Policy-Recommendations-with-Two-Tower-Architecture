import React, { useEffect, useState } from 'react';
import { BarChart, PieChart, LineChart } from '../components/dashboard/Charts';
import AreaChart from '../components/dashboard/AreaChart';
import DonutChart from '../components/dashboard/DonutChart';
import DotChart from '../components/dashboard/DotChart';
import WaveChart from '../components/dashboard/WaveChart';
import SankeyChart from '../components/dashboard/SankeyChart';
import { RecentActivity } from '../components/dashboard/RecentActivity';
import TrackingInsights from '../components/dashboard/TrackingInsights';
import Navbar from '../components/NavBar';
import { FiRefreshCw, FiAlertCircle, FiActivity, FiUsers, FiTrendingUp, FiDollarSign } from 'react-icons/fi';

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
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

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
      setLastUpdated(new Date());
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

  const formatCurrency = (value?: number) => {
    if (!value) return '₹--';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);
  };

  // Sample data for new charts with SBI Life theme colors
  const sbiColors = {
    primary: '#46166B',
    secondary: '#E31E54',
    tertiary: '#007DC5',
    light: '#F5F3F7',
    gradients: {
      purple: 'rgba(70, 22, 107, 0.8)',
      pink: 'rgba(227, 30, 84, 0.8)',
      blue: 'rgba(0, 125, 197, 0.8)',
      purpleLight: 'rgba(70, 22, 107, 0.2)',
      pinkLight: 'rgba(227, 30, 84, 0.2)',
      blueLight: 'rgba(0, 125, 197, 0.2)'
    }
  };

  // Customer Journey Sankey Data
  const customerJourneyData = {
    nodes: [
      // Awareness Stage
      { id: 'website_visitors', name: 'Website Visitors', category: 'awareness' },
      { id: 'social_media', name: 'Social Media', category: 'awareness' },
      { id: 'referrals', name: 'Referrals', category: 'awareness' },
      
      // Interest Stage
      { id: 'policy_browsers', name: 'Policy Browsers', category: 'interest' },
      { id: 'quote_requests', name: 'Quote Requests', category: 'interest' },
      
      // Consideration Stage
      { id: 'comparison_users', name: 'Policy Comparers', category: 'consideration' },
      { id: 'cart_additions', name: 'Cart Additions', category: 'consideration' },
      
      // Purchase Stage
      { id: 'applications', name: 'Applications', category: 'purchase' },
      { id: 'purchases', name: 'Purchases', category: 'purchase' },
      
      // Retention Stage
      { id: 'renewals', name: 'Renewals', category: 'retention' },
      { id: 'upsells', name: 'Upsells', category: 'retention' }
    ],
    links: [
      // Awareness to Interest
      { source: 'website_visitors', target: 'policy_browsers', value: 8500 },
      { source: 'social_media', target: 'policy_browsers', value: 3200 },
      { source: 'referrals', target: 'policy_browsers', value: 1800 },
      { source: 'policy_browsers', target: 'quote_requests', value: 4200 },
      
      // Interest to Consideration
      { source: 'quote_requests', target: 'comparison_users', value: 2800 },
      { source: 'comparison_users', target: 'cart_additions', value: 1900 },
      
      // Consideration to Purchase
      { source: 'cart_additions', target: 'applications', value: 1200 },
      { source: 'applications', target: 'purchases', value: 850 },
      
      // Purchase to Retention
      { source: 'purchases', target: 'renewals', value: 680 },
      { source: 'purchases', target: 'upsells', value: 170 }
    ]
  };

  const claimsAreaData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Claims Opened',
        data: [120, 145, 168, 184, 212, 195],
        backgroundColor: sbiColors.gradients.purpleLight,
        borderColor: sbiColors.primary,
        fill: true,
        tension: 0.4
      },
      {
        label: 'Claims Settled',
        data: [100, 130, 155, 170, 190, 185],
        backgroundColor: sbiColors.gradients.pinkLight,
        borderColor: sbiColors.secondary,
        fill: true,
        tension: 0.4
      }
    ]
  };

  const policyTypeDonutData = {
    labels: ['Term Insurance', 'Health Insurance', 'ULIP', 'Endowment', 'Money Back'],
    datasets: [
      {
        label: 'Policy Distribution',
        data: [35, 25, 20, 15, 5],
        backgroundColor: [
          sbiColors.primary,
          sbiColors.secondary,
          sbiColors.tertiary,
          sbiColors.gradients.purple,
          sbiColors.gradients.pink
        ],
        borderColor: [
          sbiColors.primary,
          sbiColors.secondary,
          sbiColors.tertiary,
          sbiColors.primary,
          sbiColors.secondary
        ],
        borderWidth: 2
      }
    ]
  };

  const customerEngagementDotData = {
    datasets: [
      {
        label: 'Customer Engagement',
        data: [
          { x: 25, y: 85 },
          { x: 35, y: 92 },
          { x: 45, y: 78 },
          { x: 55, y: 88 },
          { x: 30, y: 95 },
          { x: 40, y: 82 },
          { x: 50, y: 90 }
        ],
        backgroundColor: sbiColors.secondary,
        borderColor: sbiColors.primary,
        pointRadius: 8
      }
    ]
  };

  const premiumTrendAreaData = {
    labels: ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024'],
    datasets: [
      {
        label: 'Premium Collection (Crores)',
        data: [450, 520, 480, 650, 580, 720],
        backgroundColor: sbiColors.gradients.blueLight,
        borderColor: sbiColors.tertiary,
        fill: true,
        tension: 0.4
      }
    ]
  };

  if (loading && !data.metrics) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar userType="admin" />
        <div className="p-8 flex justify-center items-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-violet-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error && !data.metrics) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar userType="admin" />
        <div className="p-8 text-center text-red-500">
          <FiAlertCircle className="inline-block text-2xl mb-2" />
          <p>Error loading dashboard: {error}</p>
          <button 
            onClick={fetchData}
            className="mt-4 px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar userType="admin" />
      
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">SBI Life Analytics Dashboard</h1>
            <p className="text-gray-600 mt-1">
              Last updated: {lastUpdated.toLocaleTimeString()}
              {loading && <span className="ml-2 text-blue-600">• Refreshing...</span>}
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3">
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-white border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
            <button 
              className="bg-violet-600 text-white px-4 py-2 rounded-md flex items-center gap-2 hover:bg-violet-700 transition-colors text-sm"
              onClick={fetchData}
              disabled={loading}
            >
              <FiRefreshCw className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-violet-500">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-gray-500 text-sm font-medium">Total Policies</h3>
                <p className="text-3xl font-bold mt-2 text-gray-900">
                  {formatNumber(data.metrics?.total_policies)}
                </p>
              </div>
              <FiActivity className="text-violet-500 text-2xl" />
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-gray-500 text-sm font-medium">Active Customers</h3>
                <p className="text-3xl font-bold mt-2 text-gray-900">
                  {formatNumber(data.metrics?.total_customers)}
                </p>
              </div>
              <FiUsers className="text-blue-500 text-2xl" />
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-gray-500 text-sm font-medium">Conversion Rate</h3>
                <p className="text-3xl font-bold mt-2 text-gray-900">
                  {data.metrics?.conversion_rate ? `${data.metrics.conversion_rate.toFixed(1)}%` : '--'}
                </p>
              </div>
              <FiTrendingUp className="text-green-500 text-2xl" />
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-yellow-500">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-gray-500 text-sm font-medium">Avg. Premium</h3>
                <p className="text-3xl font-bold mt-2 text-gray-900">
                  {formatCurrency(data.metrics?.avg_premium)}
                </p>
              </div>
              <FiDollarSign className="text-yellow-500 text-2xl" />
            </div>
          </div>
        </div>

        {/* Customer Journey Sankey Diagram */}
        <div className="mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Customer Journey Flow</h2>
                <p className="text-sm text-gray-600 mt-1">Track customer progression from awareness to retention</p>
              </div>
              <div className="flex items-center space-x-4 text-xs">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-blue-200 mr-2"></div>
                  <span className="text-gray-600">Awareness</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-blue-400 mr-2"></div>
                  <span className="text-gray-600">Interest</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-blue-600 mr-2"></div>
                  <span className="text-gray-600">Consideration</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-blue-800 mr-2"></div>
                  <span className="text-gray-600">Purchase</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-blue-900 mr-2"></div>
                  <span className="text-gray-600">Retention</span>
                </div>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <SankeyChart 
                data={customerJourneyData}
                width={800}
                height={400}
                title=""
              />
            </div>
            
            <div className="mt-6 grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <div className="text-lg font-bold text-green-800">6.3%</div>
                <div className="text-xs text-green-600">Overall Conversion</div>
              </div>
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-lg font-bold text-blue-800">13,500</div>
                <div className="text-xs text-blue-600">Total Visitors</div>
              </div>
              <div className="text-center p-3 bg-purple-50 rounded-lg">
                <div className="text-lg font-bold text-purple-800">80%</div>
                <div className="text-xs text-purple-600">Retention Rate</div>
              </div>
            </div>
          </div>
        </div>

        {/* Wave Chart - Featured */}
        {/* <div className="mb-8">
          <WaveChart title="AI Model Performance Comparison" />
        </div> */}

        {/* Tracking Insights */}
        <div className="mb-8">
          <TrackingInsights 
            recentActivities={data.recent_activities || []}
            topPolicies={data.top_policies || []}
          />
        </div>

        {/* Charts Section - Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Claims Trend</h2>
            <div className="h-80">
              <AreaChart 
                data={claimsAreaData}
                title="Monthly Claims Overview"
              />
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Policy Type Distribution</h2>
            <div className="h-80">
              <DonutChart 
                data={policyTypeDonutData}
                title="Active Policies by Type"
              />
            </div>
          </div>
        </div>

        {/* Charts Section - Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Premium Collection Trend</h2>
            <div className="h-80">
              <AreaChart 
                data={premiumTrendAreaData}
                title="Quarterly Premium Collection"
              />
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Customer Engagement</h2>
            <div className="h-80">
              <DotChart 
                data={customerEngagementDotData}
                title="Age vs Engagement Score"
                options={{
                  scales: {
                    x: {
                      title: {
                        display: true,
                        text: 'Age',
                        color: '#374151',
                        font: {
                          family: 'Inter, sans-serif',
                          size: 12,
                          weight: 'bold'
                        }
                      }
                    },
                    y: {
                      title: {
                        display: true,
                        text: 'Engagement Score',
                        color: '#374151',
                        font: {
                          family: 'Inter, sans-serif',
                          size: 12,
                          weight: 'bold'
                        }
                      }
                    }
                  }
                }}
              />
            </div>
          </div>
        </div>

        {/* Original Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Policy Performance</h2>
            {data.policy_performance ? (
              <div className="h-80">
                <BarChart data={data.policy_performance} />
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <FiActivity className="mx-auto text-3xl mb-2" />
                  <p>No policy performance data</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Customer Segments</h2>
            {data.customer_segments ? (
              <div className="h-80">
                <PieChart data={data.customer_segments} />
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <FiUsers className="mx-auto text-3xl mb-2" />
                  <p>No customer segment data</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Sales Trend</h2>
            {data.sales_trend ? (
              <div className="h-64">
                <LineChart data={data.sales_trend} />
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <FiTrendingUp className="mx-auto text-3xl mb-2" />
                  <p>No sales trend data</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Recent Customer Activity</h2>
            <div className="h-64 overflow-y-auto">
              {data.recent_activities ? (
                <RecentActivity activities={data.recent_activities} />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <FiActivity className="mx-auto text-3xl mb-2" />
                    <p>No recent activity</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;