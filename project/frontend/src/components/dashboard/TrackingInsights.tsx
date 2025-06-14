import React from 'react';
import { Activity, Eye, MousePointer, ShoppingCart, TrendingUp } from 'lucide-react';

interface TrackingInsightsProps {
  recentActivities: any[];
  topPolicies: any[];
}

const TrackingInsights: React.FC<TrackingInsightsProps> = ({ recentActivities, topPolicies }) => {
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'view':
        return <Eye className="text-blue-500\" size={16} />;
      case 'click':
        return <MousePointer className="text-green-500" size={16} />;
      case 'cart':
        return <ShoppingCart className="text-purple-500" size={16} />;
      default:
        return <Activity className="text-gray-500" size={16} />;
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / (1000 * 60));
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      
      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      return date.toLocaleDateString();
    } catch {
      return 'Recently';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Real-time Activity Feed */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center gap-2 mb-6">
          <Activity className="text-blue-600" />
          <h2 className="text-xl font-semibold">Live Activity Feed</h2>
          <div className="ml-auto">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-500">Live</span>
            </div>
          </div>
        </div>

        <div className="space-y-4 max-h-96 overflow-y-auto">
          {recentActivities.length === 0 ? (
  <div className="text-center py-8 text-gray-500">
    <Activity className="mx-auto mb-2\" size={32} />
    <p>No recent activity</p>
  </div>
) : (
  // Filter out activities with anonymous or missing data
  recentActivities
    .filter(activity => 
      activity.customer && 
      activity.customer !== 'anonymous' && 
      activity.customer !== 'Anonymous' &&
      activity.policy && 
      activity.policy !== 'anonymous' && 
      activity.policy !== 'Anonymous'
    ).map((activity, index) => (
              <div key={activity.id || index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="mt-1">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-start">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {activity.customer}
                    </p>
                    <span className="text-xs text-gray-500 whitespace-nowrap ml-2">
                      {formatTime(activity.time)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 truncate">
                    {activity.type === 'view' ? 'Viewed' : 
                     activity.type === 'click' ? 'Clicked on' :
                     activity.type === 'cart' ? 'Added to cart' : 'Interacted with'} 
                    <span className="font-medium ml-1">{activity.policy}</span>
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Top Performing Policies */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="text-green-600" />
          <h2 className="text-xl font-semibold">Top Performing Policies</h2>
        </div>

        <div className="space-y-4">
          {topPolicies.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <TrendingUp className="mx-auto mb-2\" size={32} />
              <p>No policy data available</p>
            </div>
          ) : (
            topPolicies.slice(0, 8).map((policy, index) => (
              <div key={policy.id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                    index === 0 ? 'bg-yellow-500' :
                    index === 1 ? 'bg-gray-400' :
                    index === 2 ? 'bg-orange-500' :
                    'bg-blue-500'
                  }`}>
                    {index + 1}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {policy.name}
                  </p>
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>{policy.views} views</span>
                    <span className="flex items-center gap-1">
                      <TrendingUp size={12} className="text-green-500" />
                      {policy.conversions}% conversion
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold text-gray-900">
                    {policy.views}
                  </div>
                  <div className="text-xs text-gray-500">interactions</div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default TrackingInsights;