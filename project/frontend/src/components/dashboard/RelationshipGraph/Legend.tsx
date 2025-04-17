import React from 'react';
import { nodeColors, edgeColors } from './theme';

const Legend: React.FC = () => {
  return (
    <div className="flex flex-wrap gap-4 mt-4 text-sm">
      <div className="space-y-2">
        <h3 className="font-medium">Nodes</h3>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: nodeColors.person.family }} />
          <span>Family Members</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: nodeColors.policy.health }} />
          <span>Health Policies</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: nodeColors.policy.investment }} />
          <span>Investment Policies</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: nodeColors.policy.education }} />
          <span>Education Policies</span>
        </div>
      </div>
      
      <div className="space-y-2">
        <h3 className="font-medium">Relationships</h3>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: edgeColors.purchases }} />
          <span>Purchases</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: edgeColors.benefits }} />
          <span>Benefits</span>
        </div>
      </div>
    </div>
  );
};

export default Legend;