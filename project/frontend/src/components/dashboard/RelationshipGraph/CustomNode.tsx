import React from 'react';
import { Handle, Position } from 'reactflow';
import { NodeData } from './types';

interface CustomNodeProps {
  data: NodeData;
  style?: React.CSSProperties;
}

const CustomNode: React.FC<CustomNodeProps> = ({ data, style }) => {
  return (
    <div
      className="px-4 py-2 rounded-lg shadow-md text-white min-w-[120px] text-center"
      style={{
        ...style,
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        border: '2px solid rgba(255, 255, 255, 0.2)'
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#fff' }} />
      <span className="font-medium">{data.label}</span>
      <Handle type="source" position={Position.Bottom} style={{ background: '#fff' }} />
    </div>
  );
};

export default CustomNode;