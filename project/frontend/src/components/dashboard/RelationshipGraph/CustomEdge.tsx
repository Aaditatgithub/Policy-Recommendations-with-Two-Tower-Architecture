import React from 'react';
import { EdgeProps, getBezierPath } from 'reactflow';
import { edgeColors } from './theme';
import { EdgeData } from './types';

const CustomEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data
}) => {
  const edgeData = data as EdgeData;
  const color = edgeColors[edgeData.type || 'purchases'];
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <path
        id={id}
        style={{ stroke: color }}
        className="react-flow__edge-path"
        strokeWidth={2}
        d={edgePath}
      />
      <text>
        <textPath
          href={`#${id}`}
          className="fill-gray-700 text-sm font-medium"
          startOffset="50%"
          textAnchor="middle"
        >
          {edgeData.label}
          {edgeData.percentage && ` (${edgeData.percentage}%)`}
        </textPath>
      </text>
    </>
  );
};

export default CustomEdge;