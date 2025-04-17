import React from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { nodes, edges } from './data';
import CustomNode from './CustomNode';
import CustomEdge from './CustomEdge';
import Legend from './Legend';

const nodeTypes = {
  custom: CustomNode,
};

const edgeTypes = {
  custom: CustomEdge,
};

const RelationshipGraph: React.FC = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Policy Relationship Patterns</h2>
      <div className="h-[500px] border border-gray-200 rounded-lg">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
      <Legend />
    </div>
  );
};

export default RelationshipGraph;