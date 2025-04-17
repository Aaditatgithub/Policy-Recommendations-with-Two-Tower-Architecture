import { nodeColors } from './theme';

export const nodes = [
  // Family nodes
  { 
    id: '1', 
    position: { x: 250, y: 100 }, 
    data: { 
      label: 'Parents',
      type: 'person',
      category: 'family'
    },
    style: { background: nodeColors.person.family }
  },
  
  // Policy nodes
  { 
    id: '2', 
    position: { x: 100, y: 250 }, 
    data: { 
      label: 'Smart Child Future',
      type: 'policy',
      category: 'education'
    },
    style: { background: nodeColors.policy.education }
  },
  { 
    id: '3', 
    position: { x: 400, y: 250 }, 
    data: { 
      label: 'Smart Wealth Builder',
      type: 'policy',
      category: 'investment'
    },
    style: { background: nodeColors.policy.investment }
  },
  { 
    id: '4', 
    position: { x: 250, y: 400 }, 
    data: { 
      label: 'Smart Health Premier',
      type: 'policy',
      category: 'health'
    },
    style: { background: nodeColors.policy.health }
  },
  
  // Relationship nodes
  { 
    id: '5', 
    position: { x: 550, y: 250 }, 
    data: { 
      label: 'Family Protection',
      type: 'relationship'
    },
    style: { background: nodeColors.relationship.default }
  }
];

export const edges = [
  // Parent relationships
  { 
    id: 'e1-2', 
    source: '1', 
    target: '2', 
    data: { 
      label: 'Purchases for children',
      type: 'purchases',
      percentage: 75
    }
  },
  { 
    id: 'e1-3', 
    source: '1', 
    target: '3', 
    data: { 
      label: 'Invests for future',
      type: 'purchases',
      percentage: 65
    }
  },
  { 
    id: 'e1-4', 
    source: '1', 
    target: '4', 
    data: { 
      label: 'Protects family health',
      type: 'purchases',
      percentage: 85
    }
  },
  
  // Policy relationships
  { 
    id: 'e2-5', 
    source: '2', 
    target: '5', 
    data: { 
      label: 'Secures education',
      type: 'benefits',
      percentage: 100
    }
  },
  { 
    id: 'e3-5', 
    source: '3', 
    target: '5', 
    data: { 
      label: 'Builds wealth',
      type: 'benefits',
      percentage: 100
    }
  },
  { 
    id: 'e4-5', 
    source: '4', 
    target: '5', 
    data: { 
      label: 'Ensures wellbeing',
      type: 'benefits',
      percentage: 100
    }
  }
];