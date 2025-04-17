export interface NodeData {
  label: string;
  type: 'person' | 'policy' | 'relationship';
  category?: 'family' | 'health' | 'investment' | 'education';
}

export interface EdgeData {
  label: string;
  type: 'purchases' | 'recommends' | 'benefits';
  percentage?: number;
}