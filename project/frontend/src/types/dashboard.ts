// src/types/dashboard.ts
export interface ChartData {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string[];
      borderColor?: string[];
      borderWidth?: number;
    }[];
  }
  
  export interface Activity {
    id: string;
    type: 'purchase' | 'view' | 'payment' | 'signup';
    customer: string;
    policy: string;
    time: string;
  }
  
  export interface PolicyStat {
    id: string;
    name: string;
    views: number;
    conversions: number;
  }
  
  export interface CustomerSegment {
    id: string;
    name: string;
    value: number;
    color: string;
  }