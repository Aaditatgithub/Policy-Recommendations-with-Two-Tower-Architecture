// src/components/dashboard/Charts.tsx
import React from 'react';
import { Bar, Pie, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor: string[];
      borderColor?: string[];
      borderWidth?: number;
    }[];
  };
  options?: any;
}

export const BarChart = ({ data, options }: ChartProps) => {
  const defaultOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    ...options
  };

  return <Bar data={data} options={defaultOptions} />;
};

export const PieChart = ({ data, options }: ChartProps) => {
  const defaultOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const,
      },
    },
    ...options
  };

  return <Pie data={data} options={defaultOptions} />;
};

export const LineChart = ({ data, options }: ChartProps) => {
  const defaultOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    ...options
  };

  return <Line data={data} options={defaultOptions} />;
};