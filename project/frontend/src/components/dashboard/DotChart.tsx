import React from 'react';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
);

interface DotChartProps {
  data: {
    datasets: {
      label: string;
      data: { x: number; y: number }[];
      backgroundColor: string;
      borderColor: string;
      pointRadius: number;
    }[];
  };
  options?: any;
  title?: string;
}

const DotChart: React.FC<DotChartProps> = ({ data, options, title }) => {
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#374151',
          font: {
            family: 'Inter, sans-serif',
            size: 12
          }
        }
      },
      title: {
        display: !!title,
        text: title,
        color: '#1F2937',
        font: {
          family: 'Inter, sans-serif',
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: '#1F2937',
        titleColor: '#F9FAFB',
        bodyColor: '#F9FAFB',
        borderColor: '#46166B',
        borderWidth: 1,
        cornerRadius: 8,
        titleFont: {
          family: 'Inter, sans-serif'
        },
        bodyFont: {
          family: 'Inter, sans-serif'
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: '#E5E7EB',
          drawBorder: false
        },
        ticks: {
          color: '#6B7280',
          font: {
            family: 'Inter, sans-serif',
            size: 11
          }
        }
      },
      y: {
        grid: {
          color: '#E5E7EB',
          drawBorder: false
        },
        ticks: {
          color: '#6B7280',
          font: {
            family: 'Inter, sans-serif',
            size: 11
          }
        }
      }
    },
    ...options
  };

  return <Scatter data={data} options={defaultOptions} />;
};

export default DotChart;