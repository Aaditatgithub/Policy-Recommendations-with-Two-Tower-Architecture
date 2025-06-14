import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface WaveChartProps {
  title?: string;
  className?: string;
}

const WaveChart: React.FC<WaveChartProps> = ({ title = "Policy Performance Comparison", className = "" }) => {
  // Sample data for persona approach vs new approach
  const personaData = [65, 68, 72, 70, 75, 78, 82, 85, 88, 90, 92, 95];
  const newApproachData = [70, 75, 80, 85, 88, 92, 95, 98, 102, 105, 108, 112];
  
  const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  const data = {
    labels,
    datasets: [
      {
        label: 'Persona Approach',
        data: personaData,
        borderColor: '#C084FC', // Light purple
        backgroundColor: 'rgba(192, 132, 252, 0.2)', // Light purple with transparency
        fill: true,
        tension: 0.6, // Creates smooth curves
        pointRadius: 6,
        pointHoverRadius: 8,
        pointBackgroundColor: '#C084FC',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        borderWidth: 3,
      },
      {
        label: 'Two-Tower Neural Network',
        data: newApproachData,
        borderColor: '#46166B', // Dark purple (SBI purple)
        backgroundColor: 'rgba(70, 22, 107, 0.3)', // Dark purple with transparency
        fill: true,
        tension: 0.6, // Creates smooth curves
        pointRadius: 6,
        pointHoverRadius: 8,
        pointBackgroundColor: '#46166B',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        borderWidth: 3,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#374151',
          font: {
            family: 'Inter, sans-serif',
            size: 12,
            weight: '500'
          },
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 20
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
        },
        padding: {
          bottom: 20
        }
      },
      tooltip: {
        backgroundColor: '#1F2937',
        titleColor: '#F9FAFB',
        bodyColor: '#F9FAFB',
        borderColor: '#46166B',
        borderWidth: 1,
        cornerRadius: 12,
        titleFont: {
          family: 'Inter, sans-serif',
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          family: 'Inter, sans-serif',
          size: 12
        },
        padding: 12,
        displayColors: true,
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y}% conversion rate`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(229, 231, 235, 0.5)',
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
          color: 'rgba(229, 231, 235, 0.5)',
          drawBorder: false
        },
        ticks: {
          color: '#6B7280',
          font: {
            family: 'Inter, sans-serif',
            size: 11
          },
          callback: function(value: any) {
            return value + '%';
          }
        },
        beginAtZero: false,
        min: 60,
        max: 120
      }
    },
    elements: {
      point: {
        radius: 6,
        hoverRadius: 8,
        borderWidth: 2
      },
      line: {
        tension: 0.6,
        borderWidth: 3,
        borderCapStyle: 'round' as const,
        borderJoinStyle: 'round' as const
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const
    },
    animation: {
      duration: 2000,
      easing: 'easeInOutQuart' as const
    }
  };

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
          <p className="text-sm text-gray-600 mt-1">Monthly conversion rate comparison</p>
        </div>
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-purple-400 mr-2"></div>
            <span className="text-gray-600">Persona Approach</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-purple-800 mr-2"></div>
            <span className="text-gray-600">Neural Network</span>
          </div>
        </div>
      </div>
      
      <div className="h-80">
        <Line data={data} options={options} />
      </div>
      
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="text-center p-3 bg-purple-50 rounded-lg">
          <div className="text-lg font-bold text-purple-800">+18%</div>
          <div className="text-xs text-purple-600">Improvement</div>
        </div>
        <div className="text-center p-3 bg-green-50 rounded-lg">
          <div className="text-lg font-bold text-green-800">112%</div>
          <div className="text-xs text-green-600">Peak Performance</div>
        </div>
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <div className="text-lg font-bold text-blue-800">95%</div>
          <div className="text-xs text-blue-600">Avg. Conversion</div>
        </div>
      </div>
    </div>
  );
};

export default WaveChart;