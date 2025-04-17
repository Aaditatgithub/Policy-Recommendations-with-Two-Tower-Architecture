import React from 'react';
import { Link } from 'react-router-dom';

interface WelcomeCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  to: string;
  bgColor: string;
}

const WelcomeCard: React.FC<WelcomeCardProps> = ({ 
  title, 
  description, 
  icon, 
  to,
  bgColor 
}) => {
  return (
    <Link
      to={to}
      className="flex flex-col items-center p-8 bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 w-full max-w-sm mx-4 mb-6 group"
    >
      <div className={`p-4 rounded-full mb-4 ${bgColor} group-hover:scale-110 transition-transform duration-300`}>
        {icon}
      </div>
      <h2 className="text-2xl font-semibold text-sbi-purple mb-3">{title}</h2>
      <p className="text-gray-600 text-center">{description}</p>
    </Link>
  );
};

export default WelcomeCard;