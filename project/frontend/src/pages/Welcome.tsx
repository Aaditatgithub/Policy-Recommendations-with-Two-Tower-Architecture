import React from 'react';
import { Users2, LineChart, FileText } from 'lucide-react';
import WelcomeCard from '../components/WelcomeCard';
import WelcomeHeader from '../components/welcome/WelcomeHeader';

const Welcome: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-sbi-light-purple to-white">
      <div className="container mx-auto px-4 py-12">
        <WelcomeHeader />

        <div className="flex flex-wrap justify-center gap-8 max-w-5xl mx-auto">
          <WelcomeCard
            title="Customer Experience"
            description="View personalized policy recommendations tailored to individual users"
            icon={<Users2 size={32} className="text-white" />}
            to="/customer"
            bgColor="bg-sbi-purple"
          />
          <WelcomeCard
            title="All Insurance Plans"
            description="Explore our comprehensive range of insurance solutions"
            icon={<FileText size={32} className="text-white" />}
            to="/policies"
            bgColor="bg-sbi-pink"
          />
          <WelcomeCard
            title="SBI Team Dashboard"
            description="Explore insights, performance metrics, and customer behavior analytics"
            icon={<LineChart size={32} className="text-white" />}
            to="/dashboard"
            bgColor="bg-sbi-blue"
          />
        </div>

        <footer className="text-center mt-16">
          <p className="text-gray-600">
            Powered by advanced AI for smarter insurance decisions
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Welcome;