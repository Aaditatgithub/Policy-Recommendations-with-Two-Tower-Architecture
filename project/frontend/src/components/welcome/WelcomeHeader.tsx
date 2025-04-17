import React from 'react';
import Logo from '../common/Logo';

const WelcomeHeader: React.FC = () => {
  return (
    <header className="text-center mb-12">
      <div className="flex justify-center mb-8">
        <Logo />
      </div>
      <h1 className="text-4xl font-bold text-sbi-purple mb-4">
        AI-Powered Insurance Recommendations
      </h1>
      <p className="text-xl text-gray-600">
        Personalized insurance solutions powered by advanced AI
      </p>
    </header>
  );
}

export default WelcomeHeader;