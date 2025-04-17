import React from 'react';

const Logo: React.FC = () => {
  return (
    <div className="flex items-center gap-2">
      <img
        src="https://www.sbilife.co.in/sites/SBILife/NewHomePage/images/SBI_Logo.png"
        alt="SBI Life Insurance"
        className="h-12"
      />
      <span className="text-sm text-sbi-purple font-medium">Apne Liye. Apno Ke Liye.</span>
    </div>
  );
};

export default Logo;