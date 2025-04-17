import React from 'react';
import { User, Briefcase, MapPin, CreditCard } from 'lucide-react';
import { Customer } from '../../data/customers';

interface CustomerCardProps {
  customer: Customer;
}

const CustomerCard: React.FC<CustomerCardProps> = ({ customer }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-sbi-purple">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-semibold text-sbi-purple">{customer.name}</h3>
        <span className={`px-3 py-1 rounded-full text-sm ${
          customer.maritalStatus === 'Married' 
            ? 'bg-sbi-pink/10 text-sbi-pink' 
            : 'bg-sbi-blue/10 text-sbi-blue'
        }`}>
          {customer.maritalStatus}
        </span>
      </div>
      
      <div className="space-y-2">
        <div className="flex items-center text-gray-600">
          <User size={16} className="mr-2 text-sbi-purple" />
          <span>Age: {customer.age} • Family Size: {customer.familySize}</span>
        </div>
        <div className="flex items-center text-gray-600">
          <Briefcase size={16} className="mr-2 text-sbi-purple" />
          <span>{customer.occupation}</span>
        </div>
        <div className="flex items-center text-gray-600">
          <MapPin size={16} className="mr-2 text-sbi-purple" />
          <span>{customer.city}</span>
        </div>
        <div className="flex items-center text-gray-600">
          <CreditCard size={16} className="mr-2 text-sbi-purple" />
          <span>Income: {customer.income}</span>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-100">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Risk Profile</span>
          <span className={`px-3 py-1 rounded-full text-sm ${
            customer.riskProfile === 'Conservative' 
              ? 'bg-blue-100 text-blue-700'
              : customer.riskProfile === 'Moderate'
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-red-100 text-red-700'
          }`}>
            {customer.riskProfile}
          </span>
        </div>
      </div>
    </div>
  );
};

export default CustomerCard;