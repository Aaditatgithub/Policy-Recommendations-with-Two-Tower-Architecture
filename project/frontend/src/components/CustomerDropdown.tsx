import React from 'react';
import { useNavigate } from 'react-router-dom';
import { customers } from '../data/customers';
import CustomerCard from './customer/CustomerCard';

const CustomerDropdown: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-sbi-light-purple to-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-sbi-purple mb-8 text-center">
          Select Customer Profile
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {customers.map((customer) => (
            <button
              key={customer.id}
              onClick={() => navigate(`/customer/${customer.id}`)}
              className="w-full text-left transition-transform duration-200 hover:scale-105"
            >
              <CustomerCard customer={customer} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CustomerDropdown;