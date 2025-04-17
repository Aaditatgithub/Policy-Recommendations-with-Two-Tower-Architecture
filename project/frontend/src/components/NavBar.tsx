// src/components/Navbar.tsx
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { trackEvent } from '../utils/tracking';

interface NavbarProps {
  userType?: 'customer' | 'admin';
  onSearch?: (query: string) => void;
  initialQuery?: string;
}

const Navbar: React.FC<NavbarProps> = ({ 
  userType = 'customer', 
  onSearch,
  initialQuery = ''
}) => {
  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const navigate = useNavigate();

  useEffect(() => {
    setSearchQuery(initialQuery);
  }, [initialQuery]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      trackEvent('search_executed', {
        query: searchQuery,
        source: 'navbar'
      });
      
      if (onSearch) {
        onSearch(searchQuery);
      } else {
        navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
      }
    }
  };

  return (
    <nav className="bg-violet-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link 
              to="/" 
              className="text-xl font-bold"
              onClick={() => trackEvent('nav_home_click')}
            >
              SBI Life Insurance
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            {userType === 'admin' ? (
              <>
                <Link 
                  to="/dashboard" 
                  className="hover:bg-violet-700 px-3 py-2 rounded"
                  onClick={() => trackEvent('nav_dashboard_click')}
                >
                  Dashboard
                </Link>
                <Link 
                  to="/policies" 
                  className="hover:bg-violet-700 px-3 py-2 rounded"
                  onClick={() => trackEvent('nav_policies_click')}
                >
                  All Policies
                </Link>
              </>
            ) : (
              <>
                <Link 
                  to="/customer" 
                  className="hover:bg-violet-700 px-3 py-2 rounded"
                  onClick={() => trackEvent('nav_profile_click')}
                >
                  My Profile
                </Link>
                <Link 
                  to="/cart" 
                  className="hover:bg-violet-700 px-3 py-2 rounded"
                  onClick={() => trackEvent('nav_cart_click')}
                >
                  Cart
                </Link>
              </>
            )}
            
            <form onSubmit={handleSearch} className="flex">
              <input
                type="text"
                placeholder="Search policies..."
                className="px-3 py-2 text-gray-800 rounded-l focus:outline-none w-64"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  trackEvent('search_input', { query: e.target.value });
                }}
              />
              <button 
                type="submit"
                className="bg-violet-600 hover:bg-violet-500 px-4 py-2 rounded-r"
                onClick={() => trackEvent('search_button_click')}
              >
                Search
              </button>
            </form>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;