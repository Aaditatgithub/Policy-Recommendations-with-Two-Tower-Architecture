// src/components/CartSidebar.tsx
import React, { useState, useEffect } from 'react';
import { FiShoppingCart, FiX, FiAlertTriangle } from 'react-icons/fi';

const CartSidebar = ({ customerId, onClose }) => {
  const [cartItems, setCartItems] = useState([]);
  const [isCheckingOut, setIsCheckingOut] = useState(false);
  
  useEffect(() => {
    const fetchCart = async () => {
      const response = await fetch(`http://localhost:5000/cart?customer_id=${customerId}`);
      const data = await response.json();
      setCartItems(data.items);
    };
    fetchCart();
    
    // Track cart abandonment
    const handleBeforeUnload = () => {
      if (cartItems.length > 0) {
        fetch(`http://localhost:5000/track_abandoned_cart`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ customerId, items: cartItems })
        });
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [customerId, cartItems.length]);

  const handleCheckout = async () => {
    setIsCheckingOut(true);
    try {
      const response = await fetch(`http://localhost:5000/checkout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customerId, items: cartItems })
      });
      // Handle successful checkout
    } finally {
      setIsCheckingOut(false);
    }
  };

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white shadow-xl z-50 flex flex-col">
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="text-lg font-semibold flex items-center">
          <FiShoppingCart className="mr-2" /> Your Cart ({cartItems.length})
        </h3>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
          <FiX size={20} />
        </button>
      </div>
      
      <div className="flex-grow overflow-y-auto p-4">
        {cartItems.length === 0 ? (
          <div className="text-center py-10 text-gray-500">
            Your cart is empty
          </div>
        ) : (
          <ul className="space-y-4">
            {cartItems.map(item => (
              <li key={item.policy_id} className="border-b pb-4">
                <h4 className="font-medium">{item.policy_name}</h4>
                <p className="text-sm text-gray-600">₹{item.premium_amount.toLocaleString()}</p>
                <button 
                  className="text-red-500 text-xs mt-1"
                  onClick={() => removeFromCart(item.policy_id)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      
      <div className="p-4 border-t">
        <div className="flex justify-between mb-4">
          <span>Total:</span>
          <span className="font-semibold">
            ₹{cartItems.reduce((sum, item) => sum + item.premium_amount, 0).toLocaleString()}
          </span>
        </div>
        <button
          onClick={handleCheckout}
          disabled={cartItems.length === 0 || isCheckingOut}
          className={`w-full py-2 rounded-md text-white ${cartItems.length === 0 ? 'bg-gray-400' : 'bg-violet-600 hover:bg-violet-700'}`}
        >
          {isCheckingOut ? 'Processing...' : 'Proceed to Checkout'}
        </button>
        
        {cartItems.length > 0 && (
          <div className="mt-3 flex items-center text-yellow-600 text-sm">
            <FiAlertTriangle className="mr-1" />
            <span>Your cart will be saved for 30 minutes</span>
          </div>
        )}
      </div>
    </div>
  );
};