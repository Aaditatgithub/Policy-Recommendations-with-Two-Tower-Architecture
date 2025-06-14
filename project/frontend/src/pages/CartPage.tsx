import React, { useEffect, useState } from 'react';
import Navbar from '../components/NavBar';
import { getCart, removeFromCart, checkoutCart } from '../services/cartService';
import { trackEvent } from '../utils/tracking';

interface CartItem {
  policy_id: string;
  policy_name: string;
  premium_amount: number;
  added_at: string;
}

const CartPage: React.FC = () => {
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [checkingOut, setCheckingOut] = useState(false);
  const [customerId, setCustomerId] = useState<string | null>(null);

  useEffect(() => {
    const storedCustomerId = localStorage.getItem('customerId');
    setCustomerId(storedCustomerId);
    
    if (storedCustomerId) {
      fetchCart(storedCustomerId);
    } else {
      setLoading(false);
      setError('Please select a customer profile first');
    }
  }, []);

  const fetchCart = async (customerIdParam: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const cartData = await getCart(customerIdParam);
      setCartItems(cartData.items || []);
      
      trackEvent('cart_viewed', {
        customer_id: customerIdParam,
        item_count: cartData.items?.length || 0
      });
    } catch (err: any) {
      setError(err.message || 'Failed to load cart');
      console.error('Error fetching cart:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (policyId: string) => {
    if (!customerId) return;
    
    try {
      await removeFromCart(customerId, policyId);
      
      // Update local state
      setCartItems(prev => prev.filter(item => item.policy_id !== policyId));
      
      trackEvent('cart_remove', {
        customer_id: customerId,
        policy_id: policyId
      });
      
    } catch (err: any) {
      alert(err.message || 'Failed to remove item from cart');
    }
  };

  const handleCheckout = async () => {
    if (!customerId || cartItems.length === 0) return;
    
    try {
      setCheckingOut(true);
      
      const result = await checkoutCart(customerId);
      
      trackEvent('checkout_completed', {
        customer_id: customerId,
        items_count: cartItems.length,
        total_amount: cartItems.reduce((sum, item) => sum + item.premium_amount, 0)
      });
      
      alert(`Checkout successful! ${result.items_purchased} policies purchased.`);
      setCartItems([]);
      
    } catch (err: any) {
      alert(err.message || 'Checkout failed');
    } finally {
      setCheckingOut(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(amount);
  };

  const totalAmount = cartItems.reduce((sum, item) => sum + item.premium_amount, 0);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
        <Navbar />
        <div className="max-w-4xl mx-auto p-8 flex justify-center items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-violet-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <Navbar />
      <div className="max-w-4xl mx-auto p-8">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 flex items-center">
            🛒 Your Cart
            {cartItems.length > 0 && (
              <span className="ml-3 bg-violet-100 text-violet-800 text-sm px-3 py-1 rounded-full">
                {cartItems.length} item{cartItems.length !== 1 ? 's' : ''}
              </span>
            )}
          </h1>

          {error ? (
            <div className="text-center py-10">
              <div className="text-red-500 mb-4">⚠️ {error}</div>
              {!customerId && (
                <button
                  onClick={() => window.location.href = '/customer'}
                  className="bg-violet-600 text-white px-6 py-2 rounded-lg hover:bg-violet-700"
                >
                  Select Customer Profile
                </button>
              )}
            </div>
          ) : cartItems.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-6xl mb-4">🛒</div>
              <h2 className="text-xl font-semibold text-gray-600 mb-2">Your cart is empty</h2>
              <p className="text-gray-500 mb-6">Add some policies to get started!</p>
              <button
                onClick={() => window.location.href = customerId ? `/customer/${customerId}` : '/customer'}
                className="bg-violet-600 text-white px-6 py-3 rounded-lg hover:bg-violet-700 transition-colors"
              >
                Browse Policies
              </button>
            </div>
          ) : (
            <>
              <div className="space-y-4 mb-8">
                {cartItems.map((item) => (
                  <div key={item.policy_id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {item.policy_name}
                        </h3>
                        <div className="text-sm text-gray-600 mb-2">
                          Policy ID: {item.policy_id}
                        </div>
                        <div className="text-lg font-bold text-violet-600">
                          {formatCurrency(item.premium_amount)} / year
                        </div>
                        <div className="text-xs text-gray-500 mt-2">
                          Added: {new Date(item.added_at).toLocaleDateString()}
                        </div>
                      </div>
                      <button
                        onClick={() => handleRemove(item.policy_id)}
                        className="text-red-500 hover:text-red-700 hover:bg-red-50 p-2 rounded-lg transition-colors"
                        title="Remove from cart"
                      >
                        🗑️ Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="border-t border-gray-200 pt-6">
                <div className="flex justify-between items-center mb-6">
                  <div className="text-lg font-semibold text-gray-700">
                    Total Annual Premium:
                  </div>
                  <div className="text-2xl font-bold text-violet-600">
                    {formatCurrency(totalAmount)}
                  </div>
                </div>

                <div className="flex space-x-4">
                  <button
                    onClick={() => window.location.href = customerId ? `/customer/${customerId}` : '/customer'}
                    className="flex-1 bg-gray-100 text-gray-700 py-3 px-6 rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    Continue Shopping
                  </button>
                  <button
                    onClick={handleCheckout}
                    disabled={checkingOut}
                    className="flex-1 bg-violet-600 text-white py-3 px-6 rounded-lg hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {checkingOut ? 'Processing...' : 'Proceed to Checkout'}
                  </button>
                </div>

                <div className="mt-4 text-center text-sm text-gray-500">
                  🔒 Secure checkout • Your data is protected
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default CartPage;