export const addToCart = async (customerId: string, policyId: string) => {
  try {
    const response = await fetch('http://localhost:5000/cart/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId, policy_id: policyId })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to add to cart');
    }

    const result = await response.json();
    
    // Update localStorage for immediate UI feedback
    await updateLocalStorageCart(customerId);
    
    return result;
  } catch (error) {
    console.error('Error adding to cart:', error);
    throw error;
  }
};

export const removeFromCart = async (customerId: string, policyId: string) => {
  try {
    const response = await fetch('http://localhost:5000/cart/remove', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId, policy_id: policyId })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to remove from cart');
    }

    // Update localStorage
    await updateLocalStorageCart(customerId);
    
    return await response.json();
  } catch (error) {
    console.error('Error removing from cart:', error);
    throw error;
  }
};

export const getCart = async (customerId: string) => {
  try {
    const response = await fetch(`http://localhost:5000/cart?customer_id=${customerId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to fetch cart');
    }

    const data = await response.json();
    
    // Update localStorage with server data
    localStorage.setItem('cart', JSON.stringify(data.items));
    
    return data;
  } catch (error) {
    console.error('Error fetching cart:', error);
    throw error;
  }
};

export const checkoutCart = async (customerId: string) => {
  try {
    const response = await fetch('http://localhost:5000/cart/checkout', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        customer_id: customerId
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to checkout');
    }

    const result = await response.json();
    
    // Clear localStorage after successful checkout
    localStorage.removeItem('cart');
    
    return result;
  } catch (error) {
    console.error('Error during checkout:', error);
    throw error;
  }
};

// Helper function to update localStorage with server data
const updateLocalStorageCart = async (customerId: string) => {
  try {
    const cartData = await getCart(customerId);
    return cartData.items;
  } catch (error) {
    console.error('Error updating localStorage cart:', error);
    return [];
  }
};

// Get cart from localStorage (for immediate UI updates)
export const getLocalCart = () => {
  try {
    const cart = localStorage.getItem('cart');
    return cart ? JSON.parse(cart) : [];
  } catch (error) {
    console.error('Error getting local cart:', error);
    return [];
  }
};