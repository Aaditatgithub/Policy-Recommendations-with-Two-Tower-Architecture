// src/utils/tracking.ts
import { v4 as uuidv4 } from 'uuid';

interface TrackingEvent {
  eventType: string;
  [key: string]: any;
}

let sessionId = uuidv4();
let currentPage = window.location.pathname;
let lastInteractionTime = Date.now();
let idleTimer: NodeJS.Timeout;
let viewTimers: Record<string, { startTime: number; timer: NodeJS.Timeout }> = {};

// Initialize tracking
export const initTracking = () => {
  sessionId = uuidv4();
  currentPage = window.location.pathname;
  lastInteractionTime = Date.now();
  
  // Send initial page view event
  trackPageView();
  
  // Set up idle detection (5 minutes)
  idleTimer = setInterval(checkIdleTime, 300000);
  
  // Reset idle timer on user activity
  const activityEvents = ['mousemove', 'keydown', 'scroll', 'click', 'touchstart'];
  activityEvents.forEach(event => {
    window.addEventListener(event, resetIdleTimer, { passive: true });
  });

  // Clean up on page unload
  window.addEventListener('beforeunload', cleanupTracking);
};

const resetIdleTimer = () => {
  lastInteractionTime = Date.now();
};

const checkIdleTime = () => {
  const now = Date.now();
  const idleDuration = (now - lastInteractionTime) / 1000;
  
  if (idleDuration >= 300) {
    trackEvent('user_idle', { duration: idleDuration });
  }
};

// Track page views
export const trackPageView = (additionalData: Record<string, any> = {}) => {
  const pageData = {
    eventType: 'pageview',
    page: currentPage,
    referrer: document.referrer,
    timestamp: new Date().toISOString(),
    sessionId,
    ...additionalData
  };
  
  sendToBackend('/track', pageData);
};

// Track custom events
export const trackEvent = (eventType: string, eventData: Record<string, any> = {}) => {
  const event: TrackingEvent = {
    eventType,
    ...eventData,
    timestamp: new Date().toISOString(),
    sessionId,
    currentPage,
  };
  
  sendToBackend('/track', event);
};

// Track policy interactions
export const trackPolicyInteraction = (
  customerId: string | null,
  policyId: string,
  interactionType: 'view' | 'click' | 'search' | 'compare' | 'cart_add' | 'cart_abandon' | 'purchase',
  duration?: number,
  additionalData: Record<string, any> = {}
) => {
  trackEvent('policy_interaction', {
    customerId,
    policyId,
    interactionType,
    duration,
    ...additionalData
  });
};

// Track element visibility
export const trackElementView = (
  elementId: string,
  customerId: string | null,
  policyId: string,
  threshold = 0.5
) => {
  const element = document.getElementById(elementId);
  if (!element) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Start tracking view time
        if (!viewTimers[elementId]) {
          viewTimers[elementId] = {
            startTime: Date.now(),
            timer: setInterval(() => {
              const duration = (Date.now() - viewTimers[elementId].startTime) / 1000;
              trackPolicyInteraction(
                customerId,
                policyId,
                'view',
                Math.round(duration)
              );
            }, 5000) // Report every 5 seconds
          };
          trackPolicyInteraction(customerId, policyId, 'view', 0);
        }
      } else {
        // Stop tracking view time
        if (viewTimers[elementId]) {
          clearInterval(viewTimers[elementId].timer);
          const duration = (Date.now() - viewTimers[elementId].startTime) / 1000;
          trackPolicyInteraction(
            customerId,
            policyId,
            'view',
            Math.round(duration)
          );
          delete viewTimers[elementId];
        }
      }
    });
  }, { threshold });

  observer.observe(element);

  return () => {
    if (viewTimers[elementId]) {
      clearInterval(viewTimers[elementId].timer);
      delete viewTimers[elementId];
    }
    observer.disconnect();
  };
};

// Send data to backend
const sendToBackend = async (endpoint: string, data: any) => {
  if (process.env.NODE_ENV === 'development') {
    console.log('Tracking:', data);
  }

  try {
    await fetch(`http://localhost:5000${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      keepalive: true // Ensure request completes even if page unloads
    });
  } catch (error) {
    console.error('Tracking error:', error);
  }
};

// Clean up tracking
const cleanupTracking = () => {
  clearInterval(idleTimer);
  
  // Report remaining view times
  Object.keys(viewTimers).forEach(elementId => {
    const { startTime, timer } = viewTimers[elementId];
    clearInterval(timer);
    const duration = (Date.now() - startTime) / 1000;
    // You might want to send this final duration if needed
  });
  
  trackEvent('session_end');
};

// Initialize tracking when imported
if (typeof window !== 'undefined') {
  initTracking();
}