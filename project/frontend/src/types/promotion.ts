export interface Policy {
  policy_id: string;
  policy_name: string;
  policy_type: string;
  description: string;
  premium_amount?: number | string;
  sum_assured?: number | string;
  policy_duration_years?: number | string;
  keywords?: string;
  risk_category?: string;
  customer_target_group?: string;
  'premium_amount (INR)'?: number | string;
  'sum_assured (INR)'?: number | string;
  [key: string]: any;
  score?: number;
  selected_for_promotion: boolean;
}


export interface FilterState {
  demographic: {
    age: { min: number; max: number; enabled: boolean };
    gender: string[];
    income_bracket: string[];
    employment_status: string[];
    marital_status: string[];
    location_city: string[];
  };
  policy: {
    policy_ownership_count: { min: number; max: number; enabled: boolean };
    last_policy_purchase: { from: string; to: string; enabled: boolean };
    credit_score: { min: number; max: number; enabled: boolean };
    preferred_policy_type: string[];
  };
  interaction: {
    clicked: boolean | null;
    purchased: boolean | null;
    abandoned_cart: boolean | null;
    viewed_duration: { min: number; max: number; enabled: boolean };
    comparison_count: { min: number; max: number; enabled: boolean };
  };
  persona: string[];
}

export interface FilterPreset {
  id: string;
  name: string;
  filters: FilterState;
  selected_policies: Policy[];
  created_at: string;
  target_customer_count: number;
  description?: string;
  is_active?: boolean;
}

export interface PromotionCampaign {
  id: string;
  name: string;
  preset_id: string;
  start_date: string;
  end_date: string;
  budget: number;
  status: 'draft' | 'active' | 'paused' | 'completed';
  performance_metrics?: {
    impressions: number;
    clicks: number;
    conversions: number;
    cost_per_click: number;
    roi: number;
  };
}