export interface Policy {
  id: string;
  name: string;
  premium: string;
  coverage: string;
  duration: string;
  description: string;
  tags: string[];
}

export const allPolicies: Policy[] = [
  {
    id: 'p1',
    name: 'SBI Life - eShield Next',
    premium: '₹3,600/year',
    coverage: '₹50,00,000',
    duration: '30 years',
    description: 'A pure term plan offering financial protection with multiple plan options and additional rider benefits.',
    tags: ['term', 'protection', 'rider', 'financial security']
  },
  {
    id: 'p2',
    name: 'SBI Life - Smart Scholar Plus',
    premium: '₹50,000/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A unit-linked insurance plan designed to secure your child\'s future with market-linked returns and insurance coverage.',
    tags: ['child plan', 'ULIP', 'education', 'future planning']
  },
  {
    id: 'p3',
    name: 'SBI Life - Retire Smart Plus',
    premium: '₹30,000/year',
    coverage: '₹20,00,000',
    duration: '25 years',
    description: 'A unit-linked pension plan that helps build a retirement corpus through market-linked returns with flexibility and loyalty additions.',
    tags: ['retirement', 'pension', 'ULIP', 'corpus building']
  },
  {
    id: 'p4',
    name: 'SBI Life - eWealth Plus',
    premium: '₹36,000/year',
    coverage: '₹15,00,000',
    duration: '20 years',
    description: 'An online unit-linked insurance plan offering dual benefits of market-linked returns and life cover with easy online access.',
    tags: ['ULIP', 'online plan', 'investment', 'life cover']
  },
  {
    id: 'p5',
    name: 'SBI Life - Smart Elite Plus',
    premium: '₹2,50,000/year',
    coverage: '₹25,00,000',
    duration: '15 years',
    description: 'A unit-linked insurance plan providing life cover and wealth maximization opportunities with a wide range of fund options.',
    tags: ['ULIP', 'high net-worth', 'investment', 'life cover']
  },
  {
    id: 'p6',
    name: 'SBI Life - Smart Platina Supreme',
    premium: '₹50,000/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A non-linked, non-participating savings plan offering regular guaranteed income during the payout period with flexibility to suit life goals.',
    tags: ['savings', 'guaranteed income', 'long term', 'financial planning']
  },
  {
    id: 'p7',
    name: 'SBI Life - Smart Platina Assure',
    premium: '₹50,000/year',
    coverage: '₹5,00,000',
    duration: '15 years',
    description: 'A non-linked, non-participating life insurance savings product assuring guaranteed returns with limited premium payment terms.',
    tags: ['savings', 'guaranteed returns', 'limited premium', 'life insurance']
  },
  {
    id: 'p8',
    name: 'SBI Life - New Smart Samriddhi',
    premium: '₹30,000/year',
    coverage: '₹7,00,000',
    duration: '12 years',
    description: 'A non-participating savings plan providing life cover and guaranteed additions with limited premium payment periods.',
    tags: ['savings', 'guaranteed additions', 'life cover', 'limited premium']
  },
  {
    id: 'p9',
    name: 'SBI Life - Smart Lifetime Saver',
    premium: '₹30,000/year',
    coverage: '₹15,00,000',
    duration: '100 years',
    description: 'A savings plan offering life insurance cover till the age of 100 years with annual survival income and optional rider benefits.',
    tags: ['savings', 'lifetime cover', 'survival income', 'optional rider']
  },
  {
    id: 'p10',
    name: 'SBI Life - Smart Bachat Plus',
    premium: '₹5,100/year',
    coverage: '₹5,00,000',
    duration: '20 years',
    description: 'A non-linked, participating life insurance savings product with limited premium payment terms and two plan options.',
    tags: ['endowment', 'savings', 'limited premium', 'participating plan']
  },
  {
    id: 'p11',
    name: 'SBI Life - Smart Money Back Gold',
    premium: '₹9,500/year',
    coverage: '₹5,00,000',
    duration: '20 years',
    description: 'A traditional money-back plan offering financial liquidity through periodic payouts at specific intervals with life cover.',
    tags: ['money back', 'periodic payouts', 'life cover', 'traditional plan']
  },
  {
    id: 'p12',
    name: 'SBI Life - Smart Money Planner',
    premium: '₹15,000/year',
    coverage: '₹10,00,000',
    duration: '25 years',
    description: 'A non-linked, participating money-back plan ensuring regular income during the payment period and life coverage throughout the policy term.',
    tags: ['money back', 'regular income', 'life cover', 'participating plan']
  },
  {
    id: 'p13',
    name: 'SBI Life - Smart Income Protect',
    premium: '₹10,000/year',
    coverage: '₹7,00,000',
    duration: '20 years',
    description: 'A non-linked, participating insurance plan providing regular income and life cover with guaranteed annual payouts.',
    tags: ['money back', 'regular income', 'life cover', 'guaranteed payouts']
  },
  {
    id: 'p14',
    name: 'SBI Life - Smart Annuity Plus',
    premium: '₹50,000/year',
    coverage: '₹5,00,000',
    duration: '100 years',
    description: 'An annuity plan offering both immediate and deferred annuity options with joint life coverage and guaranteed lifelong regular income.',
    tags: ['annuity', 'retirement', 'regular income', 'joint life']
  },
  {
    id: 'p15',
    name: 'SBI Life - Saral Pension',
    premium: '₹1,00,000/year',
    coverage: '₹10,00,000',
    duration: '100 years',
    description: 'A standard immediate annuity plan providing a regular income stream post-retirement with simple terms and conditions.',
    tags: ['pension', 'immediate annuity', 'regular income', 'retirement']
  },
  {
    id: 'p16',
    name: 'SBI Life - Smart Champ Insurance',
    premium: '₹6,000/year',
    coverage: '₹5,00,000',
    duration: '20 years',
    description: 'A participating child insurance plan ensuring financial support for your child\'s education with benefits like premium waiver and assured benefits.',
    tags: ['child plan', 'education', 'premium waiver', 'assured benefits']
  },
  {
    id: 'p17',
    name: 'SBI Life - Smart Wealth Builder',
    premium: '₹50,000/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A unit-linked insurance plan offering multiple fund options to enhance investment opportunities with life insurance coverage.',
    tags: ['ULIP', 'investment', 'life cover', 'fund options']
  },
  {
    id: 'p18',
    name: 'SBI Life - Smart Wealth Assure',
    premium: '₹1,00,000/year',
    coverage: '₹15,00,000',
    duration: '10 years',
    description: 'A unit-linked insurance plan requiring a one-time premium payment, providing benefits throughout the policy tenure with a choice of fund options.',
    tags: ['ULIP', 'single premium', 'investment', 'life cover']
  },
  {
    id: 'p20',
    name: 'SBI Life - Smart Swadhan Plus',
    premium: '₹2,500/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A non-linked, non-participating term insurance plan that offers return of premium on maturity, providing financial protection and savings.',
    tags: ['term', 'return of premium', 'financial protection', 'savings']
  },
  {
    id: 'p21',
    name: 'SBI Life - Smart Shield',
    premium: '₹3,000/year',
    coverage: '₹50,00,000',
    duration: '25 years',
    description: 'A traditional term insurance plan offering financial protection to your family at an affordable cost with multiple plan options.',
    tags: ['term', 'pure protection', 'affordable', 'multiple options']
  },
  {
    id: 'p22',
    name: 'SBI Life - Saral Jeevan Bima',
    premium: '₹1,500/year',
    coverage: '₹25,00,000',
    duration: '15 years',
    description: 'A standard term insurance plan with simple terms and conditions, providing financial protection to your family.',
    tags: ['term', 'standard plan', 'simple terms', 'financial protection']
  },
  {
    id: 'p23',
    name: 'SBI Life - Smart Swadhan Neo',
    premium: '₹5,000/year',
    coverage: '₹20,00,000',
    duration: '20 years',
    description: 'A non-linked, non-participating term insurance plan that offers return of premium on maturity, ensuring financial protection.',
    tags: ['term', 'return of premium', 'financial protection', 'non-participating']
  },
  {
    id: 'p24',
    name: 'SBI Life - Sampoorna Cancer Suraksha',
    premium: '₹4,000/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A non-linked, non-participating health insurance plan providing financial protection against cancer with multiple sum assured options.',
    tags: ['health', 'cancer coverage', 'financial protection', 'non-participating']
  },
  {
    id: 'p25',
    name: 'SBI Life - Poorna Suraksha',
    premium: '₹6,000/year',
    coverage: '₹15,00,000',
    duration: '20 years',
    description: 'A non-linked, non-participating health insurance plan offering comprehensive coverage against major critical illnesses with life cover.',
    tags: ['health', 'critical illness', 'comprehensive coverage', 'life cover']
  },
  {
    id: 'p26',
    name: 'SBI Life - Smart Health Insurance',
    premium: '₹3,500/year',
    coverage: '₹5,00,000',
    duration: '15 years',
    description: 'A non-linked, non-participating health insurance plan providing coverage against hospitalization expenses with optional riders.',
    tags: ['health', 'hospitalization', 'basic coverage', 'optional riders']
  },
  {
    id: 'p27',
    name: 'SBI Life - Smart Money Planner',
    premium: '₹12,000/year',
    coverage: '₹10,00,000',
    duration: '20 years',
    description: 'A non-linked, participating money-back plan ensuring regular income during the payment period and life coverage throughout the policy term.',
    tags: ['money back', 'regular income', 'life cover', 'participating plan']
  },
  {
    id: 'p28',
    name: 'SBI Life - Smart Income Protect',
    premium: '₹10,000/year',
    coverage: '₹7,00,000',
    duration: '20 years',
    description: 'A non-linked, participating insurance plan providing regular income and life cover with guaranteed annual payouts.',
    tags: ['money back', 'income protection', 'life cover', 'guaranteed payouts']
  },
  {
    id: 'p29',
    name: 'SBI Life - Smart Money Back Gold',
    premium: '₹9,500/year',
    coverage: '₹5,00,000',
    duration: '20 years',
    description: 'A traditional money-back plan offering financial liquidity through periodic payouts at specific intervals with life cover.',
    tags: ['money back', 'periodic payouts', 'life cover', 'traditional plan']
  },
  {
    id: 'p30',
    name: 'SBI Life - Smart Humsafar',
    premium: '₹15,000/year',
    coverage: '₹10,00,000',
    duration: '25 years',
    description: 'A joint life insurance plan providing coverage for both spouses. Offers benefits like premium waiver in case of death of one spouse and maturity benefits.',
    tags: ['endowment', 'joint life cover', 'premium waiver', 'maturity benefits']
  },
  {
    id: 'p31',
    name: 'SBI Life - Smart Bachat',
    premium: '₹5,100/year',
    coverage: '₹5,00,000',
    duration: '20 years',
    description: 'A non-linked, participating life insurance savings product with limited premium payment terms and two plan options.',
    tags: ['endowment', 'savings', 'limited premium', 'participating plan']
  },
  {
    id: 'p32',
    name: 'SBI Life - Smart Guaranteed Savings Plan',
    premium: '₹50,000/year',
    coverage: '₹10,00,000',
    duration: '15 years',
    description: 'A non-linked, non-participating endowment plan offering guaranteed additions and life cover. Provides flexibility with premium payment terms and policy durations.',
    tags: ['endowment', 'guaranteed returns', 'life cover', 'flexible terms']
  },
  {
    id: 'p33',
    name: 'SBI Life - Smart Future Choices',
    premium: '₹30,000/year',
    coverage: '₹7,00,000',
    duration: '20 years',
    description: 'A non-linked, participating endowment plan offering flexibility in premium payment and policy term with bonuses.',
    tags: ['endowment', 'flexible savings', 'premium payment', 'bonuses']
  },
  {
    id: 'p34',
    name: 'SBI Life - Smart Women Advantage',
    premium: '₹25,000/year',
    coverage: '₹5,00,000',
    duration: '15 years',
    description: 'A non-linked, participating endowment plan designed specifically for women, offering life cover and savings with additional benefits.',
    tags: ['endowment', 'women', 'life cover', 'savings', 'additional benefits']
  },
{
  id: 'p35',
  name: 'SBI Life - Smart Power Insurance',
  premium: '₹30,000/year',
  coverage: '₹20,00,000',
  duration: '15 years',
  description: 'A unit-linked insurance plan designed for individuals seeking high returns with life cover. Offers two fund options: Trigger Fund and Smart Fund, catering to different risk appetites.',
  tags: ['ULIP', 'high returns', 'life cover', 'investment', 'fund options']
},
{
  id: 'p36',
  name: 'SBI Life - Smart Wealth Assure',
  premium: '₹1,00,000/year',
  coverage: '₹15,00,000',
  duration: '10 years',
  description: 'A unit-linked insurance plan that requires only a one-time premium payment, providing benefits throughout the policy tenure. Offers a choice of fund options for market-linked returns.',
  tags: ['ULIP', 'single premium', 'investment', 'life cover', 'fund options']
},
{
  id: 'p37',
  name: 'SBI Life - Smart Wealth Builder',
  premium: '₹50,000/year',
  coverage: '₹10,00,000',
  duration: '20 years',
  description: 'A unit-linked insurance plan offering 11 different fund options to enhance investment opportunities. Provides life insurance coverage throughout the policy tenure.',
  tags: ['ULIP', 'investment', 'life cover', 'fund options']
},
{
  id: 'p38',
  name: 'SBI Life - Smart Swadhan Supreme',
  premium: '₹6,000/year',
  coverage: '₹10,00,000',
  duration: '20 years',
  description: 'A term insurance plan that provides life cover at affordable premiums and returns the total premium paid at the end of the policy term upon survival.',
  tags: ['term', 'return of premium', 'affordable', 'life cover']
},
{
  id: 'p39',
  name: 'SBI Life - Saral Swadhan Supreme',
  premium: '₹7,750/year',
  coverage: '₹5,00,000',
  duration: '10 years',
  description: 'A non-linked, non-participating term assurance plan with return of premium. Provides life cover with guaranteed maturity benefit equal to total premiums paid.',
  tags: ['term', 'return of premium', 'life cover', 'simple plan']
},
{
  id: 'p40',
  name: 'SBI Life - Smart Shield Premier',
  premium: '₹9,500/year',
  coverage: '₹50,00,000',
  duration: '25 years',
  description: 'An exclusive term plan offering higher coverage with flexible premium payment options and choice of benefit options to suit protection needs.',
  tags: ['term', 'high coverage', 'flexible premium', 'benefit options']
},
{
  id: 'p41',
  name: 'SBI Life - eShield Insta',
  premium: '₹2,259/year',
  coverage: '₹25,00,000',
  duration: '20 years',
  description: 'A digital term plan available at your fingertips, providing financial protection with easy enrolment through instant and swift processing.',
  tags: ['term', 'digital plan', 'instant', 'financial protection']
},
{
  id: 'p42',
  name: 'SBI Life - Smart Swadhan Neo',
  premium: '₹5,000/year',
  coverage: '₹20,00,000',
  duration: '20 years',
  description: 'A term insurance plan that offers return of premium on maturity, ensuring financial protection at a reasonable cost.',
  tags: ['term', 'return of premium', 'financial protection', 'affordable']
},
{
  id: 'p43',
  name: 'SBI Life - Sampoorna Cancer Suraksha',
  premium: '₹4,000/year',
  coverage: '₹10,00,000',
  duration: '20 years',
  description: 'A health insurance plan providing financial protection against cancer with multiple sum assured options and easy payout.',
  tags: ['health', 'cancer coverage', 'financial protection', 'easy payout']
},
{
  id: 'p44',
  name: 'SBI Life - Poorna Suraksha',
  premium: '₹6,000/year',
  coverage: '₹15,00,000',
  duration: '20 years',
  description: 'A health insurance plan offering comprehensive coverage against major critical illnesses with life cover and monthly income benefit option.',
  tags: ['health', 'critical illness', 'comprehensive coverage', 'life cover']
},
{
  id: 'p45',
  name: 'SBI Life - Smart Health Insurance',
  premium: '₹3,500/year',
  coverage: '₹5,00,000',
  duration: '15 years',
  description: 'A health insurance plan providing coverage against hospitalization expenses with optional riders for enhanced protection.',
  tags: ['health', 'hospitalization', 'basic coverage', 'optional riders']
},
{
  id: 'p46',
  name: 'SBI Life - Smart Money Planner',
  premium: '₹12,000/year',
  coverage: '₹10,00,000',
  duration: '20 years',
  description: 'A money-back plan ensuring regular income during the payment period and life coverage throughout the policy term.',
  tags: ['money back', 'regular income', 'life cover', 'participating plan']
},
{
  id: 'p47',
  name: 'SBI Life - Smart Income Protect',
  premium: '₹10,000/year',
  coverage: '₹7,00,000',
  duration: '20 years',
  description: 'An insurance plan providing regular income and life cover with guaranteed annual payouts.',
  tags: ['money back', 'income protection', 'life cover', 'guaranteed payouts']
},
{
  id: 'p48',
  name: 'SBI Life - Smart Money Back Gold',
  premium: '₹9,500/year',
  coverage: '₹5,00,000',
  duration: '20 years',
  description: 'A traditional money-back plan offering financial liquidity through periodic payouts at specific intervals with life cover.',
  tags: ['money back', 'periodic payouts', 'life cover', 'traditional plan']
},
{
  id: 'p49',
  name: 'SBI Life - Smart Humsafar',
  premium: '₹15,000/year',
  coverage: '₹10,00,000',
  duration: '25 years',
  description: 'A joint life insurance plan providing coverage for both spouses with benefits like premium waiver in case of death of one spouse and maturity benefits.',
  tags: ['endowment', 'joint life cover', 'premium waiver', 'maturity benefits']
},
{
  id: 'p50',
  name: 'SBI Life - Smart Bachat',
  premium: '₹5,100/year',
  coverage: '₹5,00,000',
  duration: '20 years',
  description: 'A life insurance savings product with limited premium payment terms and two plan options, offering flexibility and savings.',
  tags: ['endowment', 'savings', 'limited premium', 'participating plan']
}
];

// Grouping policies by type
export const policies: Record<string, Policy[]> = {
'term': allPolicies.filter(p => p.tags.includes('term')),
'ulip': allPolicies.filter(p => p.tags.includes('ULIP')),
'health': allPolicies.filter(p => p.tags.includes('health')),
'savings': allPolicies.filter(p => p.tags.includes('savings')),
'child': allPolicies.filter(p => p.tags.includes('child plan')),
'retirement': allPolicies.filter(p => p.tags.includes('retirement')),
'money_back': allPolicies.filter(p => p.tags.includes('money back')),
'endowment': allPolicies.filter(p => p.tags.includes('endowment'))
};