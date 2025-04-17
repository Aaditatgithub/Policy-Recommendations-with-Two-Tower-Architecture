export interface Customer {
  id: string;
  name: string;
  age: number;
  income: string;
  familySize: number;
  maritalStatus: 'Married' | 'Single';
  occupation: string;
  city: string;
  riskProfile: 'Conservative' | 'Moderate' | 'Aggressive';
  existingPolicies: number;
}

export const customers: Customer[] = [
  {
    id: '1',
    name: 'Aaditya Patil',
    age: 35,
    income: '15-20L',
    familySize: 4,
    maritalStatus: 'Married',
    occupation: 'Software Engineer',
    city: 'Bangalore',
    riskProfile: 'Moderate',
    existingPolicies: 2
  },
  {
    id: '2',
    name: 'Omkar Lolage',
    age: 28,
    income: '10-15L',
    familySize: 2,
    maritalStatus: 'Single',
    occupation: 'Marketing Manager',
    city: 'Mumbai',
    riskProfile: 'Aggressive',
    existingPolicies: 1
  },
  {
    id: '3',
    name: 'Swaraj Phand',
    age: 42,
    income: '25-30L',
    familySize: 5,
    maritalStatus: 'Married',
    occupation: 'Business Owner',
    city: 'Delhi',
    riskProfile: 'Conservative',
    existingPolicies: 3
  },
  {
    id: '4',
    name: 'Aditya Deore',
    age: 31,
    income: '20-25L',
    familySize: 3,
    maritalStatus: 'Married',
    occupation: 'Doctor',
    city: 'Hyderabad',
    riskProfile: 'Moderate',
    existingPolicies: 2
  },
  {
    id: '5',
    name: 'Aditya Lad',
    age: 45,
    income: '30-40L',
    familySize: 6,
    maritalStatus: 'Married',
    occupation: 'Entrepreneur',
    city: 'Chandigarh',
    riskProfile: 'Aggressive',
    existingPolicies: 4
  },
  {
    id: '6',
    name: 'Anshu Parihar',
    age: 29,
    income: '12-18L',
    familySize: 2,
    maritalStatus: 'Single',
    occupation: 'Data Scientist',
    city: 'Pune',
    riskProfile: 'Moderate',
    existingPolicies: 1
  },
  {
    id: '7',
    name: 'Suresh Nair',
    age: 50,
    income: '20-25L',
    familySize: 5,
    maritalStatus: 'Married',
    occupation: 'Government Employee',
    city: 'Kochi',
    riskProfile: 'Conservative',
    existingPolicies: 3
  },
  {
    id: '8',
    name: 'Meera Desai',
    age: 34,
    income: '15-22L',
    familySize: 3,
    maritalStatus: 'Married',
    occupation: 'Chartered Accountant',
    city: 'Ahmedabad',
    riskProfile: 'Moderate',
    existingPolicies: 2
  },
  {
    id: '9',
    name: 'Vikram Bhat',
    age: 40,
    income: '22-28L',
    familySize: 4,
    maritalStatus: 'Married',
    occupation: 'Investment Banker',
    city: 'Chennai',
    riskProfile: 'Aggressive',
    existingPolicies: 5
  },
  {
    id: '10',
    name: 'Pooja Verma',
    age: 27,
    income: '8-12L',
    familySize: 1,
    maritalStatus: 'Single',
    occupation: 'Content Writer',
    city: 'Jaipur',
    riskProfile: 'Conservative',
    existingPolicies: 1
  }
];