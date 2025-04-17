/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'sbi-purple': '#46166B',
        'sbi-pink': '#E31E54',
        'sbi-blue': '#007DC5',
        'sbi-light-purple': '#F5F3F7',
      },
    },
  },
  plugins: [],
};