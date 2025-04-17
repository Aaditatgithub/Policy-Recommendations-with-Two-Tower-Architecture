  import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
  import Welcome from './pages/Welcome';
  import CustomerDropdown from './components/CustomerDropdown';
  import CustomerProfile from './pages/CustomerProfile';
  import Dashboard from './pages/Dashboard';
  import PolicyList from './components/PolicyList';
  import './styles/theme.css';
  import HeroGeometric from './pages/HeroGeometric';

  function App() {
    return (
      <Router>
        <Routes>
          <Route path="/welcome" element={<Welcome />} />
          <Route path="/" element={<HeroGeometric />} />
          <Route path="/customer" element={<CustomerDropdown />} />
          <Route path="/customer/:id" element={<CustomerProfile />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/policies" element={<PolicyList />} />
        </Routes>
      </Router>
    );
  }

  export default App;