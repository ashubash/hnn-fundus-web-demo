// src/components/LoadingSpinner.jsx
import React from 'react';

const LoadingSpinner = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <div className="spinner"></div>
    <p>Loading Models...</p>
  </div>
);

export default LoadingSpinner;