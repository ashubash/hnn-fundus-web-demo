// src/App.jsx
import React, { Suspense } from 'react';
import FundusDemo from './FundusDemo';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  return (
    <div>
      {/* Remove the h1 from here—it's already in FundusDemo */}
      <Suspense fallback={<LoadingSpinner />}>
        <FundusDemo />
      </Suspense>
    </div>
  );
}

export default App;