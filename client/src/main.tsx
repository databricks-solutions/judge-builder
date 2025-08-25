import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { OpenAPI } from './fastapi_client'

// Configure API base URL for development
if (import.meta.env.DEV) {
  OpenAPI.BASE = 'http://localhost:9000'
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)