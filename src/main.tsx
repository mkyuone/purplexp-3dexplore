import { createRoot } from 'react-dom/client';
import './styles.css';

const root = createRoot(document.getElementById('root')!);

if (window.location.pathname.replace(/\/+$/, '') === '/diagnostics') {
  document.title = 'Hand camera diagnostics · Spatial CAD Explorer';
  void import('./hand-diagnostics').then(({ default: HandDiagnostics }) => root.render(<HandDiagnostics />));
} else {
  void import('./App').then(({ App }) => root.render(<App />));
}
