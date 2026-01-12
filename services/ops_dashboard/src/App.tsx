import { useState, useEffect } from 'react';
import './index.css';

interface Service {
  name: string;
  status: string;
  details?: string;
}

interface Node {
  id: string;
  name: string;
  url: string;
  services: Service[];
  isOnline: boolean;
}

const DASHBOARD_NODES = [
  { id: '5090', name: 'RTX 5090 (Compute)', url: 'http://192.168.20.148:8010' },
  // { id: 'pi', name: 'Raspberry Pi', url: 'http://pi.local:8010' },
  // { id: 'minipc', name: 'Mini PC', url: 'http://minipc.local:8010' },
];

function App() {
  const [nodes, setNodes] = useState<Node[]>([]);

  const fetchNodeData = async (nodeMeta: { id: string, name: string, url: string }) => {
    try {
      const resp = await fetch(`${nodeMeta.url}/services`);
      if (!resp.ok) throw new Error('Failed to fetch');
      const services = await resp.json();
      return { ...nodeMeta, services, isOnline: true };
    } catch (err) {
      return { ...nodeMeta, services: [], isOnline: false };
    }
  };

  const refreshAll = async () => {
    const updatedNodes = await Promise.all(DASHBOARD_NODES.map(fetchNodeData));
    setNodes(updatedNodes);
  };

  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  const handleAction = async (nodeUrl: string, serviceName: string, action: string) => {
    try {
      const resp = await fetch(`${nodeUrl}/services/${serviceName}/${action}`, { method: 'POST' });
      if (resp.ok) refreshAll();
    } catch (err) {
      console.error('Action failed', err);
    }
  };

  const handleDeploy = async (nodeUrl: string) => {
    if (!confirm('This will pull latest code and RESTART all services. Proceed?')) return;
    try {
      const resp = await fetch(`${nodeUrl}/deploy`, { method: 'POST' });
      const data = await resp.json();
      alert(`Deploy finished: ${data.restarted_services.join(', ')}`);
      refreshAll();
    } catch (err) {
      alert('Deploy failed');
    }
  };

  return (
    <div className="app">
      <header className="dashboard-header glass">
        <div>
          <h1>Voice Assist Ops</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>Distributed Command Center</p>
        </div>
        <button className="primary" onClick={refreshAll}>Refresh All</button>
      </header>

      <div className="node-grid">
        {nodes.map(node => (
          <div key={node.id} className="node-card glass">
            <div className="node-title">
              <span className="node-name">{node.name}</span>
              <span className={`status-badge ${node.isOnline ? 'status-online' : 'status-offline'}`}>
                {node.isOnline ? 'Online' : 'Offline'}
              </span>
            </div>

            <div className="service-list">
              {node.isOnline ? (
                <>
                  {node.services.length === 0 && <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No managed services found.</p>}
                  {node.services.map(svc => (
                    <div key={svc.name} className="service-item">
                      <div className="service-info">
                        <div className={`service-status-dot ${svc.status}`} />
                        <span>{svc.name}</span>
                      </div>
                      <div className="controls">
                        <button onClick={() => handleAction(node.url, svc.name, 'start')}>▶</button>
                        <button onClick={() => handleAction(node.url, svc.name, 'restart')}>↺</button>
                        <button onClick={() => handleAction(node.url, svc.name, 'stop')}>■</button>
                      </div>
                    </div>
                  ))}
                  <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--glass-border)', paddingTop: '1rem' }}>
                    <button className="deploy-btn" onClick={() => handleDeploy(node.url)}>
                      🚀 One-Click Deploy
                    </button>
                  </div>
                </>
              ) : (
                <p style={{ color: 'var(--status-stopped)', fontSize: '0.8rem' }}>Agent at {node.url} unreachable.</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
