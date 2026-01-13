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
  { id: 'pi', name: 'Raspberry Pi (Edge)', url: 'http://voice-pi.local:8010' },
  // { id: 'minipc', name: 'Mini PC', url: 'http://minipc.local:8010' },
];

function App() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [envData, setEnvData] = useState<Record<string, Record<string, string>>>({});
  const [expandedNode, setExpandedNode] = useState<string | null>(null);

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

  const fetchEnv = async (nodeUrl: string, nodeId: string) => {
    try {
      const resp = await fetch(`${nodeUrl}/env`);
      if (resp.ok) {
        const data = await resp.json();
        setEnvData(prev => ({ ...prev, [nodeId]: data }));
      }
    } catch (err) {
      console.error('Failed to fetch env', err);
    }
  };

  const saveEnv = async (nodeUrl: string, nodeId: string) => {
    try {
      const resp = await fetch(`${nodeUrl}/env`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(envData[nodeId])
      });
      if (resp.ok) {
        alert('Environment updated successfully');
      }
    } catch (err) {
      alert('Failed to save environment');
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

  const updateEnvKey = (nodeId: string, oldKey: string, newKey: string) => {
    setEnvData(prev => {
      const nodeEnv = { ...prev[nodeId] };
      const val = nodeEnv[oldKey];
      delete nodeEnv[oldKey];
      nodeEnv[newKey] = val;
      return { ...prev, [nodeId]: nodeEnv };
    });
  };

  const updateEnvVal = (nodeId: string, key: string, val: string) => {
    setEnvData(prev => ({
      ...prev,
      [nodeId]: { ...prev[nodeId], [key]: val }
    }));
  };

  const addEnvRow = (nodeId: string) => {
    setEnvData(prev => ({
      ...prev,
      [nodeId]: { ...prev[nodeId], '': '' }
    }));
  };

  const deleteEnvRow = (nodeId: string, key: string) => {
    setEnvData(prev => {
      const nodeEnv = { ...prev[nodeId] };
      delete nodeEnv[key];
      return { ...prev, [nodeId]: nodeEnv };
    });
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

                  <button className="deploy-btn" onClick={() => handleDeploy(node.url)}>
                    🚀 One-Click Deploy
                  </button>

                  <div className="env-editor">
                    <div className="env-header">
                      <h3>Environment Variables</h3>
                      <button onClick={() => {
                        if (expandedNode === node.id) setExpandedNode(null);
                        else {
                          setExpandedNode(node.id);
                          if (!envData[node.id]) fetchEnv(node.url, node.id);
                        }
                      }}>
                        {expandedNode === node.id ? 'Collapse' : 'Manage'}
                      </button>
                    </div>

                    {expandedNode === node.id && (
                      <>
                        <div className="env-grid">
                          {Object.entries(envData[node.id] || {}).map(([key, val]) => (
                            <div key={key} className="env-row">
                              <input
                                placeholder="KEY"
                                value={key}
                                onChange={(e) => updateEnvKey(node.id, key, e.target.value)}
                              />
                              <input
                                placeholder="VALUE"
                                value={val}
                                onChange={(e) => updateEnvVal(node.id, key, e.target.value)}
                              />
                              <button style={{ padding: '0.2rem 0.5rem', background: 'transparent' }} onClick={() => deleteEnvRow(node.id, key)}>✕</button>
                            </div>
                          ))}
                        </div>
                        <div className="env-actions">
                          <button onClick={() => addEnvRow(node.id)}>+ Add Var</button>
                          <button className="primary" onClick={() => saveEnv(node.url, node.id)}>Save Config</button>
                        </div>
                      </>
                    )}
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
