import React, { useState } from 'react';
import { WebSocketProvider, useWebSocket } from './contexts/WebSocketContext';
import { Header } from './components/Header';
import { Toolbox } from './components/Toolbox';
import { RecipePanel } from './components/panels/RecipePanel';
import { ConversationPanel } from './components/panels/ConversationPanel';
import { TimersPanel } from './components/panels/TimersPanel';
import { TimerModal } from './components/modals/TimerModal';
import './App.css';

const Layout: React.FC = () => {
  const { isConnected, activeCard } = useWebSocket();
  const [isTimerModalOpen, setIsTimerModalOpen] = useState(false);

  // Dynamic class for tool focus if needed
  const toolFocusClass = activeCard ? 'tool-focus' : '';

  return (
    <div className={`app-container ${toolFocusClass}`}>
      <Header />

      <main className="main-content">
        <Toolbox onOpenTimerModal={() => setIsTimerModalOpen(true)} />

        <RecipePanel />

        <div className="side-panel">
          <ConversationPanel />
          <TimersPanel />
        </div>
      </main>

      <TimerModal isOpen={isTimerModalOpen} onClose={() => setIsTimerModalOpen(false)} />

      {!isConnected && (
        <div className="connection-status">
          Disconnected - Reconnecting...
        </div>
      )}
    </div>
  );
};

const App: React.FC = () => {
  return (
    <WebSocketProvider>
      <Layout />
    </WebSocketProvider>
  );
};

export default App;
