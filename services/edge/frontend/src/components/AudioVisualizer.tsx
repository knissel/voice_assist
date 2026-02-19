import React from 'react';

type VisualizerState = 'idle' | 'listening' | 'recording' | 'processing' | 'thinking' | 'speaking' | 'executing' | 'thinking_local';

interface AudioVisualizerProps {
    state: VisualizerState;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ state }) => {
    // Map backend status to visualizer state
    // backend: "listening", "recording", "processing", "thinking", "speaking", "executing"

    const getVisual = () => {
        switch (state) {
            case 'listening':
                return (
                    <div className="orb-container">
                        <div className="orb-ring ring-1"></div>
                        <div className="orb-ring ring-2"></div>
                        <div className="orb-core"></div>
                    </div>
                );
            case 'recording':
                return (
                    <div className="wave-container">
                        {[...Array(5)].map((_, i) => (
                            <div key={i} className="mic-bar" style={{ animationDelay: `${i * 0.1}s` }}></div>
                        ))}
                    </div>
                );
            case 'processing':
            case 'thinking':
            case 'executing':
                return (
                    <div className="loader-container">
                        <div className="scanner"></div>
                        <div className="status-text">{state.toUpperCase()}</div>
                    </div>
                );
            case 'thinking_local':
                return (
                    <div className="loader-container">
                        <div className="scanner local"></div>
                        <div className="status-text" style={{ color: 'var(--accent-secondary)' }}>LOCAL BRAIN</div>
                        <style>{`
                       .scanner.local::after { background: var(--accent-secondary); }
                     `}</style>
                    </div>
                );
            case 'speaking':
                return (
                    <div className="equalizer">
                        {[...Array(10)].map((_, i) => (
                            <div key={i} className="eq-bar" style={{
                                height: '20%',
                                animationDuration: `${0.4 + Math.random() * 0.5}s`
                            }}></div>
                        ))}
                    </div>
                );
            default: // idle
                return <div className="orb-core idle"></div>;
        }
    };

    return (
        <div className="visualizer-wrapper">
            {getVisual()}
            <style>{`
        .visualizer-wrapper {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        /* Listening Orb */
        .orb-container { position: relative; width: 100px; height: 100px; display: flex; justify-content: center; align-items: center; }
        .orb-core { 
          width: 40px; height: 40px; border-radius: 50%; background: var(--accent-primary); 
          box-shadow: 0 0 20px var(--accent-primary);
        }
        .orb-core.idle { background: var(--text-muted); box-shadow: none; opacity: 0.3; }
        
        .orb-ring {
          position: absolute; border: 2px solid var(--accent-primary); border-radius: 50%; opacity: 0;
          animation: ripple 2s infinite linear;
        }
        .ring-1 { width: 40px; height: 40px; animation-delay: 0s; }
        .ring-2 { width: 40px; height: 40px; animation-delay: 1s; }

        @keyframes ripple {
          0% { transform: scale(1); opacity: 0.8; }
          100% { transform: scale(3); opacity: 0; }
        }

        /* Mic Bars */
        .wave-container { display: flex; gap: 6px; align-items: center; height: 60px; }
        .mic-bar {
          width: 8px; height: 10px; background: var(--accent-warn); border-radius: 4px;
          animation: micWave 0.5s infinite ease-in-out alternate;
        }
        @keyframes micWave {
           0% { height: 10px; opacity: 0.5; }
           100% { height: 50px; opacity: 1; }
        }

        /* Processing/Thinking */
        .loader-container { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
        .scanner {
          width: 120px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; position: relative; overflow: hidden;
        }
        .scanner::after {
          content: ''; position: absolute; inset: 0; background: var(--accent-secondary);
          width: 40%; animation: scan 1s infinite linear;
        }
        .status-text { font-size: 0.8rem; letter-spacing: 2px; color: var(--accent-secondary); animation: blink 1s infinite; }
        
        @keyframes scan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(250%); }
        }
        @keyframes blink { 50% { opacity: 0.5; } }

        /* Speaking Equalizer */
        .equalizer { display: flex; gap: 4px; align-items: center; height: 80px; }
        .eq-bar {
          width: 10px; background: linear-gradient(to top, var(--accent-primary), var(--accent-secondary));
          border-radius: 4px;
          animation: eqBounce 1s infinite ease-in-out;
        }
        @keyframes eqBounce {
          0%, 100% { height: 20%; }
          50% { height: 90%; }
        }
      `}</style>
        </div>
    );
};

export default AudioVisualizer;
