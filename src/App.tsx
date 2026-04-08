import { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Camera, CameraOff, Download, RefreshCw, Shield, Zap, Info, Maximize2, Settings2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---
interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

interface AppState {
  isModelLoading: boolean;
  isCameraActive: boolean;
  error: string | null;
  fps: number;
  detections: Detection[];
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const requestRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const [state, setState] = useState<AppState>({
    isModelLoading: true,
    isCameraActive: false,
    error: null,
    fps: 0,
    detections: [],
  });

  const [targetObject, setTargetObject] = useState<string>('person');
  const [autoSnapshot, setAutoSnapshot] = useState(false);
  const [lastSnapshotTime, setLastSnapshotTime] = useState(0);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');

  // --- Initialization ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2' // Faster for browser
        });
        modelRef.current = model;
        setState(prev => ({ ...prev, isModelLoading: false }));
      } catch (err) {
        console.error('Failed to load model:', err);
        setState(prev => ({ ...prev, error: 'Failed to initialize AI model', isModelLoading: false }));
      }
    };

    loadModel();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // --- Camera Control ---
  const startCamera = async () => {
    try {
      // Stop existing tracks if any
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setState(prev => ({ ...prev, isCameraActive: true, error: null }));
      }
    } catch (err) {
      console.error('Camera access denied:', err);
      setState(prev => ({ ...prev, error: 'Camera access denied. Please check permissions.' }));
    }
  };

  useEffect(() => {
    if (state.isCameraActive) {
      startCamera();
    }
  }, [facingMode]);

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setState(prev => ({ ...prev, isCameraActive: false, detections: [] }));
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
  };

  // --- Detection Loop ---
  const detectFrame = useCallback(async (time: number) => {
    if (!videoRef.current || !modelRef.current || !canvasRef.current || videoRef.current.readyState !== 4) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    // Calculate FPS
    const delta = time - lastTimeRef.current;
    if (delta > 0) {
      const currentFps = Math.round(1000 / delta);
      setState(prev => ({ ...prev, fps: currentFps }));
    }
    lastTimeRef.current = time;

    // Run Detection
    const detections = await modelRef.current.detect(videoRef.current);
    setState(prev => ({ ...prev, detections }));

    // Draw Results
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      detections.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        
        // Draw Bounding Box
        ctx.strokeStyle = prediction.class === targetObject ? '#10b981' : '#3b82f6';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw Label Background
        ctx.fillStyle = prediction.class === targetObject ? '#10b981' : '#3b82f6';
        const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw Label Text
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 14px JetBrains Mono';
        ctx.fillText(label, x + 5, y - 7);

        // Auto Snapshot Logic
        if (autoSnapshot && prediction.class === targetObject && prediction.score > 0.7) {
          const now = Date.now();
          if (now - lastSnapshotTime > 5000) { // Throttle snapshots to every 5s
            takeSnapshot();
            setLastSnapshotTime(now);
          }
        }
      });
    }

    requestRef.current = requestAnimationFrame(detectFrame);
  }, [targetObject, autoSnapshot, lastSnapshotTime]);

  useEffect(() => {
    if (state.isCameraActive && !state.isModelLoading) {
      requestRef.current = requestAnimationFrame(detectFrame);
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [state.isCameraActive, state.isModelLoading, detectFrame]);

  // --- Actions ---
  const takeSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoRef.current.videoWidth;
    tempCanvas.height = videoRef.current.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    if (!ctx) return;

    // Draw video frame
    ctx.drawImage(videoRef.current, 0, 0);
    // Draw detections
    ctx.drawImage(canvasRef.current, 0, 0);

    const dataUrl = tempCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `vision-snapshot-${Date.now()}.png`;
    link.href = dataUrl;
    link.click();
  };

  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Header */}
      <header className="p-6 border-b border-[#2a2b2e] flex items-center justify-between bg-[#0f1012]">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg">
            <Zap className="w-6 h-6 text-emerald-500" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">VISIONARY AI</h1>
            <p className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">Neural Processing Unit v2.4</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className={`status-glow ${state.isModelLoading ? 'status-loading' : modelRef.current ? 'status-active' : 'status-inactive'}`} />
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider">
              Model: {state.isModelLoading ? 'Loading...' : 'Ready'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`status-glow ${state.isCameraActive ? 'status-active' : 'status-inactive'}`} />
            <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider">
              Stream: {state.isCameraActive ? 'Active' : 'Offline'}
            </span>
          </div>
          <div className="px-3 py-1 bg-[#1a1b1e] rounded border border-[#2a2b2e]">
            <span className="text-[10px] font-mono text-emerald-500">{state.fps} FPS</span>
          </div>
        </div>
      </header>

      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 overflow-hidden">
        {/* Left: Video Feed */}
        <section className="lg:col-span-8 flex flex-col gap-4">
          <div className="relative hardware-card flex-1 overflow-hidden group">
            {!state.isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0f1012] z-10">
                <div className="w-20 h-20 rounded-full bg-[#1a1b1e] flex items-center justify-center mb-4 border border-[#2a2b2e]">
                  <CameraOff className="w-8 h-8 text-gray-600" />
                </div>
                <h2 className="text-lg font-medium text-gray-400">Camera Feed Offline</h2>
                <button 
                  onClick={startCamera}
                  disabled={state.isModelLoading}
                  className="mt-6 px-8 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all flex items-center gap-2 shadow-lg shadow-emerald-900/20"
                >
                  <Camera className="w-5 h-5" />
                  Initialize Stream
                </button>
              </div>
            )}

            {state.error && (
              <div className="absolute top-4 left-4 right-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3 z-20">
                <Shield className="w-5 h-5 text-red-500" />
                <p className="text-sm text-red-200">{state.error}</p>
              </div>
            )}

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              onLoadedMetadata={() => {
                if (canvasRef.current && videoRef.current) {
                  canvasRef.current.width = videoRef.current.videoWidth;
                  canvasRef.current.height = videoRef.current.videoHeight;
                }
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            {/* Overlay Controls */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <button 
                onClick={takeSnapshot}
                className="p-3 bg-white/10 backdrop-blur-md border border-white/20 rounded-full hover:bg-white/20 transition-colors"
                title="Take Snapshot"
              >
                <Download className="w-5 h-5" />
              </button>
              <button 
                onClick={stopCamera}
                className="p-3 bg-red-500/20 backdrop-blur-md border border-red-500/40 rounded-full hover:bg-red-500/40 transition-colors"
                title="Stop Stream"
              >
                <CameraOff className="w-5 h-5 text-red-400" />
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between px-2">
            <div className="flex items-center gap-4 text-xs font-mono text-gray-500 uppercase tracking-wider">
              <span className="flex items-center gap-1"><Maximize2 className="w-3 h-3" /> 1280x720</span>
              <span className="flex items-center gap-1"><RefreshCw className="w-3 h-3" /> Real-time Sync</span>
            </div>
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 text-gray-600 cursor-help" />
            </div>
          </div>
        </section>

        {/* Right: Controls & Data */}
        <aside className="lg:col-span-4 flex flex-col gap-6">
          {/* Detection Settings */}
          <div className="hardware-card p-6">
            <div className="flex items-center gap-2 mb-6">
              <Settings2 className="w-4 h-4 text-emerald-500" />
              <h3 className="text-xs font-mono uppercase tracking-widest text-gray-400">System Parameters</h3>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-2">Target Object</label>
                <select 
                  value={targetObject}
                  onChange={(e) => setTargetObject(e.target.value)}
                  className="w-full bg-[#0f1012] border border-[#2a2b2e] rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
                >
                  <option value="person">Person</option>
                  <option value="cell phone">Cell Phone</option>
                  <option value="laptop">Laptop</option>
                  <option value="mouse">Mouse</option>
                  <option value="keyboard">Keyboard</option>
                  <option value="car">Car</option>
                  <option value="bicycle">Bicycle (Bike)</option>
                  <option value="motorcycle">Motorcycle</option>
                  <option value="bus">Bus</option>
                  <option value="truck">Truck</option>
                  <option value="cup">Cup</option>
                  <option value="bottle">Bottle</option>
                  <option value="chair">Chair</option>
                  <option value="dog">Dog</option>
                  <option value="cat">Cat</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-2">Camera Source</label>
                <div className="flex p-1 bg-[#0f1012] border border-[#2a2b2e] rounded-lg">
                  <button
                    onClick={() => setFacingMode('user')}
                    className={`flex-1 py-2 text-[10px] font-mono uppercase tracking-wider rounded-md transition-all ${facingMode === 'user' ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                  >
                    Front Cam
                  </button>
                  <button
                    onClick={() => setFacingMode('environment')}
                    className={`flex-1 py-2 text-[10px] font-mono uppercase tracking-wider rounded-md transition-all ${facingMode === 'environment' ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                  >
                    Back Cam
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-[#0f1012] rounded-lg border border-[#2a2b2e]">
                <div>
                  <h4 className="text-sm font-medium">Auto-Snapshot</h4>
                  <p className="text-[10px] text-gray-500 font-mono">Capture on target detection</p>
                </div>
                <button 
                  onClick={() => setAutoSnapshot(!autoSnapshot)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${autoSnapshot ? 'bg-emerald-600' : 'bg-[#2a2b2e]'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${autoSnapshot ? 'left-7' : 'left-1'}`} />
                </button>
              </div>
            </div>
          </div>

          {/* Live Data Feed */}
          <div className="hardware-card flex-1 flex flex-col min-h-0">
            <div className="p-6 border-b border-[#2a2b2e] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-500" />
                <h3 className="text-xs font-mono uppercase tracking-widest text-gray-400">Live Telemetry</h3>
              </div>
              <span className="text-[10px] font-mono text-gray-600">{state.detections.length} Objects</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
              <AnimatePresence mode="popLayout">
                {state.detections.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-gray-600 py-12">
                    <div className="w-12 h-12 rounded-full border border-dashed border-gray-700 flex items-center justify-center mb-3">
                      <RefreshCw className="w-5 h-5 animate-spin-slow" />
                    </div>
                    <p className="text-[10px] font-mono uppercase tracking-widest">Scanning Environment...</p>
                  </div>
                ) : (
                  state.detections.map((det, i) => (
                    <motion.div
                      key={`${det.class}-${i}`}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className={`p-3 rounded-lg border flex items-center justify-between ${det.class === targetObject ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-[#0f1012] border-[#2a2b2e]'}`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${det.class === targetObject ? 'bg-emerald-500' : 'bg-blue-500'}`} />
                        <span className="text-sm font-medium capitalize">{det.class}</span>
                      </div>
                      <span className="text-xs font-mono text-gray-500">{Math.round(det.score * 100)}%</span>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>
        </aside>
      </main>

      {/* Footer */}
      <footer className="p-4 border-t border-[#2a2b2e] bg-[#0f1012] flex items-center justify-between">
        <div className="flex items-center gap-4">
          <p className="text-[10px] font-mono text-gray-600 uppercase tracking-widest">System Status: Nominal</p>
          <div className="h-3 w-[1px] bg-[#2a2b2e]" />
          <p className="text-[10px] font-mono text-gray-600 uppercase tracking-widest">Engine: TensorFlow.js Core</p>
        </div>
        <div className="flex items-center gap-4">
          <a href="#" className="text-[10px] font-mono text-gray-500 hover:text-white transition-colors uppercase tracking-widest">Documentation</a>
          <a href="#" className="text-[10px] font-mono text-gray-500 hover:text-white transition-colors uppercase tracking-widest">API Access</a>
        </div>
      </footer>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #2a2b2e;
          border-radius: 10px;
        }
        .animate-spin-slow {
          animation: spin 3s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
