import React, { useState, useEffect } from "react";
import { 
  Layers, 
  Network, 
  Database, 
  Cpu, 
  GitBranch, 
  Search, 
  ArrowRight, 
  Clock, 
  Plus, 
  X, 
  Sparkles, 
  BookOpen, 
  CheckCircle2, 
  Bookmark, 
  ChevronRight, 
  ChevronLeft, 
  MonitorPlay, 
  Info, 
  Settings, 
  RotateCcw,
  Library,
  GraduationCap
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// Types for Saved Topics
interface SavedTopic {
  id: string;
  title: string;
  category: "networking" | "database" | "operating-systems" | "algorithms";
  lastUsed: string;
  stepsCount: number;
  iconType: "network" | "database" | "cpu" | "algorithm";
}

// Pre-defined Example Topics
const EXAMPLE_TOPICS = [
  { text: "TCP Three-Way Handshake", category: "networking" },
  { text: "OSI Model Packet Flow", category: "networking" },
  { text: "How Database Commit Works", category: "database" },
  { text: "CPU Round-Robin Scheduling", category: "operating-systems" },
  { text: "Binary Search Algorithm", category: "algorithms" },
  { text: "Dijkstra's Shortest Path", category: "algorithms" }
];

// Initial mock data for Saved/Recent Topics
const MOCK_SAVED_TOPICS: SavedTopic[] = [
  {
    id: "1",
    title: "TCP Three-Way Handshake",
    category: "networking",
    lastUsed: "2 hours ago",
    stepsCount: 4,
    iconType: "network"
  },
  {
    id: "2",
    title: "PostgreSQL B-Tree Index Search",
    category: "database",
    lastUsed: "Yesterday",
    stepsCount: 5,
    iconType: "database"
  },
  {
    id: "3",
    title: "CPU Round-Robin Scheduling",
    category: "operating-systems",
    lastUsed: "Oct 12, 2026",
    stepsCount: 6,
    iconType: "cpu"
  },
  {
    id: "4",
    title: "Dijkstra's Shortest Path Alg",
    category: "algorithms",
    lastUsed: "Oct 5, 2026",
    stepsCount: 6,
    iconType: "algorithm"
  },
  {
    id: "5",
    title: "DNS Query Resolution Flow",
    category: "networking",
    lastUsed: "Sep 28, 2026",
    stepsCount: 5,
    iconType: "network"
  }
];

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface SlideStep { phase: string; description: string; bullets: string[]; }
interface Explainer {
  id: string; topic: string; title: string; category: string;
  summary: string; steps: SlideStep[]; steps_count: number;
  b2_url: string | null; manifest_url: string | null; html_url: string | null; generated_at: string;
}
interface LibraryApiItem {
  id: string; topic: string; title: string; category: string;
  summary: string; steps_count: number; generated_at: string; b2_url: string;
}

export default function App() {
  const [inputValue, setInputValue] = useState("");
  const [isEmptyLibrary, setIsEmptyLibrary] = useState(false);
  const [savedTopics, setSavedTopics] = useState<SavedTopic[]>([]);
  
  // Simulation and interactive states
  const [screenState, setScreenState] = useState<"default" | "loading" | "success">("default");
  const [loadingStep, setLoadingStep] = useState(0);
  const [generatedTopic, setGeneratedTopic] = useState("");
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewStep, setPreviewStep] = useState(0);
  const [generatedExplainer, setGeneratedExplainer] = useState<Explainer | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // States to facilitate a rich simulation demo
  const loadingMessages = [
    { text: "Parsing technical topic semantic structure...", duration: 800 },
    { text: "Synthesizing visual layers & transition timeline...", duration: 1000 },
    { text: "Generating interactive smartboard controller...", duration: 900 },
    { text: "Optimizing typography & layout contrasts for classroom projectors...", duration: 700 }
  ];

  // Real API integration for generation
  const handleGenerate = async (topicText: string) => {
    if (!topicText.trim()) return;
    setGeneratedTopic(topicText);
    setScreenState("loading");
    setLoadingStep(0);
    setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: topicText }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? "Generation failed");
      }
      const data = await res.json();
      const explainer: Explainer = data.explainer;
      setGeneratedExplainer(explainer);
      setScreenState("success");
      setPreviewStep(0);
      setShowPreviewModal(true);
      // Add to library list
      setSavedTopics(prev => [{
        id: explainer.id, title: explainer.title,
        category: explainer.category as SavedTopic["category"],
        lastUsed: "Just now", stepsCount: explainer.steps_count,
        iconType: explainer.category === "database" ? "database"
                : explainer.category === "operating-systems" ? "cpu"
                : explainer.category === "algorithms" ? "algorithm" : "network",
      }, ...prev]);
    } catch (err: unknown) {
      setApiError(err instanceof Error ? err.message : String(err));
      setScreenState("default");
    }
  };

  const handleLoadExplainer = async (id: string, fallbackTitle: string) => {
    setScreenState("loading");
    setApiError(null);
    setGeneratedTopic(fallbackTitle);
    try {
      const res = await fetch(`${API_BASE}/api/explainer/${id}`);
      if (!res.ok) {
        // If not real backend ID, just mock it
        setScreenState("success");
        setPreviewStep(0);
        setShowPreviewModal(true);
        return;
      }
      const data = await res.json();
      setGeneratedExplainer(data.explainer);
      setGeneratedTopic(data.explainer.title);
      setScreenState("success");
      setPreviewStep(0);
      setShowPreviewModal(true);
    } catch (err: unknown) {
      setApiError(err instanceof Error ? err.message : String(err));
      setScreenState("default");
    }
  };

  // Run the loading step animation for visual feedback while fetching
  useEffect(() => {
    if (screenState !== "loading") return;

    if (loadingStep < loadingMessages.length - 1) {
      const timer = setTimeout(() => {
        setLoadingStep(prev => prev + 1);
      }, loadingMessages[loadingStep].duration);
      return () => clearTimeout(timer);
    }
  }, [screenState, loadingStep]);

  // Fetch library from B2 on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/library`)
      .then(r => r.json())
      .then((data: { items: LibraryApiItem[] }) => {
        if (data.items && data.items.length > 0) {
            setSavedTopics(data.items.map(item => ({
            id: item.id, title: item.title,
            category: item.category as SavedTopic["category"],
            lastUsed: new Date(item.generated_at).toLocaleDateString(),
            stepsCount: item.steps_count,
            iconType: item.category === "database" ? "database"
                    : item.category === "operating-systems" ? "cpu"
                    : item.category === "algorithms" ? "algorithm" : "network",
            })));
        } else {
            setIsEmptyLibrary(true);
        }
      })
      .catch(e => console.warn("Library fetch failed:", e));
  }, []);

  // Choose appropriate icon for cards
  const renderCardIcon = (iconType: SavedTopic["iconType"]) => {
    switch (iconType) {
      case "network":
        return <Network className="w-5 h-5 text-indigo-600" />;
      case "database":
        return <Database className="w-5 h-5 text-emerald-600" />;
      case "cpu":
        return <Cpu className="w-5 h-5 text-amber-600" />;
      case "algorithm":
        return <GitBranch className="w-5 h-5 text-purple-600" />;
    }
  };

  // Generate category pill styles
  const getCategoryBadge = (category: SavedTopic["category"]) => {
    switch (category) {
      case "networking":
        return <span className="px-2.5 py-1 text-xs font-medium rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100">Networking</span>;
      case "database":
        return <span className="px-2.5 py-1 text-xs font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">Database</span>;
      case "operating-systems":
        return <span className="px-2.5 py-1 text-xs font-medium rounded-full bg-amber-50 text-amber-700 border border-amber-100">OS</span>;
      case "algorithms":
        return <span className="px-2.5 py-1 text-xs font-medium rounded-full bg-purple-50 text-purple-700 border border-purple-100">Algorithms</span>;
    }
  };

  // Mock presentation slides based on the selected/generated topic
  const getMockPresentationSlides = (topic: string) => {
    return [
      {
        title: "Phase 1: Initialization & Request Setup",
        description: `Starting the execution flow for "${topic}". We initialize system resources, establish handshakes, or prepare data boundaries for processing.`,
        bullets: [
          "State transition to READY triggers event handlers.",
          "Network memory buffers are allocated to prevent overflow.",
          "Logger records structural initiation timestamps."
        ]
      },
      {
        title: "Phase 2: Core State Verification",
        description: `The processing loop engages. The algorithm or packet inspects header values, compares query nodes, or aligns task scheduler parameters.`,
        bullets: [
          "Evaluating condition flags against validation rules.",
          "Routing internal pipeline commands to active worker threads.",
          "Maintaining thread synchronization barriers."
        ]
      },
      {
        title: "Phase 3: Real-time Payload Transport",
        description: `Data transformation and transfer is finalized. Memory blocks are committed, packets are verified via checksum matches, or branch paths are traversed.`,
        bullets: [
          "Checksum validation returns logical true match.",
          "Updating local registries to reflect state modification.",
          "Signaling completion triggers back to queue manager."
        ]
      },
      {
        title: "Phase 4: Flow Handshake & Complete Teardown",
        description: `Process execution safely concludes. Connected streams are formally closed, index registers unlock, and the smartboard renders terminal states.`,
        bullets: [
          "Releasing process locks and clearing tracking metadata.",
          "Broadcasting terminal ACK status code to listener sockets.",
          "Process completed successfully. Visualizer returned 200 OK."
        ]
      }
    ];
  };

  // Use real LLM output if available, fall back to mock for items loaded from library
  const slides = generatedExplainer?.steps.map(s => ({
    title: s.phase,
    description: s.description,
    bullets: s.bullets,
  })) ?? getMockPresentationSlides(generatedTopic || "Selected Topic");

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col antialiased selection:bg-indigo-100">
      


      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 flex flex-col justify-center">
        
        <div className="w-full max-w-4xl mx-auto py-6 sm:py-10">
          
          <AnimatePresence mode="wait">
            
            {/* Screen State: DEFAULT or TYPING */}
            {screenState === "default" && (
              <motion.div
                key="default-screen"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.3 }}
                className="space-y-12"
              >
                {/* 1. Header with tagline */}
                <div className="text-center space-y-3">
                  <div className="inline-flex items-center justify-center p-3.5 bg-indigo-600 text-white rounded-2xl shadow-md shadow-indigo-100 mb-2">
                    <Layers className="w-8 h-8" />
                  </div>
                  <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 font-display">
                    Pulse<span className="text-indigo-600">Board</span>
                  </h1>
                  <p className="text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto font-medium">
                    Turn any complex process or technical flow into an interactive visual explainer for classroom smartboards.
                  </p>
                </div>

                {/* 2. Primary Hero Input & 4. Primary Generate Button */}
                <div className="bg-white rounded-3xl p-6 sm:p-8 shadow-xl shadow-slate-200 border border-slate-100 space-y-6">
                  <div className="space-y-2">
                    <label htmlFor="topic-input" className="block text-sm font-semibold text-slate-700 uppercase tracking-wider font-mono">
                      Target Presentation Topic
                    </label>
                    
                    <div className="relative flex flex-col md:flex-row gap-3">
                      <div className="relative flex-1">
                        <span className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-slate-400">
                          <Search className="w-6 h-6" />
                        </span>
                        <input
                          id="topic-input"
                          type="text"
                          className="w-full pl-12 pr-4 py-4.5 bg-slate-50 border-2 border-slate-200 rounded-2xl text-slate-900 placeholder:text-slate-400 font-medium text-lg sm:text-xl transition-all focus:bg-white focus:border-indigo-600 focus:ring-0 outline-none"
                          placeholder="Try: 'OSI model packet flow' or 'How TCP handshake works'"
                          value={inputValue}
                          onChange={(e) => setInputValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleGenerate(inputValue);
                          }}
                        />
                        {inputValue && (
                          <button
                            id="clear-input-btn"
                            onClick={() => setInputValue("")}
                            className="absolute inset-y-0 right-0 pr-4 flex items-center text-slate-400 hover:text-slate-600 cursor-pointer"
                          >
                            <X className="w-6 h-6" />
                          </button>
                        )}
                      </div>
                      
                      <button
                        id="generate-btn"
                        onClick={() => handleGenerate(inputValue)}
                        disabled={!inputValue.trim()}
                        className={`md:w-auto w-full py-4.5 px-8 font-bold text-lg rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 shadow-lg cursor-pointer min-h-[56px] ${
                          inputValue.trim()
                            ? "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-100 hover:shadow-indigo-200 active:scale-[0.98]"
                            : "bg-slate-100 text-slate-400 border border-slate-200 shadow-none cursor-not-allowed"
                        }`}
                      >
                        <span>Generate Explainer</span>
                        <ArrowRight className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* 3. Tappable "Example Topic" Chips */}
                  <div className="space-y-3 pt-2">
                    <div className="flex items-center gap-2 text-slate-500 text-sm font-semibold uppercase tracking-wide font-mono">
                      <Sparkles className="w-4 h-4 text-indigo-500" />
                      <span>Suggested topics for today's lesson</span>
                    </div>
                    <div className="flex flex-wrap gap-2.5">
                      {EXAMPLE_TOPICS.map((topic, idx) => (
                        <button
                          key={idx}
                          id={`example-chip-${idx}`}
                          onClick={() => setInputValue(topic.text)}
                          className={`px-4 py-2.5 text-sm sm:text-base font-medium rounded-xl transition border text-left cursor-pointer ${
                            inputValue === topic.text
                              ? "bg-indigo-50 text-indigo-700 border-indigo-300 ring-2 ring-indigo-100"
                              : "bg-slate-50 hover:bg-slate-100 text-slate-700 border-slate-200 hover:border-slate-300"
                          }`}
                        >
                          {topic.text}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 5. Recent / Saved Topics Library or Empty Library State */}
                <div className="space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                    <div className="flex items-center gap-2.5">
                      <Library className="w-5.5 h-5.5 text-slate-700" />
                      <h2 className="text-xl sm:text-2xl font-bold text-slate-900 font-display">
                        Your Saved Topics Library
                      </h2>
                    </div>
                    <span className="text-sm font-semibold text-slate-500 bg-slate-100 px-3 py-1 rounded-full">
                      {isEmptyLibrary ? "0" : savedTopics.length} saved explainer{savedTopics.length === 1 ? "" : "s"}
                    </span>
                  </div>

                  {isEmptyLibrary ? (
                    /* Core Element #5: Empty Recent Topics State */
                    <motion.div
                      id="empty-library-view"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="bg-white border border-slate-200 border-dashed rounded-3xl p-10 text-center space-y-4 max-w-xl mx-auto shadow-sm"
                    >
                      <div className="inline-flex p-4 bg-slate-50 text-slate-400 rounded-full">
                        <BookOpen className="w-8 h-8" />
                      </div>
                      <div className="space-y-1.5">
                        <h3 className="text-lg font-bold text-slate-800">Your saved topics will appear here</h3>
                        <p className="text-slate-500 text-sm sm:text-base">
                          Generate topics during the semester to build a reusable library of touch-interactive visual flowcharts.
                        </p>
                      </div>
                      <div className="pt-2">
                        <button 
                          id="empty-load-sample-btn"
                          onClick={() => {
                            setIsEmptyLibrary(false);
                            setSavedTopics(MOCK_SAVED_TOPICS);
                          }}
                          className="px-4 py-2 bg-slate-100 hover:bg-slate-200 active:bg-slate-300 text-slate-700 font-semibold text-xs rounded-xl transition cursor-pointer"
                        >
                          Load Sample Library
                        </button>
                      </div>
                    </motion.div>
                  ) : (
                    /* Populated Recent Topics State (Grid of Saved Cards) */
                    <motion.div 
                      id="populated-library-grid"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="grid grid-cols-1 md:grid-cols-2 gap-4"
                    >
                      {savedTopics.map((item) => (
                        <div
                          key={item.id}
                          id={`saved-topic-card-${item.id}`}
                          onClick={() => {
                            setInputValue(item.title);
                            handleLoadExplainer(item.id, item.title);
                          }}
                          className="group bg-white p-5 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300 transition-all cursor-pointer flex justify-between items-start text-left"
                        >
                          <div className="space-y-3">
                            <div className="flex items-center gap-2">
                              <span className="p-2 bg-slate-50 group-hover:bg-indigo-50 group-hover:text-indigo-600 rounded-xl transition">
                                {renderCardIcon(item.iconType)}
                              </span>
                              {getCategoryBadge(item.category)}
                            </div>
                            
                            <div>
                              <h3 className="text-lg font-bold text-slate-800 group-hover:text-indigo-600 transition tracking-tight">
                                {item.title}
                              </h3>
                              <p className="text-slate-400 text-xs flex items-center gap-1.5 mt-1 font-medium font-mono">
                                <Clock className="w-3.5 h-3.5" />
                                Last taught: {item.lastUsed} • {item.stepsCount} visual steps
                              </p>
                            </div>
                          </div>

                          <span className="p-1.5 text-slate-400 group-hover:text-indigo-600 group-hover:bg-slate-50 rounded-lg transition-all self-center">
                            <ChevronRight className="w-5 h-5" />
                          </span>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </div>

                {/* Smartboard Display Tip Footer */}
                <div className="bg-indigo-50/50 border border-indigo-100 rounded-2xl p-4 flex gap-3 text-slate-700 text-sm">
                  <Info className="w-5 h-5 text-indigo-600 flex-shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <p className="font-bold text-slate-800">Smartboard Display Mode Optimized</p>
                    <p className="text-slate-600 leading-relaxed">
                      Tap any topic or type a custom query. The generated slides feature large fonts, high contrast ratios, and touch-drag zones to easily show flows from across the classroom.
                    </p>
                  </div>
                </div>

              </motion.div>
            )}

            {/* Screen State: LOADING STATE (Process simulation) */}
            {screenState === "loading" && (
              <motion.div
                key="loading-screen"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.02 }}
                transition={{ duration: 0.25 }}
                className="bg-white rounded-3xl p-8 sm:p-12 shadow-xl shadow-slate-200 border border-slate-100 text-center space-y-8 max-w-2xl mx-auto"
              >
                {apiError && (
                  <div className="bg-red-50 text-red-700 p-4 rounded-xl text-sm font-medium border border-red-200 mb-4 text-left">
                    <span className="font-bold">Error:</span> {apiError}
                  </div>
                )}
                <div className="relative w-24 h-24 mx-auto flex items-center justify-center">
                  {/* Rotating elegant custom circular loader */}
                  <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-t-indigo-600 border-r-indigo-600 rounded-full animate-spin"></div>
                  <Layers className="w-8 h-8 text-indigo-600 animate-pulse" />
                </div>

                <div className="space-y-3">
                  <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-800 tracking-tight font-display">
                    Synthesizing Flow Blueprint
                  </h2>
                  <div className="px-4 py-2 bg-indigo-50/80 rounded-xl inline-block max-w-full">
                    <p className="text-indigo-900 font-bold text-sm sm:text-base truncate">
                      Topic: "{generatedTopic}"
                    </p>
                  </div>
                </div>

                {/* Loading step progression visualizer */}
                <div className="max-w-md mx-auto space-y-4">
                  <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-indigo-600"
                      initial={{ width: "0%" }}
                      animate={{ 
                        width: `${((loadingStep + 1) / (loadingMessages.length + 1)) * 100}%` 
                      }}
                      transition={{ duration: 0.4 }}
                    />
                  </div>
                  
                  <div className="h-10 flex items-center justify-center">
                    <AnimatePresence mode="wait">
                      <motion.p
                        key={loadingStep}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        className="text-slate-500 font-semibold text-sm sm:text-base font-mono"
                      >
                        {loadingStep < loadingMessages.length 
                          ? loadingMessages[loadingStep].text 
                          : "Finalizing packaging..."}
                      </motion.p>
                    </AnimatePresence>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row justify-center gap-3 pt-4 border-t border-slate-100">
                  <button
                    id="cancel-generation-btn"
                    onClick={() => {
                      setScreenState("default");
                      setLoadingStep(0);
                    }}
                    className="px-5 py-2.5 text-slate-500 hover:text-slate-800 font-semibold text-sm rounded-xl transition cursor-pointer"
                  >
                    Cancel Generation
                  </button>
                </div>
              </motion.div>
            )}

            {/* Screen State: SUCCESS STATE (Ready to launch) */}
            {screenState === "success" && (
              <motion.div
                key="success-screen"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.3 }}
                className="bg-white rounded-3xl p-8 sm:p-12 shadow-xl shadow-slate-200 border border-slate-100 text-center space-y-8 max-w-2xl mx-auto"
              >
                <div className="inline-flex p-4 bg-emerald-50 text-emerald-600 rounded-full">
                  <CheckCircle2 className="w-12 h-12" />
                </div>

                <div className="space-y-2">
                  <span className="text-xs font-bold tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full uppercase font-mono">
                    Visual Explainer Synthesized
                  </span>
                  <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-800 tracking-tight font-display">
                    Interactive Presentation Ready!
                  </h2>
                  <p className="text-slate-500 text-base sm:text-lg max-w-md mx-auto">
                    The visual flow has been created and indexed with multi-layered controls. Ready to be projected in class.
                  </p>
                </div>

                <div className="bg-slate-50 p-5 rounded-2xl border border-slate-100 space-y-2 max-w-md mx-auto">
                  <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">Current Active Topic</div>
                  <div className="text-xl font-bold text-slate-800">{generatedTopic}</div>
                  <div className="text-xs text-indigo-600 font-bold bg-indigo-50 py-1 px-3 rounded-full inline-block mt-2">
                    4 Step Interactive Slide Deck Generated
                  </div>
                </div>

                {/* Buttons to View or Return */}
                <div className="flex flex-col sm:flex-row gap-3 justify-center max-w-md mx-auto">
                  <button
                    id="view-explainer-btn"
                    onClick={() => {
                      setPreviewStep(0);
                      setShowPreviewModal(true);
                    }}
                    className="flex-1 py-4 px-6 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-bold rounded-2xl transition shadow-lg shadow-indigo-100 flex items-center justify-center gap-2 cursor-pointer text-lg min-h-[56px]"
                  >
                    <MonitorPlay className="w-5.5 h-5.5" />
                    <span>View Explainer</span>
                  </button>

                  <button
                    id="start-over-btn"
                    onClick={() => {
                      setInputValue("");
                      setScreenState("default");
                    }}
                    className="py-4 px-6 bg-slate-100 hover:bg-slate-200 active:bg-slate-300 text-slate-700 font-bold rounded-2xl transition flex items-center justify-center gap-2 cursor-pointer text-base"
                  >
                    <span>Create Another</span>
                  </button>
                </div>

                <div className="text-xs text-slate-400 font-mono pt-2">
                  This topic was saved automatically to your offline library history.
                </div>
              </motion.div>
            )}

          </AnimatePresence>

        </div>
      </main>

      {/* 
        PREVIEW PLAYER MODAL (Gives the user the feeling of a complete presentation mockup!)
        Optimized for Classroom Smartboards
      */}
      <AnimatePresence>
        {showPreviewModal && (
          <motion.div
            id="smartboard-presentation-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-slate-950/90 z-50 flex items-center justify-center p-4 sm:p-6"
          >
            <motion.div
              initial={{ scale: 0.95, y: 15 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 15 }}
              transition={{ type: "spring", damping: 25, stiffness: 350 }}
              className="bg-white w-full max-w-5xl rounded-3xl overflow-hidden shadow-2xl flex flex-col h-[85vh] sm:h-[80vh]"
            >
              
              {/* Modal Header */}
              <div className="bg-slate-900 px-6 py-4 flex items-center justify-between text-white border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="p-1.5 bg-indigo-600 rounded-lg text-white">
                    <Layers className="w-5 h-5" />
                  </span>
                  <div>
                    <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest font-mono">Smartboard Presentation Mode</div>
                    <div className="text-base sm:text-lg font-bold truncate max-w-xs sm:max-w-md text-slate-100">
                      {generatedTopic}
                    </div>
                  </div>
                </div>

                <button
                  id="close-preview-modal-btn"
                  onClick={() => setShowPreviewModal(false)}
                  className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-xl transition cursor-pointer"
                  title="Close presentation"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Interactive Presentation Body */}
              {generatedExplainer?.html_url ? (
                /* ── Animated HTML page from B2 ───────────────────────────── */
                <div className="flex-1 flex flex-col bg-slate-950">
                  <iframe
                    id="animated-explainer-iframe"
                    src={generatedExplainer.html_url}
                    title={`Animated explainer: ${generatedTopic}`}
                    className="flex-1 w-full border-0"
                    sandbox="allow-scripts allow-same-origin"
                  />
                  <div className="bg-slate-900 px-6 py-3 text-slate-400 text-xs flex justify-between items-center border-t border-slate-800 font-mono">
                    <a
                      href={generatedExplainer.html_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-indigo-400 hover:text-indigo-300 underline"
                    >
                      Open full-screen ↗
                    </a>
                    <span className="hidden sm:inline">Spacebar = Next step · ESC = Close</span>
                  </div>
                </div>
              ) : (
                /* ── Slide-deck fallback (no HTML yet or B2 not configured) ─ */
                <div className="flex-1 bg-slate-950 p-6 sm:p-8 flex flex-col justify-between text-white overflow-y-auto">
                  
                  {/* Active Slide Visual Layout */}
                  <div className="space-y-6 flex-1 flex flex-col justify-center max-w-3xl mx-auto w-full">
                    
                    {/* Step indicators */}
                    <div className="flex items-center gap-2 justify-center">
                      {slides.map((_, idx) => (
                        <button
                          key={idx}
                          id={`step-indicator-btn-${idx}`}
                          onClick={() => setPreviewStep(idx)}
                          className={`h-2.5 rounded-full transition-all cursor-pointer ${
                            idx === previewStep ? "w-10 bg-indigo-500" : "w-2.5 bg-slate-700 hover:bg-slate-500"
                          }`}
                          title={`Go to step ${idx + 1}`}
                        />
                      ))}
                    </div>

                    {/* Active Slide Content */}
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={previewStep}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.25 }}
                        className="space-y-6 bg-slate-900/60 p-6 sm:p-8 rounded-2xl border border-slate-800"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs sm:text-sm font-bold tracking-widest text-indigo-400 uppercase font-mono">
                            STEP {previewStep + 1} OF {slides.length}
                          </span>
                          <span className="px-2.5 py-0.5 bg-indigo-500/20 text-indigo-300 rounded text-xs font-mono">
                            Live Active
                          </span>
                        </div>

                        <h3 className="text-2xl sm:text-3xl font-extrabold text-slate-100 font-display">
                          {slides[previewStep].title}
                        </h3>

                        <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
                          {slides[previewStep].description}
                        </p>

                        {/* Dynamic interactive mock bullet visualization list */}
                        <div className="space-y-2.5 pt-2">
                          {slides[previewStep].bullets.map((bullet, bIdx) => (
                            <div key={bIdx} className="flex items-start gap-2.5">
                              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                              <span className="text-slate-400 text-sm sm:text-base">{bullet}</span>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    </AnimatePresence>

                  </div>

                  {/* Presentation Navigation Controls (Large targets for Touchscreens / Smartboards) */}
                  <div className="mt-6 pt-6 border-t border-slate-900 flex items-center justify-between gap-4 max-w-3xl mx-auto w-full">
                    
                    <button
                      id="prev-slide-btn"
                      onClick={() => setPreviewStep(prev => Math.max(0, prev - 1))}
                      disabled={previewStep === 0}
                      className={`flex items-center gap-2 py-3 px-5 sm:px-6 rounded-xl font-bold transition-all text-sm sm:text-base cursor-pointer min-h-[48px] ${
                        previewStep === 0
                          ? "bg-slate-900 text-slate-600 cursor-not-allowed opacity-50"
                          : "bg-slate-800 hover:bg-slate-700 text-white"
                      }`}
                    >
                      <ChevronLeft className="w-5 h-5" />
                      <span>Previous</span>
                    </button>

                    <div className="text-slate-400 text-xs sm:text-sm font-mono font-bold">
                      Slide {previewStep + 1} / {slides.length}
                    </div>

                    {previewStep < slides.length - 1 ? (
                      <button
                        id="next-slide-btn"
                        onClick={() => setPreviewStep(prev => Math.min(slides.length - 1, prev + 1))}
                        className="flex items-center gap-2 py-3 px-5 sm:px-6 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl transition-all text-sm sm:text-base cursor-pointer min-h-[48px]"
                      >
                        <span>Next Step</span>
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    ) : (
                      <button
                        id="complete-slide-btn"
                        onClick={() => setShowPreviewModal(false)}
                        className="flex items-center gap-2 py-3 px-5 sm:px-6 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-xl transition-all text-sm sm:text-base cursor-pointer min-h-[48px]"
                      >
                        <CheckCircle2 className="w-5 h-5" />
                        <span>Finish Preview</span>
                      </button>
                    )}

                  </div>

                </div>
              )}

              {/* Modal Footer Controls info */}
              <div className="bg-slate-900 px-6 py-3 text-slate-400 text-xs flex justify-between items-center border-t border-slate-800 font-mono">
                <span>Rendering Engine: PulseBoard Visual Core</span>
                <span className="hidden sm:inline">Press ESC or click outside to dismiss preview</span>
              </div>


            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Footer */}
      <footer className="mt-auto bg-white border-t border-slate-200 py-6 text-center text-slate-400 text-sm font-medium">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-slate-400" />
            <span className="text-slate-600 font-bold font-display">PulseBoard</span>
            <span className="text-slate-400 font-mono">• Academic Smartboard Tool</span>
          </div>
          <p className="text-xs text-slate-400">
            © 2026 PulseBoard. Created for educators to transform complex process-flows into clean interactive displays.
          </p>
        </div>
      </footer>

    </div>
  );
}
