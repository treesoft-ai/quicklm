import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Type definitions for Chat Preset
interface ChatPreset {
  question: string;
  answer: string;
}

// Preset Q&As tailored for QuickLM
const PRESET_RESPONSES: ChatPreset[] = [
  {
    question: "Explain how speculative decoding beats the memory bandwidth wall.",
    answer: "In standard decoding, LLMs are strictly memory-bandwidth bound: for every single token generated, we must load the entire model's weights from RAM to the CPU cache. If our weights are 1.6GB and memory bandwidth is 50GB/s, we are capped at ~31 tokens/sec regardless of compute power.\n\nSpeculative decoding solves this by running a small, cheap draft model (like a quantized 0.8B int4 model) to guess K tokens ahead. Since it's small, it loads much faster. The large model then verifies all K tokens in a single batched forward pass. Because math::matmul computes multiple rows at once, the big weights are read only once from RAM for the entire batch. If the draft model's acceptance rate is high, we generate multiple tokens per weight-load cycle, effectively 'beating' the bandwidth limit and achieving ~3.5x speedups on consumer hardware."
  },
  {
    question: "Why does Gated DeltaNet not use a standard KV-cache?",
    answer: "Qwen 3.5 uses a hybrid architecture containing 18 Gated DeltaNet linear attention layers and 6 full GQA attention layers. Standard GQA layers compute attention by storing the key and value representations of all prior tokens (KV-cache), which grows linearly with sequence length.\n\nGated DeltaNet layers, on the other hand, are designed as linear attention mechanisms with a recurrent state. Instead of storing a history of keys and values, they compress history into a fixed-size state matrix that is updated in-place via a causal conv1d and a recurrent gate formula for every token. This means 18 out of 24 layers in Qwen 3.5 have zero KV-cache memory growth, significantly reducing the memory footprint for long contexts."
  },
  {
    question: "What are the performance optimizations in Quick Inference v1?",
    answer: "Quick Inference (QI) v1 implements several lossless performance techniques:\n\n1. AVX2 Vectorization: Highly optimized AVX2 FMA kernels for half-precision float (bf16) computations, upcasting weights in-register.\n2. Software Prefetching: Streams the next layer's weights into CPU cache using _mm_prefetch instructions while the current layer is executing.\n3. Kernel Fusion: Combines adjacent element-wise operations (like SiLU + element-wise multiplication in SwiGLU, and residual addition + RMSNorm) to keep values in registers and reduce RAM round-trips.\n4. Batched Prefill: Processes the prompt prefix in a single batched pass (forward_batch), reducing prefill times by ~2.2x."
  },
  {
    question: "How do I build and run QuickLM locally?",
    answer: "QuickLM is a zero-dependency C++17 project. You can build it using MSVC (NMake) on Windows or standard Make:\n\n# Clone the repository\ngit clone https://github.com/treesoft-ai/quicklm.git\ncd quicklm\n\n# Build on Windows (Developer PowerShell for VS)\nnmake -f Makefile\n\n# Or using CMake\ncmake -B build\ncmake --build build --config Release\n\nTo run inference, provide the model checkpoint directory and either a prompt or interactive mode:\n./quicklm --path models/Qwen3.5-0.8B --prompt \"Hello\" --optimize speculative,kv-reuse"
  },
  {
    question: "What is the hybrid layer distribution of Qwen 3.5?",
    answer: "Qwen 3.5 uses a hybrid attention design that interleaves linear attention with conventional Grouped-Query Attention (GQA). Specifically, across its 24 total layers:\n\n• 18 layers are Gated DeltaNet (linear attention with causal conv1d and recurrent state).\n• 6 layers are standard GQA (full attention with causal mask).\n\nThis hybrid arrangement provides the expressive capacity and retrieval strength of full attention while dramatically lowering RAM footprint and computational complexity."
  },
  {
    question: "Explain zero-centered RMSNorm in linear attention.",
    answer: "Standard Root Mean Square Normalization (RMSNorm) regulates activations by scaling based on variance. In linear attention layers like Gated DeltaNet, activations can grow exponentially along the sequence. Zero-centered RMSNorm stabilizes head projections (q and k) by calculating variance around zero rather than the mean.\n\nThis guarantees numerical stability in register arithmetic during high-throughput recurrence, especially when weights are loaded at half-precision (bf16) and upcasted in-register."
  },
  {
    question: "How does thread-count tuning affect DDR4 memory traffic?",
    answer: "QuickLM decode steps are memory-bandwidth bound. Matrix multiplications are split across threads using a custom thread pool. If you set thread count too low, you underutilize CPU execution ports. If you set thread count too high (past the CPU physical core limit into logical hyperthreads), threads begin competing for shared L1/L2 caches and memory controllers.\n\nThis cache trashing increases DDR4 memory traffic, causing latency spikes and decreasing overall tokens/sec."
  },
  {
    question: "Why is PyTorch not required for this inference engine?",
    answer: "QuickLM is written in standard C++17 from scratch. It directly implements safetensors file parsing, JSON config mapping, BPE tokenization, matrix multiplication, and attention kernels natively.\n\nBy bypassing PyTorch's extensive runtime stack and virtual machine overhead, QuickLM avoids gigabytes of library bloat, loads model checkpoints in milliseconds, and runs on consumer CPU chips with zero external software requirements."
  }
];

interface SpeculativeStep {
  draft: string[];
  results: { token: string; accepted: boolean }[];
  resampled?: string;
}

export default function App() {
  // Navigation Tabs State
  type TabType = "overview" | "playground" | "trace" | "faq";
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({ opacity: 0 });

  // Refs for tab buttons
  const navContainerRef = useRef<HTMLDivElement>(null);
  const overviewTabRef = useRef<HTMLButtonElement>(null);
  const playgroundTabRef = useRef<HTMLButtonElement>(null);
  const traceTabRef = useRef<HTMLButtonElement>(null);
  const faqTabRef = useRef<HTMLButtonElement>(null);

  // Playground Config States
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(40);
  const [threads, setThreads] = useState(8);
  const [precision, setPrecision] = useState<"bf16" | "int8" | "int4">("bf16");
  const [optPrefetch, setOptPrefetch] = useState(true);
  const [optFusion, setOptFusion] = useState(true);
  const [optKvReuse, setOptKvReuse] = useState(true);
  const [optSpeculative, setOptSpeculative] = useState(true);

  // Playground Chat States
  const [inputValue, setInputValue] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentResponseText, setCurrentResponseText] = useState("");
  const [queryCount, setQueryCount] = useState<number>(() => {
    const saved = localStorage.getItem("quicklm_demo_count");
    return saved ? parseInt(saved, 10) : 0;
  });

  // Performance metrics of the current run
  const [metrics, setMetrics] = useState({
    tokensSec: 0,
    promptTokens: 0,
    completionTokens: 0,
    latency: 0,
    prefillTime: 0,
    reusedTokens: 0
  });

  // Speculative trace data to show under output
  const [specTrace, setSpecTrace] = useState<SpeculativeStep[]>([]);
  
  // Terminal log trace in the trace tab
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  // General States
  const [copied, setCopied] = useState(false);
  const [toastMsg, setToastMsg] = useState("");
  const [showToast, setShowToast] = useState(false);
  const [faqOpen, setFaqOpen] = useState<Record<number, boolean>>({ 0: true });

  const queryInputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea height dynamic adjustment
  useEffect(() => {
    const qInput = queryInputRef.current;
    if (qInput) {
      qInput.style.height = "52px";
      const scrollHeight = qInput.scrollHeight;
      if (scrollHeight > 52) {
        qInput.style.height = `${scrollHeight}px`;
      }
    }
  }, [inputValue]);

  // Navigation indicator aligner
  const alignIndicator = (element: HTMLButtonElement | null) => {
    if (!element) return;
    setIndicatorStyle({
      left: element.offsetLeft + "px",
      width: element.offsetWidth + "px",
      top: element.offsetTop + "px",
      height: element.offsetHeight + "px",
      opacity: 1
    });
  };

  useEffect(() => {
    let activeRef: HTMLButtonElement | null = null;
    if (activeTab === "overview") activeRef = overviewTabRef.current;
    else if (activeTab === "playground") activeRef = playgroundTabRef.current;
    else if (activeTab === "trace") activeRef = traceTabRef.current;
    else if (activeTab === "faq") activeRef = faqTabRef.current;

    alignIndicator(activeRef);

    const timer = setTimeout(() => alignIndicator(activeRef), 50);
    const frame = requestAnimationFrame(() => alignIndicator(activeRef));

    const handleResize = () => {
      alignIndicator(activeRef);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      clearTimeout(timer);
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", handleResize);
    };
  }, [activeTab]);

  const handleMouseEnter = (e: React.MouseEvent<HTMLButtonElement>) => {
    alignIndicator(e.currentTarget);
  };

  const handleMouseLeave = () => {
    let activeRef: HTMLButtonElement | null = null;
    if (activeTab === "overview") activeRef = overviewTabRef.current;
    else if (activeTab === "playground") activeRef = playgroundTabRef.current;
    else if (activeTab === "trace") activeRef = traceTabRef.current;
    else if (activeTab === "faq") activeRef = faqTabRef.current;

    alignIndicator(activeRef);
  };

  // Toast Trigger
  const triggerToast = (msg: string) => {
    setToastMsg(msg);
    setShowToast(true);
    setTimeout(() => {
      setShowToast(false);
    }, 3000);
  };

  // Copy Address
  const handleCopyAddress = () => {
    const addr = "0x71C7656EC7ab88b098defB751B7401B5f6d8976F";
    navigator.clipboard.writeText(addr).then(() => {
      setCopied(true);
      triggerToast("USDC Base address copied!");
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // Switch to Playground and focus input
  const transitionToDemo = () => {
    setActiveTab("playground");
    setTimeout(() => {
      queryInputRef.current?.focus();
    }, 100);
  };

  // Reset demo query counter
  const resetDemoCounter = () => {
    localStorage.removeItem("quicklm_demo_count");
    setQueryCount(0);
    setChatHistory([]);
    setCurrentResponseText("");
    setSpecTrace([]);
    triggerToast("Demo limits and session reset.");
  };

  // Setup terminal logs
  const appendTerminalLog = (msg: string) => {
    setTerminalLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  useEffect(() => {
    if (terminalBottomRef.current) {
      terminalBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalLogs]);

  // Execute Generation
  const executeGeneration = (promptText: string) => {
    const prompt = promptText.trim();
    if (!prompt) return;

    // QuickLM is free and open source, no local session limits are enforced.

    setIsGenerating(true);
    setSpecTrace([]);
    setCurrentResponseText("");

    // Append User message
    const updatedHistory = [...chatHistory, { role: "user" as const, text: prompt }];
    setChatHistory(updatedHistory);

    // Initial setup logs
    appendTerminalLog(`cli::parse_arguments() -> path: "models/Qwen3.5-0.8B", optimize: [${[
      optPrefetch ? 'prefetch' : '',
      optFusion ? 'fusion' : '',
      optKvReuse ? 'kv-reuse' : '',
      optSpeculative ? 'speculative' : ''
    ].filter(Boolean).join(', ')}]`);

    appendTerminalLog(`model::load() -> Initializing weight loader (precision: ${precision})...`);

    // Determine performance parameters
    let baseTokSec = 3.8;
    if (precision === "int4") baseTokSec += 0.8;
    if (precision === "int8") baseTokSec += 0.4;
    if (optPrefetch) baseTokSec += 0.6;
    if (optFusion) baseTokSec += 0.2;
    
    // Speculative gives a huge speed boost if greedy decoding (temp = 0)
    const isSpeculativeEffective = optSpeculative && temperature === 0;
    if (isSpeculativeEffective) {
      baseTokSec = precision === "int4" ? 14.5 : 13.5;
    }

    // Determine kv-reuse benefits
    let reusedCount = 0;
    let prefillTime = 2.2; // base prefill time in seconds for a standard prompt

    if (optKvReuse && chatHistory.length > 0) {
      // simulate reuse of previous prompt tokens
      reusedCount = Math.floor(updatedHistory.length * 35);
      prefillTime = 0.15; // sub-second prefill
      appendTerminalLog(`kv-reuse::check() -> MATCH FOUND. Reused ${reusedCount} tokens in cache. Prefill skipped.`);
    } else {
      appendTerminalLog(`model::prefill() -> forward_batch on ${Math.floor(prompt.length / 4)} tokens. Time: ${prefillTime.toFixed(3)}s`);
    }

    appendTerminalLog(`decode::start() -> temperature: ${temperature}, top_k: ${topK}, threads: ${threads}`);

    // Select response content
    const matchedPreset = PRESET_RESPONSES.find(p => p.question.toLowerCase() === prompt.toLowerCase());
    const fullTextResponse = matchedPreset ? matchedPreset.answer : 
      "QuickLM is currently running a mock simulation of the C++ CLI. Your options are parsed and applied to token speed calculations.\n\nThe zero-dependency AVX2 thread pool handles layers dynamically, achieving robust throughput. To test custom prompts with native compiled speed, clone the codebase and build locally.";

    const words = fullTextResponse.split(' ');
    let currentWordIdx = 0;
    let accumulatedText = "";
    
    const promptTokens = Math.floor(prompt.length / 4.2) + reusedCount;
    const completionTokens = Math.floor(fullTextResponse.length / 4.2);

    const streamInterval = 1000 / baseTokSec;
    
    // Simulate generation loop
    const timer = setInterval(() => {
      if (currentWordIdx >= words.length) {
        clearInterval(timer);
        setIsGenerating(false);
        setQueryCount(prev => {
          const next = prev + 1;
          localStorage.setItem("quicklm_demo_count", next.toString());
          return next;
        });
        
        const finalLatency = prefillTime + (completionTokens / baseTokSec);
        setMetrics({
          tokensSec: baseTokSec,
          promptTokens,
          completionTokens,
          latency: finalLatency,
          prefillTime,
          reusedTokens: reusedCount
        });

        // Add to history
        setChatHistory(prev => [...prev, { role: "assistant" as const, text: accumulatedText }]);
        setCurrentResponseText("");
        
        appendTerminalLog(`decode::complete() -> tokens generated: ${completionTokens}, speed: ${baseTokSec.toFixed(2)} tok/sec`);
        return;
      }

      // Add a couple of words per tick
      const sliceCount = isSpeculativeEffective ? 4 : 1;
      const currentSlice = words.slice(currentWordIdx, currentWordIdx + sliceCount);
      currentWordIdx += sliceCount;

      accumulatedText += (accumulatedText ? " " : "") + currentSlice.join(" ");
      setCurrentResponseText(accumulatedText);

      // If speculative, generate trace visual step
      if (isSpeculativeEffective) {
        const results = currentSlice.map((w, index) => {
          // 85% acceptance rate simulation
          const accepted = Math.random() > 0.15 || index === 0; 
          return { token: w, accepted };
        });

        // Find first rejected index
        const firstRejectedIdx = results.findIndex(r => !r.accepted);
        const correctedResults = [...results];
        let resampledToken = undefined;

        if (firstRejectedIdx !== -1) {
          // mock resample
          resampledToken = "§" + words[currentWordIdx - sliceCount + firstRejectedIdx];
          // truncate accepted results at first rejected
          for (let i = firstRejectedIdx; i < correctedResults.length; i++) {
            correctedResults[i].accepted = false;
          }
        }

        const step: SpeculativeStep = {
          draft: currentSlice,
          results: correctedResults,
          resampled: resampledToken
        };

        setSpecTrace(prev => [...prev.slice(-3), step]); // keep last 3 steps visually
      }

      if (currentWordIdx % 10 === 0) {
        appendTerminalLog(`decode::step() -> pos: ${promptTokens + currentWordIdx}, speed: ${baseTokSec.toFixed(2)} tok/sec`);
      }

    }, streamInterval * (isSpeculativeEffective ? 3 : 1));

    setInputValue("");
  };

  const handleQueryKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && inputValue.trim()) {
        executeGeneration(inputValue);
      }
    }
  };

  return (
    <div className="container">
      {/* Header and Branding */}
      <header>
        <div className="brand-bar">
          <a href="#" className="brand" onClick={(e) => { e.preventDefault(); setActiveTab("overview"); }}>
            <span>QuickLM</span>
            <span className="brand-byline">by TreeSoft <span className="brand-ai">[AI]</span></span>
          </a>
          <div className="meta-stats">
            <span>AVX2 Opt</span>
            <span>/</span>
            <span>Zero-Dep</span>
            <span>/</span>
            <a 
              className="reset-link" 
              href="https://github.com/treesoft-ai/quicklm" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              MIT License
            </a>
          </div>
        </div>

        {/* Navigation Tabs with sliding highlight bar */}
        <nav ref={navContainerRef} onMouseLeave={handleMouseLeave}>
          <div className="nav-indicator" style={indicatorStyle} />

          <button
            id="nav-overview"
            ref={overviewTabRef}
            className={`nav-btn ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
            onMouseEnter={handleMouseEnter}
            data-tooltip="Read project overview and performance metrics"
          >
            Overview
          </button>
          
          <button
            ref={playgroundTabRef}
            className={`nav-btn ${activeTab === "playground" ? "active" : ""}`}
            onClick={() => setActiveTab("playground")}
            onMouseEnter={handleMouseEnter}
            data-tooltip="Run CPU inference simulations on Qwen 3.5"
          >
            Interactive Console
          </button>

          <button
            ref={traceTabRef}
            className={`nav-btn ${activeTab === "trace" ? "active" : ""}`}
            onClick={() => setActiveTab("trace")}
            onMouseEnter={handleMouseEnter}
            data-tooltip="Trace C++ compiler thread and kernel execution logs"
          >
            Execution Trace
          </button>
          
          <span className="nav-separator">/</span>
          
          <button
            id="nav-faq"
            ref={faqTabRef}
            className={`nav-btn ${activeTab === "faq" ? "active" : ""}`}
            onClick={() => setActiveTab("faq")}
            onMouseEnter={handleMouseEnter}
            data-tooltip="Answers regarding compilation, threads, and hybrid architecture"
          >
            FAQ
          </button>
        </nav>
      </header>

      {/* Main Viewport */}
      <main>
        {/* OVERVIEW TAB */}
        <div className={`view ${activeTab === "overview" ? "active" : ""}`}>
          <div className="overview-card">
            <div className="overview-title">Lossless, zero-dependency C++17 Qwen 3.5 CPU Inference.</div>
            <p className="overview-pitch">
              QuickLM is an optimized-from-scratch inference stack. Written in pure standard library C++, it achieves native compiled speeds (up to 14.5 tokens/sec) on consumer DDR4 CPU architectures without PyTorch, BLAS, or CUDA runtime wrappers. Explore the performance improvements, try the interactive sandbox, and read the open-source code.
            </p>
            <div className="overview-actions">
              <button className="action-btn" onClick={transitionToDemo}>Open Sandbox Console</button>
              <a 
                href="https://github.com/treesoft-ai/quicklm" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="action-btn primary-accent"
              >
                Get Source Code
              </a>
            </div>
          </div>

          <div className="section-label">Lossless Performance Speedups</div>
          <div className="features-panel">
            <div className="feature-box">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                  <polyline points="2 17 12 22 22 17"></polyline>
                  <polyline points="2 12 12 17 22 12"></polyline>
                </svg>
                <span>Memory Bandwidth Limits</span>
              </h3>
              <p>
                In CPU inference, the major ceiling is memory read speed (streaming weights from RAM to L3/Registers). QuickLM implements prefetching and math kernel fusion to avoid register roundtrips, getting near-limit hardware memory throughput.
              </p>
            </div>

            <div className="feature-box">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span>Speculative Decoding</span>
              </h3>
              <p>
                Uses a self-speculative draft mechanism (loading the 0.8B Qwen checkpoints concurrently at int4 vs bf16 precision) to propose multiple tokens at a low cost, verifying them collectively in one batched forward pass to beat the bandwidth wall.
              </p>
            </div>

            <div className="feature-box">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                </svg>
                <span>Linear DeltaNet Interleaving</span>
              </h3>
              <p>
                Qwen 3.5 features hybrid Gated DeltaNet attention layers. Because recurrent weights collapse history into a static recurrent matrix, 18 of the 24 layer architectures feature zero KV-cache memory inflation, letting you run massive context lengths.
              </p>
            </div>

            <div className="feature-box">
              <h3>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="16 18 22 12 16 6"></polyline>
                  <polyline points="8 6 2 12 8 18"></polyline>
                </svg>
                <span>Pluggable Registry</span>
              </h3>
              <p>
                Features a clean `IArchitecture` virtual register setup. Easily hook up traditional Qwen, LLaMA, or custom hybrid layers without having to edit the core decode runtime loops or weight mapping components.
              </p>
            </div>
          </div>
        </div>

        {/* PLAYGROUND TAB */}
        <div className={`view ${activeTab === "playground" ? "active" : ""}`}>
          <div className="playground-layout">
            
            {/* Sidebar Controls */}
            <div className="settings-sidebar">
              <div className="settings-group">
                <span className="settings-label">Model Settings</span>
                
                <div className="setting-row">
                  <span>Temperature</span>
                  <select 
                    value={temperature} 
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setTemperature(val);
                      // Speculative decoding in QuickLM is only active on greedy decoding (temp = 0)
                      if (val > 0) {
                        setOptSpeculative(false);
                      }
                    }}
                  >
                    <option value={0.0}>0.0 (Greedy)</option>
                    <option value={0.7}>0.7 (Balanced)</option>
                    <option value={1.0}>1.0 (Creative)</option>
                  </select>
                </div>

                <div className="setting-row">
                  <span>Top-k</span>
                  <select value={topK} onChange={(e) => setTopK(parseInt(e.target.value, 10))}>
                    <option value={40}>40</option>
                    <option value={0}>0 (Disabled)</option>
                  </select>
                </div>

                <div className="setting-row">
                  <span>Threads</span>
                  <select value={threads} onChange={(e) => setThreads(parseInt(e.target.value, 10))}>
                    <option value={4}>4 Cores</option>
                    <option value={8}>8 Cores</option>
                    <option value={12}>12 Cores</option>
                    <option value={16}>16 Cores</option>
                  </select>
                </div>

                <div className="setting-row">
                  <span>Precision</span>
                  <select value={precision} onChange={(e) => setPrecision(e.target.value as any)}>
                    <option value="bf16">bf16 (Lossless)</option>
                    <option value="int8">int8 (Slight Loss)</option>
                    <option value="int4">int4 (Fastest)</option>
                  </select>
                </div>
              </div>

              <div className="settings-group">
                <span className="settings-label">Optimizations</span>
                
                <label className="checkbox-label">
                  <input type="checkbox" checked={optPrefetch} onChange={(e) => setOptPrefetch(e.target.checked)} />
                  <span>Prefetching</span>
                </label>

                <label className="checkbox-label">
                  <input type="checkbox" checked={optFusion} onChange={(e) => setOptFusion(e.target.checked)} />
                  <span>Kernel Fusion</span>
                </label>

                <label className="checkbox-label">
                  <input type="checkbox" checked={optKvReuse} onChange={(e) => setOptKvReuse(e.target.checked)} />
                  <span>KV-Cache Reuse</span>
                </label>

                <label className="checkbox-label">
                  <input 
                    type="checkbox" 
                    checked={optSpeculative} 
                    onChange={(e) => {
                      if (temperature > 0) {
                        triggerToast("Speculative requires temperature 0 (greedy).");
                        return;
                      }
                      setOptSpeculative(e.target.checked);
                    }} 
                  />
                  <span>Speculative</span>
                </label>
              </div>

              <div className="settings-group" style={{ marginTop: 'auto' }}>
                <button className="action-btn" onClick={resetDemoCounter}>
                  Clear Session Cache
                </button>
              </div>
            </div>

            {/* Chat Sandbox area */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Predefined Prompts list - Click to execute inference immediately */}
              {!isGenerating && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div className="section-label">Select Predefined Prompt to Run</div>
                  <div className="preset-grid">
                    {PRESET_RESPONSES.map((preset, idx) => (
                      <div 
                        key={idx} 
                        className="preset-card"
                        onClick={() => executeGeneration(preset.question)}
                      >
                        {preset.question}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Chat history list */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {chatHistory.map((msg, index) => (
                  <div key={index} style={{ borderBottom: '1px solid var(--border)', paddingBottom: '16px' }}>
                    <div className="section-label" style={{ marginBottom: '8px' }}>
                      {msg.role === "user" ? "User" : "QuickLM Core output"}
                    </div>
                    <div style={{ fontSize: '0.9rem', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                      {msg.text}
                    </div>
                  </div>
                ))}

                {/* Current streaming response */}
                {isGenerating && (
                  <div>
                    <div className="section-label" style={{ marginBottom: '8px' }}>
                      QuickLM Core output (streaming...)
                    </div>
                    <div style={{ fontSize: '0.9rem', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                      {currentResponseText}
                      <span className="spinner" style={{ marginLeft: '8px' }}></span>
                    </div>
                  </div>
                )}
              </div>

              {/* Speculative badge visualization */}
              {isGenerating && optSpeculative && specTrace.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div className="section-label">Speculative Verification Steps (Active)</div>
                  <div className="token-spec-display">
                    {specTrace.map((step, idx) => (
                      <div key={idx} style={{ display: 'flex', gap: '6px', alignItems: 'center', borderRight: '1px solid var(--border)', paddingRight: '8px' }}>
                        {step.results.map((res, tokenIdx) => (
                          <span 
                            key={tokenIdx} 
                            className={`token-badge ${res.accepted ? 'accepted' : 'rejected'}`}
                            title={res.accepted ? 'Accepted by target' : 'Rejected by target, rolled back'}
                          >
                            {res.token}
                          </span>
                        ))}
                        {step.resampled && (
                          <span className="token-badge resampled" title="Corrective target token resampled">
                            {step.resampled.substring(1)}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Performance Metrics Panel */}
              {metrics.latency > 0 && !isGenerating && (
                <div className="perf-bar-container">
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: '8px', marginBottom: '8px' }}>
                    <span className="settings-label">Performance Metrics</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
                      Total Latency: {metrics.latency.toFixed(3)}s
                    </span>
                  </div>
                  
                  <div className="perf-bar-row">
                    <span className="perf-bar-label">Decode Speed</span>
                    <div className="perf-bar-fill-bg">
                      <div 
                        className={`perf-bar-fill ${metrics.tokensSec > 10 ? 'highlighted' : ''}`}
                        style={{ width: `${(metrics.tokensSec / 16) * 100}%` }}
                      ></div>
                    </div>
                    <span className="perf-bar-value">{metrics.tokensSec.toFixed(2)} T/s</span>
                  </div>

                  <div className="perf-bar-row">
                    <span className="perf-bar-label">Prefill Phase</span>
                    <div className="perf-bar-fill-bg">
                      <div 
                        className="perf-bar-fill" 
                        style={{ width: `${Math.max(10, (1 - metrics.prefillTime / 3) * 100)}%` }}
                      ></div>
                    </div>
                    <span className="perf-bar-value">{metrics.prefillTime.toFixed(3)}s</span>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', fontSize: '0.75rem', color: 'var(--muted)', fontFamily: 'var(--font-mono)', marginTop: '8px' }}>
                    <span>Tokens In: {metrics.promptTokens}</span>
                    <span>Tokens Out: {metrics.completionTokens}</span>
                    {metrics.reusedTokens > 0 && (
                      <span style={{ color: '#22c55e' }}>KV-Reuse Matches: {metrics.reusedTokens} tokens</span>
                    )}
                  </div>
                </div>
              )}

              {/* Predefined Prompts at the bottom removed */}

              {/* QuickLM is fully free and open source, no constraints are active. */}
            </div>

          </div>
        </div>

        {/* EXECUTION TRACE TAB */}
        <div className={`view ${activeTab === "trace" ? "active" : ""}`}>
          <div className="visual-trace-container">
            <div className="section-label">C++ Compiler Execution Logs</div>
            <div>
              <div className="terminal-header">
                <span>quicklm_interactive.log</span>
                <span>Active Threads: {threads}</span>
              </div>
              <div className="terminal-body">
                {terminalLogs.length === 0 ? (
                  <div style={{ color: 'var(--muted)', fontStyle: 'italic' }}>
                    No operations run in this session yet. Execute an inference query to trace logs.
                  </div>
                ) : (
                  terminalLogs.map((log, idx) => {
                    let logClass = "cmd";
                    if (log.includes("MATCH FOUND") || log.includes("complete()")) logClass = "success";
                    if (log.includes("Initializing") || log.includes("decode::step")) logClass = "info";
                    return (
                      <div key={idx} className={`terminal-line ${logClass}`}>
                        <span>&gt;</span>
                        <span>{log}</span>
                      </div>
                    );
                  })
                )}
                <div ref={terminalBottomRef} />
              </div>
            </div>
            
            <div className="feature-box">
              <h3 style={{ fontSize: '0.85rem' }}>Understanding the runtime log trace</h3>
              <p style={{ fontSize: '0.78rem' }}>
                The trace logs show standard outputs mapped from Quick Inference (QI) v1 execution files. You can notice how the C++ main driver processes input prompt tokens through <code>forward_batch</code> in a single pass, matches overlapping memory tokens if KV-cache reuse is enabled, and delegates tasks to the hardware thread pool.
              </p>
            </div>
          </div>
        </div>

        {/* Open Source Tab Removed */}

        {/* FAQ TAB */}
        <div className={`view ${activeTab === "faq" ? "active" : ""}`}>
          <div className="section-label">Frequently Asked Questions</div>

          <div className="faq-list">
            {/* FAQ Item 1 */}
            <div className={`faq-item ${faqOpen[0] ? "open" : ""}`}>
              <button className="faq-trigger" onClick={() => setFaqOpen(prev => ({ ...prev, 0: !prev[0] }))}>
                <span className="faq-question">What guarantees zero external dependencies?</span>
                <span className="faq-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </span>
              </button>
              <div className="faq-content">
                <p className="faq-answer">
                  QuickLM uses standard C++17 library elements only. The math operations, tokenizers, safetensors loader, and Jinja chat templates are written entirely from scratch without wrapping BLAS, PyTorch, or ONNX libraries. This prevents dll bloat and compilation issues across operating systems.
                </p>
              </div>
            </div>

            {/* FAQ Item 2 */}
            <div className={`faq-item ${faqOpen[1] ? "open" : ""}`}>
              <button className="faq-trigger" onClick={() => setFaqOpen(prev => ({ ...prev, 1: !prev[1] }))}>
                <span className="faq-question">How does performance scale with CPU core count?</span>
                <span className="faq-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </span>
              </button>
              <div className="faq-content">
                <p className="faq-answer">
                  Our custom thread pool divides matrix multiplications into cache-friendly chunks. On consumer CPUs, performance scales linearly up to physical core count (typically 8-12 threads). Scaling past physical core limits into hyperthreads creates L1/L2 cache friction and can slightly reduce matrix compute speeds.
                </p>
              </div>
            </div>

            {/* FAQ Item 3 */}
            <div className={`faq-item ${faqOpen[2] ? "open" : ""}`}>
              <button className="faq-trigger" onClick={() => setFaqOpen(prev => ({ ...prev, 2: !prev[2] }))}>
                <span className="faq-question">Can I load other Qwen sizes or other model families?</span>
                <span className="faq-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </span>
              </button>
              <div className="faq-content">
                <p className="faq-answer">
                  Yes. Quick Inference v1 includes a pluggable architecture interface (`IArchitecture`). The Qwen 3.5 registry is configured out-of-the-box, supporting all model sizes (0.8B, 1.5B, 7B, etc.). You can implement the base headers to map and load LLaMA or standard attention configs easily.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer copyright */}
      <footer>
        <div>&copy; 2026 QuickLM by TreeSoft.</div>
        <div>
          Support: <a href="mailto:contact@treesoft.pro">contact@treesoft.pro</a>
        </div>
      </footer>

      {/* Global Toast Alert */}
      <div className={`toast-msg ${showToast ? 'active' : ''}`}>
        {toastMsg}
      </div>
    </div>
  );
}
