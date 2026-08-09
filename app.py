import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EmotionX - Face Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to hide standard Streamlit headers/footers and margins to make the custom web app look native
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            border: none;
            width: 100%;
            height: 98vh;
        }
    </style>
""", unsafe_allow_html=True)

# The full HTML / CSS / JS code for the browser-based face emotion detector
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmotionX</title>
    <!-- Google Fonts for premium typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- FontAwesome for modern icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                    },
                }
            }
        }
    </script>
    <!-- face-api.js from vladmandic package CDN -->
    <script src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.js"></script>
    <style>
        body {
            background-color: #080a16;
            color: #f3f4f6;
            font-family: 'Plus Jakarta Sans', sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }

        /* Ambient neon glow backgrounds */
        .glow-1 {
            position: absolute;
            top: -10%;
            left: -10%;
            width: 40vw;
            height: 40vw;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, rgba(6, 182, 212, 0) 70%);
            filter: blur(100px);
            pointer-events: none;
            z-index: 1;
            animation: float-slow 18s infinite alternate;
        }

        .glow-2 {
            position: absolute;
            bottom: -10%;
            right: -10%;
            width: 45vw;
            height: 45vw;
            background: radial-gradient(circle, rgba(236, 72, 153, 0.12) 0%, rgba(236, 72, 153, 0) 70%);
            filter: blur(100px);
            pointer-events: none;
            z-index: 1;
            animation: float-slow 22s infinite alternate-reverse;
        }

        @keyframes float-slow {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 20px) scale(1.05); }
        }

        /* Premium Neo-Glassmorphism effects */
        .glass {
            background: rgba(13, 17, 34, 0.55);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        /* Customize Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.01);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 99px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* 3D Floating Tilt Cards */
        .tilt-card {
            transform-style: preserve-3d;
            transition: transform 0.25s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.25s cubic-bezier(0.25, 1, 0.5, 1);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            will-change: transform;
        }

        .pop-out {
            transform: translateZ(40px);
            transform-style: preserve-3d;
        }
        
        .pop-out-lg {
            transform: translateZ(65px);
            transform-style: preserve-3d;
        }
    </style>
</head>
<body class="relative min-h-screen w-full flex items-center justify-center p-3 sm:p-6 select-none">
    
    <!-- Ambient backdrops -->
    <div class="glow-1"></div>
    <div class="glow-2"></div>

    <div class="w-full max-w-6xl h-[95vh] flex flex-col justify-between relative z-10 glass p-5 sm:p-7 rounded-3xl overflow-hidden">
        
        <!-- Header Section -->
        <header class="flex flex-col sm:flex-row items-center justify-between gap-4 pb-4 border-b border-white/5">
            <div class="flex items-center gap-3.5">
                <div class="w-11 h-11 rounded-2xl bg-gradient-to-tr from-cyan-400 via-blue-500 to-pink-500 flex items-center justify-center shadow-lg shadow-cyan-500/25">
                    <i class="fa-solid fa-face-smile-wink text-white text-xl"></i>
                </div>
                <div class="text-center sm:text-left">
                    <h1 class="text-2xl font-black tracking-tight bg-gradient-to-r from-cyan-400 via-indigo-200 to-pink-400 bg-clip-text text-transparent">EMOTIONX</h1>
                    <p class="text-[10px] text-cyan-400/80 uppercase font-bold tracking-widest mt-0.5"><i class="fa-solid fa-bolt mr-1"></i>Real-Time Emotion Engine</p>
                </div>
            </div>
            
            <!-- Real-time status indicator -->
            <div class="flex items-center gap-3 bg-white/5 py-2 px-4 rounded-xl border border-white/5 text-xs">
                <span id="status-dot" class="w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse"></span>
                <span id="status-message" class="text-gray-300 font-medium">Loading AI engine...</span>
            </div>
        </header>

        <!-- Main Display Panel -->
        <main class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch flex-grow my-5 min-h-0">
            
            <!-- Left Side: Source Video / Upload (Glass Card) -->
            <section class="lg:col-span-7 flex flex-col justify-between glass-card p-4 rounded-2xl border border-white/5 overflow-hidden tilt-card">
                
                <!-- Tab Switching Panel -->
                <div class="flex gap-2 p-1 bg-black/30 rounded-xl border border-white/5 mb-4">
                    <button id="webcam-tab" onclick="switchMode('webcam')" disabled class="flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all hover:bg-white/5 text-gray-500 border border-transparent">
                        <i class="fa-solid fa-video mr-2"></i>Live Webcam
                    </button>
                    <button id="upload-tab" onclick="switchMode('upload')" disabled class="flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all hover:bg-white/5 text-gray-500 border border-transparent">
                        <i class="fa-solid fa-image mr-2"></i>Image Upload
                    </button>
                </div>

                <!-- Center Box containing camera / upload visualizers -->
                <div class="flex-grow flex items-center justify-center bg-black/40 rounded-xl border border-white/5 relative overflow-hidden min-h-0">
                    
                    <!-- 1. Video Container -->
                    <div id="webcam-container" class="w-full h-full flex items-center justify-center relative overflow-hidden">
                        <video id="webcam-video" autoplay muted playsinline class="w-full h-full object-cover rounded-xl scale-x-[-1]"></video>
                        <canvas id="overlay-canvas" class="absolute top-0 left-0 w-full h-full pointer-events-none scale-x-[-1]"></canvas>
                    </div>
                    
                    <!-- 2. Drag & Drop Upload Container -->
                    <div id="upload-container" class="hidden w-full h-full flex flex-col items-center justify-center p-4">
                        <div id="dropzone" class="w-full h-full border-2 border-dashed border-white/10 hover:border-pink-500/50 hover:bg-pink-500/5 rounded-xl flex flex-col items-center justify-center p-6 transition-all cursor-pointer relative overflow-hidden">
                            <div id="upload-prompt" class="flex flex-col items-center gap-3">
                                <div class="w-14 h-14 rounded-full bg-pink-500/10 flex items-center justify-center text-pink-400 text-xl shadow-inner">
                                    <i class="fa-solid fa-cloud-arrow-up"></i>
                                </div>
                                <div>
                                    <p class="font-bold text-sm text-white">Drag and Drop Image Here</p>
                                    <p class="text-[10px] text-gray-400 mt-1">PNG, JPG, JPEG up to 5MB</p>
                                </div>
                                <button class="mt-1 py-2 px-4 bg-gradient-to-r from-pink-500 to-rose-500 hover:scale-105 active:scale-95 text-white rounded-lg text-[10px] font-bold shadow-lg shadow-pink-500/20 transition-all">Browse Local Files</button>
                            </div>
                            <input type="file" id="image-upload" accept="image/*" class="hidden">
                            
                            <img id="preview-image" class="hidden max-h-[380px] max-w-full object-contain rounded-lg">
                            <canvas id="image-canvas" class="absolute top-0 left-0 w-full h-full pointer-events-none"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Bottom controls bar for Webcam -->
                <div id="webcam-controls" class="flex justify-between items-center mt-4">
                    <button id="toggle-camera-btn" onclick="toggleCamera()" class="py-2 px-5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:scale-105 active:scale-95 text-white rounded-xl text-xs font-bold shadow-lg shadow-cyan-500/20 transition-all flex items-center gap-2">
                        <i class="fa-solid fa-camera"></i><span id="btn-camera-text">Turn Camera Off</span>
                    </button>
                    <div class="flex items-center gap-2 text-[9px] text-gray-400 uppercase tracking-widest font-extrabold">
                        <span class="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>Live Analysis Running
                    </div>
                </div>
                
                <!-- Bottom controls bar for Image Upload -->
                <div id="upload-controls" class="hidden flex justify-between items-center mt-4">
                    <button onclick="clearImage()" class="py-2 px-5 bg-white/5 hover:bg-white/10 hover:border-white/10 text-gray-300 rounded-xl text-xs font-bold border border-white/5 transition-all">
                        <i class="fa-solid fa-rotate-left mr-2"></i>Reset Upload
                    </button>
                    <div class="text-[9px] text-gray-400 uppercase tracking-widest font-extrabold">
                        Static Mode
                    </div>
                </div>

            </section>
            
            <!-- Right Side: Stats & breakdown graph (Glass Card) -->
            <section class="lg:col-span-5 flex flex-col gap-5 overflow-hidden">
                
                <!-- 1. Highlight Dominant Emotion Panel -->
                <div class="glass-card p-5 rounded-2xl border border-white/5 flex flex-col items-center justify-center text-center relative overflow-hidden tilt-card">
                    <div class="absolute -top-16 -right-16 w-32 h-32 rounded-full bg-pink-500/5 blur-2xl pointer-events-none"></div>
                    <span class="text-[9px] text-cyan-400 uppercase tracking-widest font-extrabold mb-2.5">Spotlight emotion</span>
                    
                    <div id="dominant-emoji" class="text-6xl mb-3 select-none filter drop-shadow-[0_0_15px_rgba(255,255,255,0.15)] transition-all duration-300 transform hover:scale-105 pop-out-lg">🎭</div>
                    <h2 id="dominant-name" class="text-3xl font-black tracking-tight text-gray-500 transition-all duration-300 pop-out">NO FACE</h2>
                    <p id="dominant-tag" class="text-[10px] text-gray-400 font-medium mt-1 pop-out">Please stand in front of the camera</p>
                </div>
                
                <!-- 2. Breakdown graph list container -->
                <div class="glass-card p-5 rounded-2xl border border-white/5 flex-grow flex flex-col justify-between min-h-0 overflow-y-auto tilt-card">
                    <h3 class="text-xs font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2 border-b border-white/5 pb-2">
                        <i class="fa-solid fa-chart-bar text-pink-400"></i>Probability Metrics
                    </h3>
                    
                    <div class="flex flex-col gap-3 flex-grow justify-center">
                        <!-- Neutral -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Neutral</span>
                                <span id="score-neutral" class="text-teal-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-neutral" class="h-full bg-teal-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#14b8a6]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Happy -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Happy</span>
                                <span id="score-happy" class="text-yellow-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-happy" class="h-full bg-yellow-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#eab308]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Sad -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Sad</span>
                                <span id="score-sad" class="text-blue-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-sad" class="h-full bg-blue-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#3b82f6]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Angry -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Angry</span>
                                <span id="score-angry" class="text-red-500 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-angry" class="h-full bg-red-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#ef4444]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Surprised -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Surprised</span>
                                <span id="score-surprised" class="text-pink-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-surprised" class="h-full bg-pink-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#ec4899]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Fearful -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Fearful</span>
                                <span id="score-fearful" class="text-purple-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-fearful" class="h-full bg-purple-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#a855f7]" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <!-- Disgusted -->
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between items-center text-[10px] font-bold">
                                <span class="text-gray-300">Disgusted</span>
                                <span id="score-disgusted" class="text-orange-400 font-mono">0%</span>
                            </div>
                            <div class="w-full h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div id="bar-disgusted" class="h-full bg-orange-500 rounded-full transition-all duration-300 shadow-[0_0_8px_#f97316]" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>

            </section>
        </main>
        
        <!-- App footer info -->
        <footer class="flex justify-between items-center text-[9px] text-gray-500 border-t border-white/5 pt-3">
            <p><i class="fa-brands fa-chrome mr-1"></i>Processing entirely in client browser</p>
            <p>© 2026 EmotionX • Designed by Antigravity</p>
        </footer>
    </div>

    <!-- Application Script Logic -->
    <script>
        const statusMessage = document.getElementById('status-message');
        const statusDot = document.getElementById('status-dot');
        const webcamTab = document.getElementById('webcam-tab');
        const uploadTab = document.getElementById('upload-tab');
        
        const video = document.getElementById('webcam-video');
        const overlayCanvas = document.getElementById('overlay-canvas');
        const webcamContainer = document.getElementById('webcam-container');
        
        const uploadContainer = document.getElementById('upload-container');
        const dropzone = document.getElementById('dropzone');
        const imageInput = document.getElementById('image-upload');
        const previewImage = document.getElementById('preview-image');
        const imageCanvas = document.getElementById('image-canvas');
        const uploadPrompt = document.getElementById('upload-prompt');
        
        let currentMode = 'webcam';
        let activeStream = null;
        let detectionInterval = null;
        let isCameraOn = true;

        // Initialize and load models
        async function init() {
            try {
                statusMessage.innerText = "Loading AI models from CDN...";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse";
                
                // CDN path for @vladmandic/face-api model weights
                const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
                
                // Load face detection and expression weights
                await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
                
                statusMessage.innerText = "AI models active! Ready.";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-green-500";
                
                // Enable mode selectors
                webcamTab.removeAttribute('disabled');
                uploadTab.removeAttribute('disabled');
                
                // Initialize 3D hover tilt listeners
                initTiltEffect();
                
                // Default mode start
                switchMode('webcam');
            } catch (err) {
                console.error("AI Initialization crash:", err);
                statusMessage.innerText = "Connection lost. Please reload the page.";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-red-500";
            }
        }

        // Tab Switching function
        async function switchMode(mode) {
            currentMode = mode;
            
            const webcamControls = document.getElementById('webcam-controls');
            const uploadControls = document.getElementById('upload-controls');
            
            if (mode === 'webcam') {
                webcamTab.className = "flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all glass text-cyan-400 border-cyan-500/30 shadow-lg shadow-cyan-500/10";
                uploadTab.className = "flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all hover:bg-white/5 text-gray-400 border border-transparent";
                
                webcamContainer.classList.remove('hidden');
                webcamControls.classList.remove('hidden');
                uploadContainer.classList.add('hidden');
                uploadControls.classList.add('hidden');
                
                isCameraOn = true;
                const toggleBtn = document.getElementById('toggle-camera-btn');
                toggleBtn.innerHTML = '<i class="fa-solid fa-camera mr-2"></i>Turn Camera Off';
                toggleBtn.className = "py-2 px-5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:scale-105 active:scale-95 text-white rounded-xl text-xs font-bold shadow-lg shadow-cyan-500/20 transition-all flex items-center gap-2";
                
                await startWebcam();
            } else {
                uploadTab.className = "flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all glass text-pink-400 border-pink-500/30 shadow-lg shadow-pink-500/10";
                webcamTab.className = "flex-1 py-2.5 px-4 rounded-lg font-bold text-xs transition-all hover:bg-white/5 text-gray-400 border border-transparent";
                
                uploadContainer.classList.remove('hidden');
                uploadControls.classList.remove('hidden');
                webcamContainer.classList.add('hidden');
                webcamControls.classList.add('hidden');
                
                stopWebcam();
                clearImage();
            }
        }

        // Access client camera
        async function startWebcam() {
            if (activeStream) return;
            try {
                statusMessage.innerText = "Connecting camera stream...";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse";
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: "user" }
                });
                
                video.srcObject = stream;
                activeStream = stream;
                
                video.onloadedmetadata = () => {
                    video.play();
                    statusMessage.innerText = "Analyzing live stream feed...";
                    statusDot.className = "w-2.5 h-2.5 rounded-full bg-green-500 animate-ping";
                    
                    const displaySize = { width: video.clientWidth, height: video.clientHeight };
                    overlayCanvas.width = displaySize.width;
                    overlayCanvas.height = displaySize.height;
                    faceapi.matchDimensions(overlayCanvas, displaySize);
                    
                    startDetectionLoop(video, overlayCanvas, displaySize);
                };
            } catch (err) {
                console.error("Camera fail:", err);
                statusMessage.innerText = "Camera access denied. Please allow permissions.";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-red-500";
            }
        }

        // Halt client camera
        function stopWebcam() {
            if (activeStream) {
                activeStream.getTracks().forEach(track => track.stop());
                activeStream = null;
            }
            video.srcObject = null;
            if (detectionInterval) {
                clearInterval(detectionInterval);
                detectionInterval = null;
            }
            const ctx = overlayCanvas.getContext('2d');
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            clearEmotionStats();
        }

        // Toggle button logic
        async function toggleCamera() {
            const toggleBtn = document.getElementById('toggle-camera-btn');
            if (isCameraOn) {
                stopWebcam();
                isCameraOn = false;
                toggleBtn.innerHTML = '<i class="fa-solid fa-power-off mr-2"></i>Turn Camera On';
                toggleBtn.className = "py-2 px-5 bg-gradient-to-r from-green-500 to-emerald-600 hover:scale-105 active:scale-95 text-white rounded-xl text-xs font-bold shadow-lg shadow-green-500/20 transition-all flex items-center gap-2";
                statusMessage.innerText = "Camera disabled.";
                statusDot.className = "w-2.5 h-2.5 rounded-full bg-gray-500";
            } else {
                isCameraOn = true;
                toggleBtn.innerHTML = '<i class="fa-solid fa-camera mr-2"></i>Turn Camera Off';
                toggleBtn.className = "py-2 px-5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:scale-105 active:scale-95 text-white rounded-xl text-xs font-bold shadow-lg shadow-cyan-500/20 transition-all flex items-center gap-2";
                await startWebcam();
            }
        }

        // Webcam Detection Loop
        function startDetectionLoop(videoElement, canvasElement, displaySize) {
            if (detectionInterval) clearInterval(detectionInterval);
            
            detectionInterval = setInterval(async () => {
                if (videoElement.paused || videoElement.ended || currentMode !== 'webcam') return;
                
                // Get scaled client size dynamically to handle window scaling
                const currentSize = { width: videoElement.clientWidth, height: videoElement.clientHeight };
                if (canvasElement.width !== currentSize.width || canvasElement.height !== currentSize.height) {
                    canvasElement.width = currentSize.width;
                    canvasElement.height = currentSize.height;
                    faceapi.matchDimensions(canvasElement, currentSize);
                }
                
                const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
                const detections = await faceapi.detectAllFaces(videoElement, options)
                    .withFaceLandmarks()
                    .withFaceExpressions();
                
                const ctx = canvasElement.getContext('2d');
                ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                
                const resizedDetections = faceapi.resizeResults(detections, currentSize);
                
                // Drawing sci-fi overlays
                drawPremiumBoxes(ctx, resizedDetections);
                
                if (resizedDetections.length > 0) {
                    updateEmotionStats(resizedDetections[0].expressions);
                } else {
                    clearEmotionStats();
                }
            }, 130); // Efficient intervals to keep frame-rates high and CPU loads minimal
        }

        // Image Drag-and-drop / Upload inputs
        dropzone.addEventListener('click', () => {
            if (currentMode === 'upload') imageInput.click();
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (currentMode !== 'upload') return;
            dropzone.classList.add('border-pink-500', 'bg-pink-500/5');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('border-pink-500', 'bg-pink-500/5');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-pink-500', 'bg-pink-500/5');
            if (currentMode !== 'upload') return;
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                loadUploadedFile(file);
            }
        });

        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) loadUploadedFile(file);
        });

        function loadUploadedFile(file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                previewImage.src = event.target.result;
                previewImage.classList.remove('hidden');
                uploadPrompt.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }

        // Trigger processing when image uploads and draws in client viewport
        previewImage.onload = async () => {
            if (currentMode !== 'upload') return;
            
            statusMessage.innerText = "Analyzing static image frame...";
            statusDot.className = "w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse";
            
            // Allow client browser thread layout rendering before matching canvas dimensions
            setTimeout(async () => {
                const displaySize = { width: previewImage.clientWidth, height: previewImage.clientHeight };
                imageCanvas.width = displaySize.width;
                imageCanvas.height = displaySize.height;
                imageCanvas.classList.remove('hidden');
                faceapi.matchDimensions(imageCanvas, displaySize);
                
                const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
                const detections = await faceapi.detectAllFaces(previewImage, options)
                    .withFaceLandmarks()
                    .withFaceExpressions();
                
                const ctx = imageCanvas.getContext('2d');
                ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                ctx.drawImage(previewImage, 0, 0, imageCanvas.width, imageCanvas.height);
                
                const resizedDetections = faceapi.resizeResults(detections, displaySize);
                drawPremiumBoxes(ctx, resizedDetections);
                
                if (resizedDetections.length > 0) {
                    statusMessage.innerText = `Analysis complete! Detected ${resizedDetections.length} face(s).`;
                    statusDot.className = "w-2.5 h-2.5 rounded-full bg-green-500";
                    updateEmotionStats(resizedDetections[0].expressions);
                } else {
                    statusMessage.innerText = "No faces found in this image. Try another image.";
                    statusDot.className = "w-2.5 h-2.5 rounded-full bg-red-500";
                    clearEmotionStats();
                }
            }, 150);
        };

        // Reset Static Upload controls
        function clearImage() {
            imageInput.value = "";
            previewImage.src = "";
            previewImage.classList.add('hidden');
            imageCanvas.classList.add('hidden');
            uploadPrompt.classList.remove('hidden');
            
            const ctx = imageCanvas.getContext('2d');
            ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
            
            clearEmotionStats();
            statusMessage.innerText = "Upload an image file to analyze emotions.";
            statusDot.className = "w-2.5 h-2.5 rounded-full bg-gray-500";
        }

        // Draw premium glowing HUD overlays over tracked faces
        function drawPremiumBoxes(ctx, detections) {
            detections.forEach(det => {
                const { x, y, width, height } = det.detection.box;
                
                // Set custom color scheme for drawing HUD (Neon Cyan)
                ctx.strokeStyle = '#06b6d4';
                ctx.lineWidth = 3.5;
                const len = Math.min(width, height) * 0.18; // corner length relative to box
                
                // Draw HUD corners
                // Top-Left
                ctx.beginPath(); ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.stroke();
                // Top-Right
                ctx.beginPath(); ctx.moveTo(x + width - len, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + len); ctx.stroke();
                // Bottom-Left
                ctx.beginPath(); ctx.moveTo(x, y + height - len); ctx.lineTo(x, y + height); ctx.lineTo(x + len, y + height); ctx.stroke();
                // Bottom-Right
                ctx.beginPath(); ctx.moveTo(x + width - len, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - len); ctx.stroke();
                
                // Subtle overlay shading
                ctx.fillStyle = 'rgba(6, 182, 212, 0.04)';
                ctx.fillRect(x, y, width, height);
                
                // Draw landmarks (glowing Magenta dots)
                ctx.fillStyle = 'rgba(236, 72, 153, 0.85)';
                const points = det.landmarks.positions;
                points.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
                    ctx.fill();
                });
                
                // Calculate Dominant Expression
                let maxEmotion = 'neutral';
                let maxVal = 0;
                Object.entries(det.expressions).forEach(([emotion, val]) => {
                    if (val > maxVal) {
                        maxVal = val;
                        maxEmotion = emotion;
                    }
                });
                
                // Draw floating tag HUD label
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 13px "Plus Jakarta Sans", sans-serif';
                ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
                ctx.shadowBlur = 4;
                ctx.fillText(`${maxEmotion.toUpperCase()} (${Math.round(maxVal * 100)}%)`, x + 5, y - 8);
                ctx.shadowBlur = 0; // reset shadow
            });
        }

        // Live stats mapping metrics
        const emojis = { neutral: '😐', happy: '😊', sad: '😢', angry: '😠', surprised: '😲', fearful: '😨', disgusted: '🤢' };
        const tags = {
            neutral: 'Calm & Collected',
            happy: 'Cheerful & Bright',
            sad: 'Somber & Low',
            angry: 'Tense & Heated',
            surprised: 'Astonished & Alert',
            fearful: 'Anxious & Concerned',
            disgusted: 'Averse & Repulsed'
        };
        const colors = {
            neutral: 'text-teal-400',
            happy: 'text-yellow-400',
            sad: 'text-blue-400',
            angry: 'text-red-400',
            surprised: 'text-pink-400',
            fearful: 'text-purple-400',
            disgusted: 'text-orange-400'
        };

        function updateEmotionStats(expressions) {
            let dominant = 'neutral';
            let maxScore = 0;
            
            Object.entries(expressions).forEach(([emotion, val]) => {
                const percent = Math.round(val * 100);
                
                // Update specific percentage text values
                const scoreLabel = document.getElementById(`score-${emotion}`);
                if (scoreLabel) scoreLabel.innerText = `${percent}%`;
                
                // Update graph bar widths
                const graphBar = document.getElementById(`bar-${emotion}`);
                if (graphBar) graphBar.style.width = `${percent}%`;
                
                if (val > maxScore) {
                    maxScore = val;
                    dominant = emotion;
                }
            });
            
            // Set dominant panels
            const emoIcon = document.getElementById('dominant-emoji');
            const emoName = document.getElementById('dominant-name');
            const emoTag = document.getElementById('dominant-tag');
            
            emoIcon.innerText = emojis[dominant];
            emoName.innerText = dominant.toUpperCase();
            emoName.className = `text-3xl font-black tracking-tight transition-all duration-300 ${colors[dominant]}`;
            emoTag.innerText = tags[dominant];
        }

        function clearEmotionStats() {
            const emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted'];
            emotions.forEach(emo => {
                const scoreLabel = document.getElementById(`score-${emo}`);
                if (scoreLabel) scoreLabel.innerText = `0%`;
                
                const graphBar = document.getElementById(`bar-${emo}`);
                if (graphBar) graphBar.style.width = `0%`;
            });
            
            const emoIcon = document.getElementById('dominant-emoji');
            const emoName = document.getElementById('dominant-name');
            const emoTag = document.getElementById('dominant-tag');
            
            emoIcon.innerText = '🎭';
            emoName.innerText = 'NO FACE';
            emoName.className = 'text-3xl font-black tracking-tight text-gray-500';
            emoTag.innerText = 'Please stand in front of the camera';
        }

        // 3D Tilt Hover effect logic
        function initTiltEffect() {
            const cards = document.querySelectorAll('.tilt-card');
            cards.forEach(card => {
                card.addEventListener('mousemove', (e) => {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const xc = rect.width / 2;
                    const yc = rect.height / 2;
                    const dx = x - xc;
                    const dy = y - yc;
                    
                    // Rotate maximum of 8 degrees for natural tilt
                    const rx = -(dy / yc) * 8;
                    const ry = (dx / xc) * 8;
                    
                    card.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
                    card.style.boxShadow = `${-ry * 1.5}px ${rx * 1.5}px 35px rgba(6, 182, 212, 0.12), 0 15px 35px rgba(0, 0, 0, 0.4)`;
                });
                
                card.addEventListener('mouseleave', () => {
                    card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)';
                    card.style.boxShadow = '0 15px 35px rgba(0, 0, 0, 0.4)';
                    card.style.transition = 'transform 0.5s ease, box-shadow 0.5s ease';
                });
                
                card.addEventListener('mouseenter', () => {
                    card.style.transition = 'transform 0.05s ease, box-shadow 0.05s ease';
                });
            });
        }

        // Fire init call
        init();
    </script>
</body>
</html>
"""

components.html(html_code, height=900, scrolling=False)