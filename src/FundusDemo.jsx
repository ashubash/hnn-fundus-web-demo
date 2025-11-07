// src/FundusDemo.jsx
import React, { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import Box from '@mui/material/Box';
import Modal from '@mui/material/Modal';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';

const modelUrl = `${import.meta.env.BASE_URL}light_hgnn_student.onnx`;
const wasmUrl = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort-wasm-simd-threaded.wasm";

// Pin to version 1.20.0 to avoid SessionOptions constructor issue
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
// Disable JSEP and threads for compatibility
ort.env.wasm.numThreads = 1;
ort.env.allowJSEPSupport = false;

const CLASSES = ["Normal", "Glaucoma", "Myopia", "Diabetes"];
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];
const IMG_SIZE = 224;

// Model performance data
const MODEL_DATA = {
  "conformal": {
    "alpha": 0.05,
    "coverage": 0.9855177407675597,
    "avg_set_size": 1.0702389572773352,
    "coverage_error": 0.03551774076755976
  },
  "student": {
    "test_acc": 0.9319333816075308
  }
};

const modalStyle = {
  position:'absolute',
  top:'50%',
  left:'50%',
  transform:'translate(-50%,-50%)',
  width:440,
  bgcolor:'#fff',
  borderRadius:'12px',
  boxShadow:24,
  p:4,
};

// CORRECTED: Model loader with robust byte-based progress tracking
function useModelLoader() {
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [progress, setProgress] = useState(0);
    const [loadingStage, setLoadingStage] = useState("Downloading Model...");
    const [error, setError] = useState(null);

    useEffect(() => {
        let isMounted = true;
        const controller = new AbortController();
        const signal = controller.signal;

        // --- REVISED Helper function to download with progress ---
        // It now reports back chunk sizes and the total file size
        const downloadWithProgress = async (url, onChunk, onTotalSizeKnown) => {
            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
            const contentLength = response.headers.get('Content-Length');
            if (!contentLength) throw new Error('Content-Length header is missing.');
            const totalBytes = parseInt(contentLength, 10);
            
            // NEW: Log model name and size
            const modelName = url.split('/').pop();
            console.log(`[ModelLoader] Fetching model: ${modelName}, size: ${totalBytes} bytes`);
            
            // Report the total size of this specific file to the main loader
            onTotalSizeKnown(totalBytes);

            const reader = response.body.getReader();
            const chunks = [];
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                // Report the size of the chunk just downloaded
                onChunk(value.length);
            }
            
            // Assemble the final buffer
            let totalLength = 0;
            for (const chunk of chunks) totalLength += chunk.length;
            const result = new Uint8Array(totalLength);
            let position = 0;
            for (const chunk of chunks) {
                result.set(chunk, position);
                position += chunk.length;
            }
            return result.buffer;
        };

        async function loadModel() {
            console.log(`[ModelLoader] Starting combined load process.`);
            setLoading(true);
            setProgress(0);
            setError(null);
            
            let totalExpectedBytes = 0;
            let totalDownloadedBytes = 0;

            // This function will be called to update the progress bar
            const updateProgress = () => {
                if (isMounted && totalExpectedBytes > 0) {
                    const combinedProgress = (totalDownloadedBytes / totalExpectedBytes) * 100;
                    setProgress(Math.round(combinedProgress));
                }
            };

            try {
                // --- Step 1: Download the main model ---
                setLoadingStage("Downloading Model...");
                const modelBuffer = await downloadWithProgress(
                    modelUrl, 
                    (chunkBytes) => { // Callback for each chunk
                        totalDownloadedBytes += chunkBytes;
                        updateProgress();
                    },
                    (fileSize) => { // Callback for the total file size
                        totalExpectedBytes += fileSize;
                        updateProgress();
                    }
                );
                console.log(`[ModelLoader] Model downloaded.`);

                // --- Step 2: Initialize the session ---
                setLoadingStage("Creating ONNX session...");
                const sessionOptions = { executionProviders: ['wasm'] };
                const sess = await ort.InferenceSession.create(modelBuffer, sessionOptions);
                
                if (!isMounted) return;
                console.log(`[ModelLoader] ONNX session created successfully.`);
                setSession(sess);
                setProgress(100); // Ensure it ends at 100
                setLoading(false);

            } catch (err) {
                if (err.name === 'AbortError') {
                    console.log(`[ModelLoader] Model load was aborted.`);
                } else {
                    console.error(`[ModelLoader] Error during model load:`, err);
                    if (isMounted) {
                        setError(err);
                        setLoading(false);
                    }
                }
            }
        }

        loadModel();

        return () => {
            isMounted = false;
            controller.abort();
        };
    }, []);

    return { session, loading, progress, loadingStage, error };
}

const preprocessImage = (imageSrc) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = IMG_SIZE;
                canvas.height = IMG_SIZE;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

                const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
                const data = imageData.data;

                const tensorData = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
                let currentRow = 0;
                const chunkSize = 32;

                const processChunk = () => {
                    const endRow = Math.min(currentRow + chunkSize, IMG_SIZE);
                    
                    for (let row = currentRow; row < endRow; row++) {
                        for (let x = 0; x < IMG_SIZE; x++) {
                            const idx = (row * IMG_SIZE + x) * 4;
                            const r = data[idx] / 255.0;
                            const g = data[idx + 1] / 255.0;
                            const b = data[idx + 2] / 255.0;

                            const normR = (r - NORMALIZE_MEAN[0]) / NORMALIZE_STD[0];
                            const normG = (g - NORMALIZE_MEAN[1]) / NORMALIZE_STD[1];
                            const normB = (b - NORMALIZE_MEAN[2]) / NORMALIZE_STD[2];

                            const outIdxR = row * IMG_SIZE + x;
                            const outIdxG = IMG_SIZE * IMG_SIZE + row * IMG_SIZE + x;
                            const outIdxB = 2 * IMG_SIZE * IMG_SIZE + row * IMG_SIZE + x;

                            tensorData[outIdxR] = normR;
                            tensorData[outIdxG] = normG;
                            tensorData[outIdxB] = normB;
                        }
                    }

                    const processedPct = ((endRow / IMG_SIZE) * 100).toFixed(0);
                    console.log(`[Preprocess] Rows ${currentRow}-${endRow} done (${processedPct}% of rows)`);

                    currentRow = endRow;
                    
                    if (currentRow < IMG_SIZE) {
                        requestAnimationFrame(processChunk);
                    } else {
                        const tensor = new ort.Tensor('float32', tensorData, [1, 3, IMG_SIZE, IMG_SIZE]);
                        console.log(`[Preprocess] Tensor ready: shape [1,3,${IMG_SIZE},${IMG_SIZE}]`);
                        resolve(tensor);
                    }
                };

                requestAnimationFrame(processChunk);

            } catch (err) {
                console.error(`[Preprocess] Error processing image:`, err);
                reject(err);
            }
        };
        img.onerror = (err) => {
            console.error(`[Preprocess] Image load failed:`, err);
            reject(new Error(`Failed to load ${imageSrc}`));
        };
        img.src = imageSrc;
    });
};

const computeProbs = (logits) => {
    const output = Array.from(logits);
    console.log(`[Probs] Computing softmax from logits:`, output);
    const probs = new Float32Array(output.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < output.length; i++) maxLogit = Math.max(maxLogit, output[i]);
    let sumExp = 0;
    for (let i = 0; i < output.length; i++) sumExp += Math.exp(output[i] - maxLogit);
    for (let i = 0; i < output.length; i++) probs[i] = Math.exp(output[i] - maxLogit) / sumExp;
    console.log(`[Probs] Softmax output:`, Array.from(probs));
    return Array.from(probs);
};

export default function FundusDemo() {
    const { session, loading: modelLoading, progress, loadingStage, error: modelError } = useModelLoader();
    const [showDemo, setShowDemo] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [testSplit, setTestSplit] = useState([]);
    const [testSplitLoading, setTestSplitLoading] = useState(true);
    const [testSplitError, setTestSplitError] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [gtLabel, setGtLabel] = useState(null);
    const [results, setResults] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);

    useEffect(() => {
        async function fetchTestSplit() {
            console.log(`[TestSplit] Fetching test_split_preproc.json...`);
            try {
                setTestSplitLoading(true);
                setTestSplitError(null);
                const res = await fetch("/test_split_preproc.json");
                if (!res.ok) {
                    const err = new Error(`HTTP ${res.status} ${res.statusText}`);
                    console.error(`[TestSplit] Fetch failed:`, err);
                    throw err;
                }
                const data = await res.json();
                const filteredData = data.map(({ original_path, class_label_remapped }) => ({
                    original_path,
                    class_label_remapped
                }));
                console.log(`[TestSplit] Loaded ${filteredData.length} samples successfully`);
                setTestSplit(filteredData);
            } catch (err) {
                console.error(`[TestSplit] Full error:`, err);
                setTestSplitError(err.message);
            } finally {
                setTestSplitLoading(false);
            }
        }
        fetchTestSplit();
    }, []);

    const handleTryDemo = () => {
        console.log(`[Demo] User clicked Try Demo`);
        setShowDemo(true);
    };

    const handlePickRandom = () => {
        console.log(`[PickRandom] Attempting to pick random sample`);
        if (testSplitLoading || testSplitError || !testSplit.length) {
            console.warn(`[PickRandom] Skipped: loading=${testSplitLoading}, error=${!!testSplitError}, samples=${testSplit.length}`);
            return;
        }

        const sample = testSplit[Math.floor(Math.random() * testSplit.length)];
        console.log(`[PickRandom] Selected: ${sample.original_path}, GT: ${CLASSES[sample.class_label_remapped]}`);
        setSelectedImage(sample.original_path);
        setGtLabel(sample.class_label_remapped);
        setResults([]);
    };

    const handleRunInference = async () => {
        console.log(`[Inference] Starting inference with session: ${!!session}, image: ${selectedImage}`);
        if (!session || !selectedImage) {
            console.warn(`[Inference] Skipped: missing session=${!session}, image=${!selectedImage}`);
            return;
        }

        setIsProcessing(true);
        try {
            const tensor = await preprocessImage(selectedImage);
            
            const start = performance.now();
            const inputName = session.inputNames[0];
            console.log(`[Inference] Feeding tensor to input: ${inputName}, shape: [${tensor.dims.join(', ')}]`);
            const feeds = { [inputName]: tensor };
            const resultsRun = await session.run(feeds);
            const outputName = session.outputNames[0];
            const output = resultsRun[outputName].data;
            console.log(`[Inference] Raw output from ${outputName}:`, output);

            const probs = computeProbs(output);
            let maxProb = 0, predIndex = 0;
            for (let i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) { maxProb = probs[i]; predIndex = i; }
            }
            console.log(`[Inference] Prediction: ${CLASSES[predIndex]} (index ${predIndex}), confidence: ${(maxProb * 100).toFixed(2)}%`);

            const end = performance.now();
            const time = (end - start).toFixed(2);
            console.log(`[Inference] Completed in ${time} ms`);

            setResults([
                { label: "Prediction", value: CLASSES[predIndex] },
                { label: "Confidence", value: `${(maxProb * 100).toFixed(2)}%` },
                { label: "Inference Time", value: `${time} ms` }
            ]);
        } catch (err) {
            console.error(`[Inference] Error during run:`, err);
            setResults([{ label: "Error", value: `Inference failed: ${err.message}` }]);
        } finally {
            setIsProcessing(false);
        }
    };

    if (!showDemo) {
      return (
        <>
          <nav className="navbar">
            <div className="navbar-text">Fundus Classification Model Demo</div>
          </nav>
          <div style={{
            display:'flex',
            flexDirection:'column',
            alignItems:'center',
            justifyContent:'center',
            gap:'32px',
            flex: 1
          }}>
            <svg width="160" height="120" viewBox="0 0 40 30" xmlns="http://www.w3.org/2000/svg">
              <circle cx="20" cy="15" r="10" fill="#4A90E2" stroke="#FFF" strokeWidth="1"/>
              <circle cx="16" cy="12" r="2" fill="#FFF"/>
              <circle cx="24" cy="12" r="2" fill="#FFF"/>
            </svg>

            <button className="try-demo-button" onClick={handleTryDemo}>
              Try Demo
            </button>
          </div>
          <footer className="footer">
            <p>&copy; 2025 Fundus Demo. Built with ONNX Runtime Web.</p>
          </footer>
        </>
      );
    }

    return (
        <>
            <nav className="navbar">
              <div className="navbar-text">Fundus Classification Model Demo</div>
            </nav>
            
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '0px', marginBottom: '0px' }}>
              <button
                onClick={() => setIsModalOpen(true)}
                style={{background:'none', border:'none', cursor:'pointer', padding:0, margin:0}}
                title="Model Details"
              >
                <svg width="160" height="132" viewBox="0 0 40 34" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="15" r="10" fill="#4A90E2" stroke="#FFF" strokeWidth="1"/>
                <circle cx="16" cy="12" r="2" fill="#FFF"/>
                <circle cx="24" cy="12" r="2" fill="#FFF"/>
                <path id="smilePath" d="M 10 15 Q 20 27 30 15" fill="none"/>
                <text font-size="2.1" fill="#FFF" font-family="Arial, sans-serif" font-weight="bold" text-anchor="middle">
                    <textPath href="#smilePath" startOffset="50%" textLength="9" dy="-5">
                    MODEL
                    </textPath>
                </text>
                </svg>
              </button>
            </div>
            
            <div className="fundus-demo" style={{ marginTop: '0px' }}>
                {modelLoading && (
                    <div className="loading-container">
                        <h3>{loadingStage}</h3>
                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                        </div>
                    </div>
                )}

                {modelError && (
                    <div className="error">
                        <h3>Model Load Error</h3>
                        <p>{modelError.message}</p>
                    </div>
                )}

                {testSplitError && (
                    <div className="error">
                        <h3>Test Split Error</h3>
                        <p>{testSplitError}</p>
                    </div>
                )}

                {!modelLoading && !modelError && (
                    <>
                        {testSplitLoading && (
                            <div className="loading-container">
                                <h3>Loading test data...</h3>
                            </div>
                        )}
                        <button onClick={handlePickRandom} disabled={testSplitLoading || !!testSplitError || !testSplit.length}>
                            Pick Random Image
                        </button>

                        <div className="image-container">
                            {selectedImage ? (
                                <img
                                    src={selectedImage}
                                    alt="Fundus"
                                    onError={(e) => {
                                        console.error(`[Image] Load failed for: ${selectedImage}`);
                                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjI0IiBoZWlnaHQ9IjIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzY2NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                                    }}
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Click "Pick Random Image" to start
                                </div>
                            )}
                        </div>

                        {gtLabel !== null && (
                            <div className="result-item">
                                <span className="result-label">Ground Truth:</span> {CLASSES[gtLabel]}
                            </div>
                        )}

                        <button onClick={handleRunInference} disabled={!selectedImage || isProcessing || !session}>
                            {isProcessing ? "Processing..." : "Run Inference"}
                        </button>

                        {results.length > 0 && (
                            <div className="results-container">
                                {results.map((result, index) => (
                                    <div key={index} className="result-item">
                                        <span className="result-label">{result.label}:</span> {result.value}
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>

            <Modal open={isModalOpen} onClose={()=>setIsModalOpen(false)}>
  <Box sx={modalStyle}>
    <IconButton
      aria-label="close"
      onClick={()=>setIsModalOpen(false)}
      sx={{
        position: 'absolute',
        right: 8,
        top: 8,
        color: (theme) => theme.palette.grey[500],
      }}
    >
      <CloseIcon />
    </IconButton>
                    <Typography variant="h6" sx={{mb:3, fontWeight: 'bold', color: '#000'}}>Student Model Information</Typography>
                    <Typography sx={{mb:1, fontWeight: 'bold', color: '#000'}}>Test Accuracy: {Math.round(MODEL_DATA.student.test_acc * 10000) / 100}%</Typography>
                    <Typography sx={{mb:3, color: '#000'}}>
                        This means the simplified model correctly identifies {Math.round(MODEL_DATA.student.test_acc * 10000) / 100}% of eye images on its own, without relying on the more complex original system.
                    </Typography>

                    <Divider sx={{my: 2, borderColor: '#e0e0e0'}} />

                    <Typography variant="subtitle1" sx={{mb:2, fontWeight: 'bold', color: '#000'}}>Conformal Prediction Metrics</Typography>
                    <Typography sx={{mb:1, fontWeight: 'bold', color: '#000'}}>Alpha (Risk Level): {MODEL_DATA.conformal.alpha}</Typography>
                    <Typography sx={{mb:1, color: '#000'}}>This sets the maximum chance of an incorrect prediction at just 5%.</Typography>
                    
                    <Typography sx={{mb:1, fontWeight: 'bold', color: '#000'}}>Coverage: {Math.round(MODEL_DATA.conformal.coverage * 10000) / 100}%</Typography>
                    <Typography sx={{mb:1, color: '#000'}}>This means the true diagnosis is included in our predictions {Math.round(MODEL_DATA.conformal.coverage * 10000) / 100}% of the time, meeting safety requirements.</Typography>
                    
                    <Typography sx={{mb:1, fontWeight: 'bold', color: '#000'}}>Average Set Size: {Math.round(MODEL_DATA.conformal.avg_set_size * 100) / 100}</Typography>
                    <Typography sx={{color: '#000'}}>This means the model typically provides one clear answer (1.07 Classification sets) per image, prescribing high confidence rates.</Typography>
                </Box>
            </Modal>

            <footer className="footer">
                <p>&copy; 2025 Fundus Demo. Built with ONNX Runtime Web.</p>
            </footer>
        </>
    );
}