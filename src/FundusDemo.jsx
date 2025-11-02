// src/FundusDemo.jsx
import React, { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";
const modelUrl = `${import.meta.env.BASE_URL}mlp_student.onnx`;
// Pin to version 1.20.0 to avoid SessionOptions constructor issue
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
// Disable JSEP and threads for compatibility
ort.env.wasm.numThreads = 1;
ort.env.allowJSEPSupport = false;

const CLASSES = ["Normal", "Glaucoma", "Myopia", "Diabetes"];
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];
const IMG_SIZE = 224;

// Enhanced model loader with progress tracking
function useModelLoader() {
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadModel() {
            console.log(`[ModelLoader] Starting model load from: ${modelUrl}`);
            setLoading(true);
            setProgress(0);
            setError(null);
            try {
                // Use XMLHttpRequest for progress tracking
                const xhr = new XMLHttpRequest();
                xhr.open('GET', modelUrl, true);
                xhr.responseType = 'arraybuffer';

                xhr.addEventListener('progress', (event) => {
                    if (event.lengthComputable) {
                        const percentComplete = (event.loaded / event.total) * 100;
                        setProgress(Math.round(percentComplete));
                        console.log(`[ModelLoader] Download progress: ${percentComplete.toFixed(1)}% (${event.loaded}/${event.total} bytes)`);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        console.log(`[ModelLoader] Model fetched successfully, size: ${xhr.response.byteLength} bytes`);
                        const modelBuffer = xhr.response;
                        
                        // Create session without SessionOptions constructor
                        const sessionOptions = {
                            executionProviders: ['wasm']
                        };

                        console.log(`[ModelLoader] Creating ONNX session with WASM provider...`);
                        ort.InferenceSession.create(modelBuffer, sessionOptions).then((sess) => {
                            console.log(`[ModelLoader] ONNX session created successfully. Input names:`, sess.inputNames);
                            console.log(`[ModelLoader] Output names:`, sess.outputNames);
                            setSession(sess);
                            setProgress(100); // Complete
                            setLoading(false);
                        }).catch((err) => {
                            console.error(`[ModelLoader] Session creation failed:`, err);
                            setError(err);
                            setLoading(false);
                        });
                    } else {
                        const err = new Error(`HTTP ${xhr.status} ${xhr.statusText}`);
                        console.error(`[ModelLoader] Fetch error:`, err);
                        setError(err);
                        setLoading(false);
                    }
                });

                xhr.addEventListener('error', (err) => {
                    console.error(`[ModelLoader] Network error:`, err);
                    setError(err);
                    setLoading(false);
                });

                xhr.send();
            } catch (err) {
                console.error(`[ModelLoader] Unexpected error:`, err);
                setError(err);
                setLoading(false);
            }
        }
        loadModel();
    }, []);

    return { session, loading, progress, error };
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
                const chunkSize = 32; // You can try 16 if 32 is still too much

                // Define the processing function that will be called recursively via rAF
                const processChunk = () => {
                    const endRow = Math.min(currentRow + chunkSize, IMG_SIZE);
                    
                    // The synchronous processing for this chunk
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
                        // Schedule the next chunk. This clears the call stack.
                        requestAnimationFrame(processChunk);
                    } else {
                        // All chunks are processed, create the tensor and resolve
                        const tensor = new ort.Tensor('float32', tensorData, [1, 3, IMG_SIZE, IMG_SIZE]);
                        console.log(`[Preprocess] Tensor ready: shape [1,3,${IMG_SIZE},${IMG_SIZE}]`);
                        resolve(tensor);
                    }
                };

                // Start the processing
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

// Compute softmax
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
    const { session, loading: modelLoading, progress, error: modelError } = useModelLoader();
    const [showDemo, setShowDemo] = useState(false);
    const [testSplit, setTestSplit] = useState([]);
    const [testSplitLoading, setTestSplitLoading] = useState(true);
    const [testSplitError, setTestSplitError] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [gtLabel, setGtLabel] = useState(null);
    const [results, setResults] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);

    // Fetch test split data (only original_path and class_label_remapped)
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
                // Filter to only needed fields
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
            // Preprocess image to tensor
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
                <div className="title-container">
                    <h1>HNN to MLP Distillation</h1>
                    <h2>Fundus Classification Web Demo</h2>
                </div>
                <button className="try-demo-button" onClick={handleTryDemo}>
                    Try Demo
                </button>
            </>
        );
    }

    return (
        <>
            <div className="title-container">
                <h1>HNN to MLP Distillation</h1>
                <h2>Fundus Classification Web Demo</h2>
            </div>
            
            <div className="fundus-demo">
                {/* Model Loading Progress */}
                {modelLoading && (
                    <div className="loading-container">
                        <h3>Downloading Model ({progress}%)</h3>
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

                {testSplitLoading && <p>Loading test split...</p>}
                {testSplitError && (
                    <div className="error">
                        <h3>Test Split Error</h3>
                        <p>{testSplitError}</p>
                    </div>
                )}

                {!modelLoading && !modelError && (
                    <>
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

                        {/* Ground Truth appears right after image is selected */}
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
        </>
    );
}