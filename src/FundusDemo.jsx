// src/FundusDemo.jsx
import React, { useEffect, useState, useRef } from "react";
import * as ort from "onnxruntime-web";
const modelUrl = `${import.meta.env.BASE_URL}student_classifier.onnx`;
// Pin to version 1.20.0 to avoid SessionOptions constructor issue
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
// Disable JSEP and threads for compatibility
ort.env.wasm.numThreads = 1;
ort.env.allowJSEPSupport = false;

const CLASSES = ["Normal", "Glaucoma", "Myopia", "Diabetes"];

// Enhanced model loader with asset imports
function useModelLoader() {
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadModel() {
            console.log(`[ModelLoader] Starting model load from: ${modelUrl}`);
            setLoading(true);
            setError(null);
            try {
                const modelRes = await fetch(modelUrl);
                if (!modelRes.ok) {
                    const err = new Error(`Model fetch failed: ${modelRes.status} ${modelRes.statusText}`);
                    console.error(`[ModelLoader] Fetch error:`, err);
                    throw err;
                }
                console.log(`[ModelLoader] Model fetched successfully, size: ${modelRes.headers.get('content-length')} bytes`);
                const modelBuffer = await modelRes.arrayBuffer();
                console.log(`[ModelLoader] Model buffer ready, length: ${modelBuffer.byteLength} bytes`);

                // Create session without SessionOptions constructor
                const sessionOptions = {
                    executionProviders: ['wasm']
                };

                console.log(`[ModelLoader] Creating ONNX session with WASM provider...`);
                const sess = await ort.InferenceSession.create(modelBuffer, sessionOptions);
                console.log(`[ModelLoader] ONNX session created successfully. Input names:`, sess.inputNames);
                console.log(`[ModelLoader] Output names:`, sess.outputNames);
                setSession(sess);
            } catch (err) {
                console.error(`[ModelLoader] Full error during load:`, err);
                setError(err);
            } finally {
                console.log(`[ModelLoader] Load process complete (session: ${!!session})`);
                setLoading(false);
            }
        }
        loadModel();
    }, []);

    return { session, loading, error };
}

// Parse .npy file (NumPy binary format, assuming version 1.0, float32, shape (256,))
const parseNpy = (arrayBuffer) => {
    console.log(`[NPY Parser] Starting parse, buffer length: ${arrayBuffer.byteLength}`);
    const u8 = new Uint8Array(arrayBuffer);

    // Magic bytes: \x93NUMPY
    if (u8[0] !== 0x93 || u8[1] !== 78 || u8[2] !== 85 || u8[3] !== 77 || u8[4] !== 80 || u8[5] !== 89) {
        const err = new Error('Invalid .npy: Wrong magic bytes');
        console.error(`[NPY Parser] Magic bytes mismatch:`, u8.subarray(0, 6));
        throw err;
    }
    console.log(`[NPY Parser] Valid magic bytes confirmed`);

    const major = u8[6], minor = u8[7];
    if (major !== 1 || minor !== 0) {
        const err = new Error('Only NumPy 1.0 format supported');
        console.error(`[NPY Parser] Version mismatch: major=${major}, minor=${minor}`);
        throw err;
    }
    console.log(`[NPY Parser] NumPy 1.0 version confirmed`);

    // Header length (little-endian uint16 for v1.0)
    const headerLen = u8[8] | (u8[9] << 8);
    console.log(`[NPY Parser] Header length: ${headerLen}`);

    const headerStart = 10;
    if (headerLen > arrayBuffer.byteLength - headerStart) {
        const err = new Error('Invalid .npy: Header too large');
        console.error(`[NPY Parser] Header length ${headerLen} exceeds buffer`);
        throw err;
    }

    const headerBytes = u8.subarray(headerStart, headerStart + headerLen);
    const headerStr = new TextDecoder('ascii').decode(headerBytes);
    console.log(`[NPY Parser] Header preview: ${headerStr.substring(0, 100)}...`);

    // Extract shape: e.g., 'shape': (256,)
    const shapeMatch = headerStr.match(/'shape':\s*\((\d+(,\s*)?)+?\)/);
    if (!shapeMatch) {
        const err = new Error('No shape found in .npy header');
        console.error(`[NPY Parser] No shape match in header`);
        throw err;
    }
    const shapeStr = shapeMatch[0].replace(/'shape':\s*\(|\)/g, '').replace(/\s/g, '');
    const shape = shapeStr.split(',').map(Number).filter(x => x > 0);
    console.log(`[NPY Parser] Parsed shape: [${shape.join(', ')}]`);

    // Extract dtype: assume '<f4' for float32
    const dtypeMatch = headerStr.match(/'descr':\s*'(<f4|float32)'/);
    if (!dtypeMatch || (dtypeMatch[1] !== '<f4' && dtypeMatch[1] !== 'float32')) {
        const err = new Error('Only float32 dtype supported');
        console.error(`[NPY Parser] Dtype mismatch: found '${dtypeMatch ? dtypeMatch[1] : 'none'}'`);
        throw err;
    }
    console.log(`[NPY Parser] Float32 dtype confirmed`);

    // Compute padding to align to 64 bytes
    const headerEnd = headerStart + headerLen;
    const pad = (64 - (headerEnd % 64)) % 64;
    const dataOffset = headerEnd + pad;
    console.log(`[NPY Parser] Data offset calculated: ${dataOffset} (padding: ${pad})`);

    const length = arrayBuffer.byteLength;
    if (dataOffset >= length) {
        const err = new Error('Invalid .npy: Data offset exceeds file size');
        console.error(`[NPY Parser] Offset ${dataOffset} >= length ${length}`);
        throw err;
    }

    const numElements = shape.reduce((a, b) => a * b, 1);
    const expectedElements = 256;
    if (numElements !== expectedElements) {
        const err = new Error(`Expected ${expectedElements} elements, got ${numElements}`);
        console.error(`[NPY Parser] Elements mismatch: expected ${expectedElements}, got ${numElements} (shape: [${shape.join(', ')}])`);
        throw err;
    }
    console.log(`[NPY Parser] Elements validation passed: ${numElements} elements`);

    if (dataOffset + numElements * 4 > length) {
        const err = new Error('Invalid .npy: Data exceeds file size');
        console.error(`[NPY Parser] Data end ${dataOffset + numElements * 4} > length ${length}`);
        throw err;
    }

    const data = new Float32Array(arrayBuffer, dataOffset, numElements);
    const dims = shape.length === 1 ? [1, shape[0]] : shape;
    console.log(`[NPY Parser] Tensor created: dims [${dims.join(', ')}], data range [${data[0].toFixed(4)}, ${data[data.length-1].toFixed(4)}]`);
    return new ort.Tensor("float32", data, dims);
};

// LRU Cache class for ephemeral tensor storage
class LRUCache {
    constructor(maxSize = 5) {
        this.maxSize = maxSize;
        this.cache = new Map();
    }

    get(key) {
        if (this.cache.has(key)) {
            const entry = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, entry);
            return entry;
        }
        return null;
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            console.log(`[Cache] Evicting oldest: ${firstKey} (size now ${this.maxSize})`);
            this.cache.delete(firstKey);
        }
        this.cache.set(key, { ...value, timestamp: Date.now() });
    }

    clear() {
        this.cache.clear();
        console.log(`[Cache] Cleared all ephemeral data`);
    }

    size() {
        return this.cache.size;
    }
}

export default function FundusDemo() {
    const { session, loading: modelLoading, error: modelError } = useModelLoader();
    const [showDemo, setShowDemo] = useState(false);
    const [testSplit, setTestSplit] = useState([]);
    const [testSplitLoading, setTestSplitLoading] = useState(true);
    const [testSplitError, setTestSplitError] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedPreprocPath, setSelectedPreprocPath] = useState(null);
    const [gtLabel, setGtLabel] = useState(null);
    const [results, setResults] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [tensorReady, setTensorReady] = useState(false);
    const [loadError, setLoadError] = useState(null);
    const [isWarmed, setIsWarmed] = useState(false);

    const cache = useRef(new LRUCache(5));

    // Clear cache on unmount
    useEffect(() => {
        return () => {
            cache.current.clear();
        };
    }, []);

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
                const data = await res.json(); // Define data here
                console.log(`[TestSplit] Loaded ${data.length} samples successfully`);
                setTestSplit(data);
            } catch (err) {
                console.error(`[TestSplit] Full error:`, err);
                setTestSplitError(err.message);
            } finally {
                setTestSplitLoading(false);
                // Remove the problematic condition here
                // The preloadNextBatch() will be called from handlePickRandom when needed
            }
        }
        fetchTestSplit();
    }, []);

    const warmStartModel = async (tensor) => {
        if (isWarmed) return;
        console.log(`[WarmStart] Running dummy inference on first tensor to warm model`);
        try {
            const inputName = session.inputNames[0];
            const noisyTensor = addNoiseToTensor(tensor);
            const feeds = { [inputName]: noisyTensor };
            await session.run(feeds);  // Silent dummy – discards output
            setIsWarmed(true);
            console.log(`[WarmStart] Model warmed (one-time done)`);
        } catch (err) {
            console.warn(`[WarmStart] Failed on first tensor:`, err);
        }
    };

    // Load .npy with logs
    const loadTensorFromNpy = async (npyPath) => {
        console.log(`[TensorLoader] Loading NPY from: ${npyPath}`);
        try {
            const response = await fetch(npyPath);
            if (!response.ok) {
                const err = new Error(`HTTP ${response.status} ${response.statusText}`);
                console.error(`[TensorLoader] Fetch failed for ${npyPath}:`, err);
                throw err;
            }
            console.log(`[TensorLoader] Fetched NPY, size: ${response.headers.get('content-length')} bytes`);
            const arrayBuffer = await response.arrayBuffer();
            if (arrayBuffer.byteLength < 1000) {
                const text = new TextDecoder().decode(arrayBuffer);
                const err = new Error(`Non-binary response: ${text.substring(0, 100)}`);
                console.error(`[TensorLoader] Invalid response content:`, err);
                throw err;
            }
            const tensor = parseNpy(arrayBuffer);
            console.log(`[TensorLoader] Tensor parsed and ready for ${npyPath}`);
            return tensor;
        } catch (err) {
            console.error(`[TensorLoader] Full error for ${npyPath}:`, err);
            throw err;
        }
    };

    // Add noise to tensor
    const addNoiseToTensor = (tensor, noiseStd = 0.001) => {
        console.log(`[NoiseAdder] Adding noise (std=${noiseStd}) to tensor`);
        const noisyData = new Float32Array(tensor.data);
        for (let i = 0; i < noisyData.length; i++) {
            noisyData[i] += (Math.random() - 0.5) * 2 * noiseStd;
        }
        const noisyTensor = new ort.Tensor("float32", noisyData, tensor.dims);
        console.log(`[NoiseAdder] Noise added, new range [${noisyData[0].toFixed(4)}, ${noisyData[noisyData.length-1].toFixed(4)}]`);
        return noisyTensor;
    };

    // Compute probabilities from logits
    const computeProbs = (output) => {
        console.log(`[Probs] Computing softmax from logits:`, output);
        const probs = new Float32Array(output.length);
        let maxLogit = -Infinity;
        for (let i = 0; i < output.length; i++) maxLogit = Math.max(maxLogit, output[i]);
        let sumExp = 0;
        for (let i = 0; i < output.length; i++) sumExp += Math.exp(output[i] - maxLogit);
        for (let i = 0; i < output.length; i++) probs[i] = Math.exp(output[i] - maxLogit) / sumExp;
        console.log(`[Probs] Softmax output:`, Array.from(probs));
        return probs;
    };

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
        setSelectedPreprocPath(sample.preproc_path);
        setGtLabel(sample.class_label_remapped);
        setResults([]);
        const cached = cache.current.get(sample.original_path);
        setTensorReady(!!cached?.tensor);
        setLoadError(null);
    };

    const handleRunInference = async () => {
        console.log(`[Inference] Starting inference with session: ${!!session}, image: ${selectedImage}, tensorReady: ${tensorReady}`);
        if (!session || !selectedImage) {
            console.warn(`[Inference] Skipped: missing session=${!session}, image=${!selectedImage}`);
            return;
        }

        setIsProcessing(true);
        try {
            let cached = cache.current.get(selectedImage);
            let tensor;
            const isFirstLoad = !cached?.tensor;  // Detect if this is a new tensor load
            if (isFirstLoad) {
                tensor = await loadTensorFromNpy(selectedPreprocPath);
                cache.current.set(selectedImage, { tensor, gt: gtLabel });
                console.log(`[Inference] Tensor loaded on-demand for ${selectedImage} (first-ever? ${!isWarmed})`);
                cached = cache.current.get(selectedImage);

                // One-time warm on this first tensor (silent, before real run)
                await warmStartModel(tensor);
            } else {
                tensor = cached.tensor;
            }
            setTensorReady(true);
            
            const noisyTensor = addNoiseToTensor(tensor);
            const start = performance.now();
            const inputName = session.inputNames[0];
            console.log(`[Inference] Feeding tensor to input: ${inputName}, shape: [${noisyTensor.dims.join(', ')}]`);
            const feeds = { [inputName]: noisyTensor };
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
            setLoadError(`Inference failed: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    useEffect(() => {
        if (!selectedPreprocPath) {
            setTensorReady(false);
            setLoadError(null);
            return;
        }
        const cached = cache.current.get(selectedImage);
        const ready = !!cached?.tensor;
        console.log(`[TensorEffect] Skipping load: path=${!!selectedPreprocPath}, cached=${ready}`);
        setTensorReady(ready);
        setLoadError(null);
    }, [selectedPreprocPath, gtLabel]);

    if (!showDemo) {
        return (
            <div className="fundus-demo">
                <h1 style={{ marginBottom: '0.5rem' }}>MLP on HNN Embeddings</h1>
                <h2 style={{ marginTop: '0', marginBottom: '2rem' }}>Fundus Classification Web Demo</h2>
                <button onClick={handleTryDemo}>
                    Try Demo
                </button>
            </div>
        );
    }

    return (
        <div className="fundus-demo">
            <h1 style={{ marginBottom: '0.5rem' }}>MLP on HNN Embeddings</h1>
            <h2 style={{ marginTop: '0', marginBottom: '2rem' }}>Fundus Classification Web Demo</h2>
            <button onClick={handlePickRandom}>
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

            <button onClick={handleRunInference} disabled={!selectedImage || isProcessing}>
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
        </div>
    );
}