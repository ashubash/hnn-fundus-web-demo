import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

export default function useModelLoader(modelUrl) {
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;

        async function loadModel() {
            try {
                setLoading(true);
                setError(null);

                // Ensure correct WASM paths
                ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";

                // Ensure features are disabled for simpler env
                ort.env.wasm.numThreads = 1;
                ort.env.allowJSEPSupport = false;

                // Correct session creation (no `Vt.SessionOptions`)
                const session = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ["wasm"],
                });

                if (!cancelled) {
                    console.log("Session created successfully");
                    setSession(session);
                }
            } catch (e) {
                console.error("Model loading failed:", e);
                if (!cancelled) setError(e);
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        loadModel();
        return () => { cancelled = true; };
    }, [modelUrl]);

    return { session, loading, error };
}