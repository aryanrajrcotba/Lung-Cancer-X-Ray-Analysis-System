import React, { useState, useRef } from 'react';
import { Upload, Activity, Cpu, CheckCircle, AlertCircle, BarChart3, TrendingUp } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

const LungCancerAnalysis = () => {
    const [image, setImage] = useState(null);
    const [images, setImages] = useState([]);
    const [isBatchMode, setIsBatchMode] = useState(false);
    const [preprocessing, setPreprocessing] = useState(false);
    const [iotStatus, setIotStatus] = useState('idle');
    const [modelResults, setModelResults] = useState(null);
    const [batchResults, setBatchResults] = useState([]);
    const [processedImage, setProcessedImage] = useState(null);
    const [filterCategory, setFilterCategory] = useState('all');
    const [showGraphs, setShowGraphs] = useState(false);
    const [currentProcessing, setCurrentProcessing] = useState(0);
    const [totalImages, setTotalImages] = useState(0);
    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                setImage(event.target.result);
                setImages([]);
                setIsBatchMode(false);
                setModelResults(null);
                setBatchResults([]);
                setProcessedImage(null);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleFolderUpload = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            setTotalImages(files.length);
            setIsBatchMode(true);
            setImage(null);
            setModelResults(null);
            setBatchResults([]);

            const imagePromises = files.map(file => {
                return new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = (event) => resolve(event.target.result);
                    reader.readAsDataURL(file);
                });
            });

            Promise.all(imagePromises).then(imageData => {
                setImages(imageData);
            });
        }
    };

    const preprocessImage = async (imageSrc) => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');

                // Resize to standard size
                canvas.width = 224;
                canvas.height = 224;

                // Draw and convert to grayscale
                ctx.drawImage(img, 0, 0, 224, 224);
                const imageData = ctx.getImageData(0, 0, 224, 224);
                const data = imageData.data;

                // Grayscale conversion
                for (let i = 0; i < data.length; i += 4) {
                    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    data[i] = avg;
                    data[i + 1] = avg;
                    data[i + 2] = avg;
                }

                // Apply histogram equalization (simplified)
                const histogram = new Array(256).fill(0);
                for (let i = 0; i < data.length; i += 4) {
                    histogram[data[i]]++;
                }

                const cdf = histogram.reduce((acc, val, idx) => {
                    acc[idx] = (acc[idx - 1] || 0) + val;
                    return acc;
                }, []);

                const cdfMin = cdf.find(v => v > 0);
                const totalPixels = 224 * 224;

                for (let i = 0; i < data.length; i += 4) {
                    const normalized = Math.round(((cdf[data[i]] - cdfMin) / (totalPixels - cdfMin)) * 255);
                    data[i] = normalized;
                    data[i + 1] = normalized;
                    data[i + 2] = normalized;
                }

                ctx.putImageData(imageData, 0, 0);
                resolve(canvas.toDataURL());
            };
            img.src = imageSrc;
        });
    };

    const simulateIoTTransmission = async (data) => {
        setIotStatus('connecting');
        await new Promise(resolve => setTimeout(resolve, 800));

        setIotStatus('transmitting');
        await new Promise(resolve => setTimeout(resolve, 1200));

        setIotStatus('processing');
        await new Promise(resolve => setTimeout(resolve, 1000));

        setIotStatus('complete');
    };

    const runMLModels = async (imageData) => {
        // Comprehensive set of ML models for research comparison
        const models = [
            // Deep CNN Architectures
            { name: 'ResNet-50', category: 'Deep CNN', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '25.6M' },
            { name: 'ResNet-101', category: 'Deep CNN', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '44.5M' },
            { name: 'ResNet-152', category: 'Deep CNN', accuracy: 0.96, confidence: 0.93, prediction: 'Malignant', params: '60.2M' },
            { name: 'ResNeXt-50', category: 'Deep CNN', accuracy: 0.95, confidence: 0.90, prediction: 'Malignant', params: '25.0M' },

            // VGG Family
            { name: 'VGG-16', category: 'VGG', accuracy: 0.91, confidence: 0.85, prediction: 'Malignant', params: '138M' },
            { name: 'VGG-19', category: 'VGG', accuracy: 0.92, confidence: 0.86, prediction: 'Malignant', params: '144M' },

            // EfficientNet Family
            { name: 'EfficientNet-B0', category: 'EfficientNet', accuracy: 0.93, confidence: 0.88, prediction: 'Malignant', params: '5.3M' },
            { name: 'EfficientNet-B3', category: 'EfficientNet', accuracy: 0.96, confidence: 0.92, prediction: 'Malignant', params: '12M' },
            { name: 'EfficientNet-B7', category: 'EfficientNet', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '66M' },
            { name: 'EfficientNetV2', category: 'EfficientNet', accuracy: 0.97, confidence: 0.95, prediction: 'Malignant', params: '21.5M' },

            // DenseNet Family
            { name: 'DenseNet-121', category: 'DenseNet', accuracy: 0.93, confidence: 0.87, prediction: 'Malignant', params: '8.0M' },
            { name: 'DenseNet-169', category: 'DenseNet', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '14.1M' },
            { name: 'DenseNet-201', category: 'DenseNet', accuracy: 0.95, confidence: 0.90, prediction: 'Malignant', params: '20.0M' },

            // Inception Family
            { name: 'Inception-v3', category: 'Inception', accuracy: 0.93, confidence: 0.88, prediction: 'Malignant', params: '23.8M' },
            { name: 'Inception-v4', category: 'Inception', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '42.7M' },
            { name: 'Inception-ResNet-v2', category: 'Inception', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '55.8M' },
            { name: 'Xception', category: 'Inception', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '22.9M' },

            // Mobile & Lightweight Models
            { name: 'MobileNet-v1', category: 'Mobile', accuracy: 0.88, confidence: 0.81, prediction: 'Benign', params: '4.2M' },
            { name: 'MobileNet-v2', category: 'Mobile', accuracy: 0.89, confidence: 0.83, prediction: 'Malignant', params: '3.5M' },
            { name: 'MobileNet-v3', category: 'Mobile', accuracy: 0.90, confidence: 0.84, prediction: 'Malignant', params: '5.4M' },
            { name: 'SqueezeNet', category: 'Mobile', accuracy: 0.86, confidence: 0.79, prediction: 'Benign', params: '1.2M' },
            { name: 'ShuffleNet', category: 'Mobile', accuracy: 0.87, confidence: 0.80, prediction: 'Benign', params: '2.3M' },

            // Vision Transformers
            { name: 'ViT-Base', category: 'Transformer', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '86M' },
            { name: 'ViT-Large', category: 'Transformer', accuracy: 0.96, confidence: 0.93, prediction: 'Malignant', params: '304M' },
            { name: 'DeiT-Base', category: 'Transformer', accuracy: 0.94, confidence: 0.90, prediction: 'Malignant', params: '87M' },
            { name: 'Swin-Transformer', category: 'Transformer', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '88M' },
            { name: 'BEiT', category: 'Transformer', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '86M' },

            // Specialized Medical Models
            { name: 'CheXNet', category: 'Medical', accuracy: 0.96, confidence: 0.92, prediction: 'Malignant', params: '7.0M' },
            { name: 'DeepChest', category: 'Medical', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '15.3M' },
            { name: 'ChestX-ray14', category: 'Medical', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '25.6M' },

            // Attention-based Models
            { name: 'SENet-154', category: 'Attention', accuracy: 0.96, confidence: 0.92, prediction: 'Malignant', params: '115M' },
            { name: 'CBAM-ResNet', category: 'Attention', accuracy: 0.95, confidence: 0.90, prediction: 'Malignant', params: '28.1M' },
            { name: 'ECA-Net', category: 'Attention', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '25.7M' },

            // NAS Models
            { name: 'NASNet-Large', category: 'NAS', accuracy: 0.96, confidence: 0.92, prediction: 'Malignant', params: '88.9M' },
            { name: 'AmoebaNet', category: 'NAS', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '86.7M' },
            { name: 'PNASNet', category: 'NAS', accuracy: 0.95, confidence: 0.90, prediction: 'Malignant', params: '86.1M' },

            // Hybrid & Ensemble Models
            { name: 'ConvNeXt', category: 'Hybrid', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '89M' },
            { name: 'CoAtNet', category: 'Hybrid', accuracy: 0.97, confidence: 0.95, prediction: 'Malignant', params: '168M' },
            { name: 'NFNet', category: 'Hybrid', accuracy: 0.96, confidence: 0.93, prediction: 'Malignant', params: '120M' },

            // Custom Ensemble
            { name: 'Ensemble (Top-5)', category: 'Ensemble', accuracy: 0.98, confidence: 0.96, prediction: 'Malignant', params: 'N/A' }
        ];

        const results = [];
        for (let i = 0; i < models.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 300));

            // Add some randomness to simulate real predictions
            const confidence = models[i].confidence + (Math.random() - 0.5) * 0.08;
            results.push({
                ...models[i],
                confidence: Math.min(0.99, Math.max(0.72, confidence)),
                processingTime: Math.random() * 300 + 80
            });

            setModelResults([...results]);
        }

        return results;
    };

    const startAnalysis = async () => {
        if (!image) return;

        setPreprocessing(true);
        setModelResults(null);

        // Step 1: Preprocess image
        const processed = await preprocessImage(image);
        setProcessedImage(processed);
        await new Promise(resolve => setTimeout(resolve, 500));
        setPreprocessing(false);

        // Step 2: IoT transmission
        await simulateIoTTransmission(processed);

        // Step 3: Run ML models
        await runMLModels(processed);
    };

    const startBatchAnalysis = async () => {
        if (images.length === 0) return;

        setPreprocessing(true);
        setBatchResults([]);
        setCurrentProcessing(0);

        const allResults = [];

        for (let i = 0; i < images.length; i++) {
            setCurrentProcessing(i + 1);

            // Preprocess image
            const processed = await preprocessImage(images[i]);

            // Simulate IoT transmission (faster for batch)
            setIotStatus('transmitting');
            await new Promise(resolve => setTimeout(resolve, 100));

            // Run models (faster simulation for batch)
            const imageResults = await runBatchMLModels(processed);

            allResults.push({
                imageIndex: i + 1,
                results: imageResults
            });

            setBatchResults([...allResults]);
        }

        setPreprocessing(false);
        setIotStatus('complete');

        // Aggregate results for summary
        aggregateBatchResults(allResults);
    };

    const runBatchMLModels = async (imageData) => {
        const models = [
            { name: 'ResNet-50', category: 'Deep CNN', accuracy: 0.94, confidence: 0.89, prediction: 'Malignant', params: '25.6M' },
            { name: 'ResNet-101', category: 'Deep CNN', accuracy: 0.95, confidence: 0.91, prediction: 'Malignant', params: '44.5M' },
            { name: 'EfficientNet-B7', category: 'EfficientNet', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '66M' },
            { name: 'DenseNet-201', category: 'DenseNet', accuracy: 0.95, confidence: 0.90, prediction: 'Malignant', params: '20.0M' },
            { name: 'ViT-Large', category: 'Transformer', accuracy: 0.96, confidence: 0.93, prediction: 'Malignant', params: '304M' },
            { name: 'Swin-Transformer', category: 'Transformer', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '88M' },
            { name: 'CheXNet', category: 'Medical', accuracy: 0.96, confidence: 0.92, prediction: 'Malignant', params: '7.0M' },
            { name: 'ConvNeXt', category: 'Hybrid', accuracy: 0.97, confidence: 0.94, prediction: 'Malignant', params: '89M' },
            { name: 'MobileNet-v3', category: 'Mobile', accuracy: 0.90, confidence: 0.84, prediction: 'Malignant', params: '5.4M' },
            { name: 'Ensemble (Top-5)', category: 'Ensemble', accuracy: 0.98, confidence: 0.96, prediction: 'Malignant', params: 'N/A' }
        ];

        return models.map(model => ({
            ...model,
            confidence: Math.min(0.99, Math.max(0.72, model.confidence + (Math.random() - 0.5) * 0.08)),
            processingTime: Math.random() * 300 + 80
        }));
    };

    const aggregateBatchResults = (allResults) => {
        const modelAggregates = {};

        allResults.forEach(imageResult => {
            imageResult.results.forEach(model => {
                if (!modelAggregates[model.name]) {
                    modelAggregates[model.name] = {
                        ...model,
                        confidences: [],
                        processingTimes: []
                    };
                }
                modelAggregates[model.name].confidences.push(model.confidence);
                modelAggregates[model.name].processingTimes.push(model.processingTime);
            });
        });

        const aggregatedResults = Object.values(modelAggregates).map(model => ({
            name: model.name,
            category: model.category,
            accuracy: model.accuracy,
            params: model.params,
            prediction: model.prediction,
            confidence: model.confidences.reduce((a, b) => a + b, 0) / model.confidences.length,
            processingTime: model.processingTimes.reduce((a, b) => a + b, 0) / model.processingTimes.length,
            minConfidence: Math.min(...model.confidences),
            maxConfidence: Math.max(...model.confidences),
            stdDev: calculateStdDev(model.confidences)
        }));

        setModelResults(aggregatedResults);
    };

    const calculateStdDev = (values) => {
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const squareDiffs = values.map(value => Math.pow(value - avg, 2));
        const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
        return Math.sqrt(avgSquareDiff);
    };

    const getBestModel = () => {
        if (!modelResults || modelResults.length === 0) return null;
        return modelResults.reduce((best, current) =>
            current.confidence > best.confidence ? current : best
        );
    };

    const getTopModels = (n = 5) => {
        if (!modelResults || modelResults.length === 0) return [];
        return [...modelResults]
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, n);
    };

    const getCategories = () => {
        if (!modelResults || modelResults.length === 0) return [];
        const cats = new Set(modelResults.map(m => m.category));
        return ['all', ...Array.from(cats)];
    };

    const getFilteredModels = () => {
        if (!modelResults) return [];
        if (filterCategory === 'all') return modelResults;
        return modelResults.filter(m => m.category === filterCategory);
    };

    const getCategoryStats = () => {
        if (!modelResults) return [];
        const stats = {};
        modelResults.forEach(model => {
            if (!stats[model.category]) {
                stats[model.category] = { category: model.category, avgConfidence: 0, avgAccuracy: 0, count: 0 };
            }
            stats[model.category].avgConfidence += model.confidence;
            stats[model.category].avgAccuracy += model.accuracy;
            stats[model.category].count += 1;
        });

        return Object.values(stats).map(stat => ({
            category: stat.category,
            avgConfidence: (stat.avgConfidence / stat.count * 100).toFixed(1),
            avgAccuracy: (stat.avgAccuracy / stat.count * 100).toFixed(1),
            count: stat.count
        }));
    };

    const getAccuracyVsParams = () => {
        if (!modelResults) return [];
        return modelResults.map(model => ({
            name: model.name,
            accuracy: (model.accuracy * 100).toFixed(1),
            params: parseFloat(model.params.replace('M', '')),
            confidence: (model.confidence * 100).toFixed(1)
        })).filter(m => !isNaN(m.params));
    };

    const getConfusionMatrix = () => {
        if (!modelResults) return null;
        const malignantPredictions = modelResults.filter(m => m.prediction === 'Malignant').length;
        const benignPredictions = modelResults.filter(m => m.prediction === 'Benign').length;

        return {
            truePositive: malignantPredictions * 0.85,
            falsePositive: benignPredictions * 0.15,
            trueNegative: benignPredictions * 0.85,
            falseNegative: malignantPredictions * 0.15
        };
    };

    const getPerformanceMetrics = () => {
        if (!modelResults) return [];
        const cm = getConfusionMatrix();
        if (!cm) return [];

        const precision = cm.truePositive / (cm.truePositive + cm.falsePositive);
        const recall = cm.truePositive / (cm.truePositive + cm.falseNegative);
        const f1Score = 2 * (precision * recall) / (precision + recall);
        const specificity = cm.trueNegative / (cm.trueNegative + cm.falsePositive);

        return [
            { metric: 'Precision', value: (precision * 100).toFixed(1), fullMark: 100 },
            { metric: 'Recall', value: (recall * 100).toFixed(1), fullMark: 100 },
            { metric: 'F1-Score', value: (f1Score * 100).toFixed(1), fullMark: 100 },
            { metric: 'Specificity', value: (specificity * 100).toFixed(1), fullMark: 100 },
            { metric: 'Accuracy', value: ((cm.truePositive + cm.trueNegative) / (cm.truePositive + cm.trueNegative + cm.falsePositive + cm.falseNegative) * 100).toFixed(1), fullMark: 100 }
        ];
    };

    const getPredictionDistribution = () => {
        if (!modelResults) return [];
        const malignant = modelResults.filter(m => m.prediction === 'Malignant').length;
        const benign = modelResults.filter(m => m.prediction === 'Benign').length;

        return [
            { name: 'Malignant', value: malignant, color: '#ef4444' },
            { name: 'Benign', value: benign, color: '#22c55e' }
        ];
    };

    const getProcessingTimeComparison = () => {
        if (!modelResults) return [];
        return modelResults
            .sort((a, b) => a.processingTime - b.processingTime)
            .slice(0, 10)
            .map(m => ({
                name: m.name.length > 15 ? m.name.substring(0, 15) + '...' : m.name,
                time: m.processingTime.toFixed(0)
            }));
    };

    const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#6366f1', '#f97316'];

    const bestModel = getBestModel();
    const topModels = getTopModels();
    const categories = getCategories();
    const filteredModels = getFilteredModels();
    const categoryStats = getCategoryStats();
    const accuracyVsParams = getAccuracyVsParams();
    const performanceMetrics = getPerformanceMetrics();
    const predictionDist = getPredictionDistribution();
    const processingTimeData = getProcessingTimeComparison();

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
            <div className="max-w-7xl mx-auto">
                <div className="bg-white rounded-2xl shadow-2xl p-8">
                    <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center gap-3">
                        <Activity className="text-blue-600" size={40} />
                        Lung Cancer X-Ray Analysis System
                    </h1>
                    <p className="text-gray-600 mb-8">AI-Powered IoT Medical Imaging Analysis</p>

                    {/* Upload Section */}
                    <div className="mb-8">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="border-4 border-dashed border-blue-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all"
                            >
                                <Upload className="mx-auto text-blue-500 mb-3" size={40} />
                                <p className="text-lg text-gray-700 mb-1 font-semibold">
                                    Single Image Upload
                                </p>
                                <p className="text-sm text-gray-500">
                                    {image ? 'Image uploaded! Click to change' : 'Click to upload one X-Ray image'}
                                </p>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={handleImageUpload}
                                    className="hidden"
                                />
                            </div>

                            <div
                                onClick={() => folderInputRef.current?.click()}
                                className="border-4 border-dashed border-purple-300 rounded-xl p-8 text-center cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-all"
                            >
                                <Upload className="mx-auto text-purple-500 mb-3" size={40} />
                                <p className="text-lg text-gray-700 mb-1 font-semibold">
                                    Batch Upload (Folder)
                                </p>
                                <p className="text-sm text-gray-500">
                                    {images.length > 0 ? `${images.length} images loaded` : 'Click to upload multiple images'}
                                </p>
                                <input
                                    ref={folderInputRef}
                                    type="file"
                                    accept="image/*"
                                    multiple
                                    webkitdirectory=""
                                    directory=""
                                    onChange={handleFolderUpload}
                                    className="hidden"
                                />
                            </div>
                        </div>

                        {image && !isBatchMode && (
                            <button
                                onClick={startAnalysis}
                                disabled={preprocessing || iotStatus !== 'idle' && iotStatus !== 'complete'}
                                className="w-full bg-blue-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
                            >
                                {preprocessing ? 'Processing...' : 'Start Single Image Analysis'}
                            </button>
                        )}

                        {images.length > 0 && isBatchMode && (
                            <div>
                                <button
                                    onClick={startBatchAnalysis}
                                    disabled={preprocessing}
                                    className="w-full bg-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
                                >
                                    {preprocessing ? `Processing ${currentProcessing}/${totalImages}...` : `Start Batch Analysis (${images.length} images)`}
                                </button>

                                {preprocessing && (
                                    <div className="mt-4 bg-white rounded-lg p-4">
                                        <div className="flex justify-between mb-2">
                                            <span className="text-sm text-gray-600">Progress</span>
                                            <span className="text-sm font-semibold text-gray-800">{currentProcessing}/{totalImages}</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-4">
                                            <div
                                                className="bg-gradient-to-r from-purple-500 to-blue-500 h-4 rounded-full transition-all duration-300"
                                                style={{ width: `${(currentProcessing / totalImages) * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {modelResults && modelResults.length > 0 && (
                            <button
                                onClick={() => setShowGraphs(!showGraphs)}
                                className="mt-4 w-full bg-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-purple-700 transition-all flex items-center justify-center gap-2"
                            >
                                <TrendingUp size={24} />
                                {showGraphs ? 'Hide Research Graphs' : 'Show Research Graphs'}
                            </button>
                        )}
                    </div>

                    {/* Research Graphs Section */}
                    {showGraphs && modelResults && modelResults.length > 0 && (
                        <div className="mt-8 space-y-6">
                            <div className="bg-white rounded-xl shadow-lg p-6 border-t-4 border-purple-500">
                                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                                    <TrendingUp className="text-purple-600" size={32} />
                                    Research Analysis & Visualization
                                </h2>

                                {/* Model Performance Comparison */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Model Confidence Comparison (Top 15)</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart data={topModels.slice(0, 15)}>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} fontSize={10} />
                                                <YAxis />
                                                <Tooltip />
                                                <Legend />
                                                <Bar dataKey="confidence" fill="#3b82f6" name="Confidence (%)" />
                                                <Bar dataKey="accuracy" fill="#8b5cf6" name="Accuracy (%)" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="bg-gradient-to-br from-green-50 to-teal-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Category-wise Performance</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart data={categoryStats}>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="category" angle={-45} textAnchor="end" height={100} fontSize={11} />
                                                <YAxis />
                                                <Tooltip />
                                                <Legend />
                                                <Bar dataKey="avgConfidence" fill="#10b981" name="Avg Confidence %" />
                                                <Bar dataKey="avgAccuracy" fill="#06b6d4" name="Avg Accuracy %" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Accuracy vs Parameters Scatter */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Accuracy vs Model Parameters</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <ScatterChart>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="params" name="Parameters (M)" label={{ value: 'Parameters (Millions)', position: 'bottom' }} />
                                                <YAxis dataKey="accuracy" name="Accuracy" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                                <Legend />
                                                <Scatter name="Models" data={accuracyVsParams} fill="#8b5cf6" />
                                            </ScatterChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="bg-gradient-to-br from-orange-50 to-yellow-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Performance Metrics (Radar)</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <RadarChart data={performanceMetrics}>
                                                <PolarGrid />
                                                <PolarAngleAxis dataKey="metric" />
                                                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                                                <Radar name="Metrics" dataKey="value" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} />
                                                <Tooltip />
                                            </RadarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Prediction Distribution and Processing Time */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                                    <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Prediction Distribution</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <PieChart>
                                                <Pie
                                                    data={predictionDist}
                                                    cx="50%"
                                                    cy="50%"
                                                    labelLine={false}
                                                    label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(0)}%)`}
                                                    outerRadius={100}
                                                    fill="#8884d8"
                                                    dataKey="value"
                                                >
                                                    {predictionDist.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                                    ))}
                                                </Pie>
                                                <Tooltip />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg p-4">
                                        <h3 className="font-bold text-gray-800 mb-4">Processing Time (Top 10 Fastest)</h3>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart data={processingTimeData} layout="vertical">
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis type="number" />
                                                <YAxis dataKey="name" type="category" width={120} fontSize={10} />
                                                <Tooltip />
                                                <Legend />
                                                <Bar dataKey="time" fill="#06b6d4" name="Time (ms)" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Model Confidence Distribution Line Chart */}
                                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-4">
                                    <h3 className="font-bold text-gray-800 mb-4">Model Confidence Trend</h3>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <LineChart data={modelResults.slice(0, 20)}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} fontSize={9} />
                                            <YAxis domain={[70, 100]} />
                                            <Tooltip />
                                            <Legend />
                                            <Line type="monotone" dataKey="confidence" stroke="#6366f1" strokeWidth={2} name="Confidence (%)" dot={{ r: 4 }} />
                                            <Line type="monotone" dataKey="accuracy" stroke="#ec4899" strokeWidth={2} name="Accuracy (%)" dot={{ r: 4 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Statistical Summary Table */}
                                <div className="mt-6 bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg p-4">
                                    <h3 className="font-bold text-gray-800 mb-4">Statistical Summary</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        <div className="bg-white p-4 rounded-lg shadow">
                                            <p className="text-sm text-gray-600">{isBatchMode ? 'Images Processed' : 'Total Models'}</p>
                                            <p className="text-2xl font-bold text-blue-600">{isBatchMode ? totalImages : modelResults.length}</p>
                                        </div>
                                        <div className="bg-white p-4 rounded-lg shadow">
                                            <p className="text-sm text-gray-600">Avg Confidence</p>
                                            <p className="text-2xl font-bold text-green-600">
                                                {(modelResults.reduce((acc, m) => acc + m.confidence, 0) / modelResults.length * 100).toFixed(1)}%
                                            </p>
                                        </div>
                                        <div className="bg-white p-4 rounded-lg shadow">
                                            <p className="text-sm text-gray-600">Avg Accuracy</p>
                                            <p className="text-2xl font-bold text-purple-600">
                                                {(modelResults.reduce((acc, m) => acc + m.accuracy, 0) / modelResults.length * 100).toFixed(1)}%
                                            </p>
                                        </div>
                                        <div className="bg-white p-4 rounded-lg shadow">
                                            <p className="text-sm text-gray-600">{isBatchMode ? 'Models Used' : 'Categories'}</p>
                                            <p className="text-2xl font-bold text-orange-600">{isBatchMode ? modelResults.length : categoryStats.length}</p>
                                        </div>
                                    </div>

                                    {isBatchMode && modelResults.length > 0 && (
                                        <div className="mt-6">
                                            <h4 className="font-semibold text-gray-800 mb-3">Batch Analysis Details</h4>
                                            <div className="bg-white rounded-lg p-4 space-y-3">
                                                {modelResults.map((model, idx) => (
                                                    <div key={idx} className="border-b last:border-b-0 pb-3 last:pb-0">
                                                        <div className="flex justify-between items-start mb-2">
                                                            <div>
                                                                <p className="font-semibold text-gray-800">{model.name}</p>
                                                                <p className="text-xs text-gray-500">{model.category} | {model.params}</p>
                                                            </div>
                                                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${model.prediction === 'Malignant'
                                                                    ? 'bg-red-100 text-red-700'
                                                                    : 'bg-green-100 text-green-700'
                                                                }`}>
                                                                {model.prediction}
                                                            </span>
                                                        </div>
                                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                                                            <div>
                                                                <p className="text-gray-500">Avg Confidence</p>
                                                                <p className="font-semibold text-blue-600">{(model.confidence * 100).toFixed(1)}%</p>
                                                            </div>
                                                            <div>
                                                                <p className="text-gray-500">Min/Max</p>
                                                                <p className="font-semibold text-gray-700">
                                                                    {(model.minConfidence * 100).toFixed(1)}% / {(model.maxConfidence * 100).toFixed(1)}%
                                                                </p>
                                                            </div>
                                                            <div>
                                                                <p className="text-gray-500">Std Dev</p>
                                                                <p className="font-semibold text-purple-600">{(model.stdDev * 100).toFixed(2)}%</p>
                                                            </div>
                                                            <div>
                                                                <p className="text-gray-500">Avg Time</p>
                                                                <p className="font-semibold text-orange-600">{model.processingTime.toFixed(0)}ms</p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Image Display */}
                        {image && (
                            <div className="space-y-4">
                                <div className="bg-gray-50 rounded-xl p-4">
                                    <h3 className="font-semibold text-gray-700 mb-3">Original Image</h3>
                                    <img src={image} alt="Original" className="w-full rounded-lg shadow" />
                                </div>

                                {processedImage && (
                                    <div className="bg-gray-50 rounded-xl p-4">
                                        <h3 className="font-semibold text-gray-700 mb-3">Preprocessed Image</h3>
                                        <img src={processedImage} alt="Processed" className="w-full rounded-lg shadow" />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Status and Results */}
                        <div className="space-y-4">
                            {/* IoT Status */}
                            {iotStatus !== 'idle' && (
                                <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
                                    <div className="flex items-center gap-3 mb-4">
                                        <Cpu className="text-purple-600" size={24} />
                                        <h3 className="font-semibold text-gray-800">IoT Device Status</h3>
                                    </div>

                                    <div className="space-y-3">
                                        <div className="flex items-center justify-between">
                                            <span className="text-gray-700">Connection</span>
                                            {iotStatus !== 'idle' ? (
                                                <CheckCircle className="text-green-500" size={20} />
                                            ) : (
                                                <AlertCircle className="text-gray-400" size={20} />
                                            )}
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-gray-700">Data Transmission</span>
                                            {['transmitting', 'processing', 'complete'].includes(iotStatus) ? (
                                                <CheckCircle className="text-green-500" size={20} />
                                            ) : (
                                                <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin" />
                                            )}
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-gray-700">Cloud Processing</span>
                                            {iotStatus === 'complete' ? (
                                                <CheckCircle className="text-green-500" size={20} />
                                            ) : iotStatus === 'processing' ? (
                                                <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin" />
                                            ) : (
                                                <AlertCircle className="text-gray-400" size={20} />
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Model Results */}
                            {modelResults && modelResults.length > 0 && (
                                <div className="bg-gradient-to-r from-green-50 to-teal-50 rounded-xl p-6 border border-green-200">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="flex items-center gap-3">
                                            <BarChart3 className="text-green-600" size={24} />
                                            <h3 className="font-semibold text-gray-800">
                                                Model Results ({modelResults.length} models)
                                            </h3>
                                        </div>
                                        <select
                                            value={filterCategory}
                                            onChange={(e) => setFilterCategory(e.target.value)}
                                            className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
                                        >
                                            {categories.map(cat => (
                                                <option key={cat} value={cat}>
                                                    {cat === 'all' ? 'All Categories' : cat}
                                                </option>
                                            ))}
                                        </select>
                                    </div>

                                    <div className="space-y-2 max-h-96 overflow-y-auto">
                                        {filteredModels.map((model, idx) => (
                                            <div key={idx} className="bg-white rounded-lg p-3 shadow-sm">
                                                <div className="flex justify-between items-center mb-2">
                                                    <div>
                                                        <span className="font-medium text-gray-800">{model.name}</span>
                                                        <span className="ml-2 text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                                                            {model.category}
                                                        </span>
                                                        <span className="ml-2 text-xs text-gray-500">
                                                            {model.params}
                                                        </span>
                                                    </div>
                                                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${model.prediction === 'Malignant'
                                                            ? 'bg-red-100 text-red-700'
                                                            : 'bg-green-100 text-green-700'
                                                        }`}>
                                                        {model.prediction}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                                                        <div
                                                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                                                            style={{ width: `${model.confidence * 100}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-xs font-semibold text-gray-700 w-12 text-right">
                                                        {(model.confidence * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <p className="text-xs text-gray-500 mt-1">
                                                    Accuracy: {(model.accuracy * 100).toFixed(1)}% | Time: {model.processingTime.toFixed(0)}ms
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Best Model Recommendation */}
                            {bestModel && (
                                <>
                                    <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl p-6 border-2 border-yellow-400">
                                        <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                                            <CheckCircle className="text-yellow-600" size={24} />
                                            Best Single Model
                                        </h3>
                                        <p className="text-2xl font-bold text-gray-800 mb-2">{bestModel.name}</p>
                                        <p className="text-sm text-gray-600 mb-2">Category: {bestModel.category} | Parameters: {bestModel.params}</p>
                                        <p className="text-gray-700">
                                            Confidence: <span className="font-bold text-blue-600">
                                                {(bestModel.confidence * 100).toFixed(1)}%
                                            </span>
                                        </p>
                                        <p className="text-gray-700">
                                            Prediction: <span className={`font-bold ${bestModel.prediction === 'Malignant' ? 'text-red-600' : 'text-green-600'
                                                }`}>
                                                {bestModel.prediction}
                                            </span>
                                        </p>
                                    </div>

                                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-400">
                                        <h3 className="font-bold text-gray-800 mb-3">Top 5 Models (Research Comparison)</h3>
                                        <div className="space-y-2">
                                            {topModels.map((model, idx) => (
                                                <div key={idx} className="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm">
                                                    <div>
                                                        <p className="font-semibold text-gray-800">#{idx + 1} {model.name}</p>
                                                        <p className="text-xs text-gray-500">{model.category}</p>
                                                    </div>
                                                    <div className="text-right">
                                                        <p className="font-bold text-blue-600">{(model.confidence * 100).toFixed(1)}%</p>
                                                        <p className="text-xs text-gray-500">{model.prediction}</p>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}

                            {bestModel && (
                                <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                                    <p className="text-sm text-gray-600 italic">
                                        ⚠️ This is a simulation for research purposes. Always consult healthcare professionals for medical diagnosis.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    <canvas ref={canvasRef} className="hidden" />
                </div>
            </div>
        </div>
    );
};

export default LungCancerAnalysis;
