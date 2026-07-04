// "use client";

// import { useState } from "react";
// import axios from "axios";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
// } from "recharts";

// export default function Home() {
//   const [formData, setFormData] = useState({
//     Crop: "Cotton",
//     Rainfall_mm: "",
//     Temperature_Celsius: "",
//     Irrigation_Used: "",
//     Fertilizer_Used: "",
//   });
//   const crops = ["Cotton", "Rice", "Barley", "Soybean", "Wheat", "Maize"];
//   const [result, setResult] = useState<any>(null);
//   const [loading, setLoading] = useState(false);

//   const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     setFormData({ ...formData, [e.target.name]: e.target.value });
//   };

//   const handleSubmit = async () => {
//     setLoading(true);
//     try {
//       const res = await axios.post("http://127.0.0.1:5000/predict", formData);
//       setResult(res.data);
//     } catch (err) {
//       console.error(err);
//       alert("Error connecting to backend!");
//     }
//     setLoading(false);
//   };

//   // Prepare probability chart data
//   const probChartData =
//     result && result.model_probabilities
//       ? Object.entries(result.model_probabilities).flatMap(([model, probs]: any) => [
//           { model: `${model}-Low`, probability: probs.Low ?? 0 },
//           { model: `${model}-High`, probability: probs.High ?? 0 },
//         ])
//       : [];

//   return (
//     <div className="min-h-screen bg-gradient-to-r from-green-200 via-yellow-100 to-green-300 flex flex-col items-center p-6">
//       {/* Heading */}
//       <h1 className="text-4xl font-bold mb-2 text-gray-800">
//         SML Crop Yield Predictor
//       </h1>
//       <div className="h-1 w-48 bg-green-500 mb-6 animate-pulse"></div>

//       {/* Input Form */}
//       <div className="bg-white shadow-md rounded-lg p-6 w-full max-w-md space-y-4">
//         {/* Crop dropdown - separate div */}
// <div className="flex flex-col">
//   <label className="mb-1 font-medium text-gray-700">Crop</label>
//   <select
//     name="Crop"
//     value={formData.Crop}
//     onChange={(e) => setFormData({ ...formData, Crop: e.target.value })}
//     className="border border-gray-300 rounded px-3 py-2 bg-white text-black focus:outline-none focus:ring-2 focus:ring-green-400 hover:border-gray-400"
//   >
//     {crops.map((crop) => (
//       <option key={crop} value={crop}>{crop}</option>
//     ))}
//   </select>
// </div>
//         {["Rainfall_mm", "Temperature_Celsius", "Irrigation_Used", "Fertilizer_Used"].map(
//           (field) => (
//             <div key={field} className="flex flex-col">
//               <label className="mb-1 font-medium text-gray-700">{field.replace("_", " ")}</label>
//               <input
//                 type="number"
//                 name={field}
//                 value={(formData as any)[field]}
//                 onChange={handleChange}
//                 className="border border-gray-300 rounded px-3 py-2 bg-white text-black focus:outline-none focus:ring-2 focus:ring-green-400 hover:border-gray-400"

//                 // className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-400 hover:border-gray-400"
//               />
//             </div>
//           )
//         )}
//         <button
//           onClick={handleSubmit}
//           className="w-full bg-green-500 text-white py-2 rounded hover:bg-green-600 transition"
//         >
//           {loading ? "Predicting..." : "Predict Yield"}
//         </button>
//       </div>

//       {/* Output */}
//       {result && (
//         <div className="bg-white shadow-md rounded-lg p-6 mt-6 w-full max-w-md space-y-4">
//           <h2 className="text-2xl font-semibold mb-2 text-gray-800">Prediction</h2>
//            <p className="mb-2 text-green-700">
//       <strong className="text-green-700">Exact Yield:</strong> {result.exact_yield.toFixed(2)} tons/ha
//     </p>

//     <p className="mb-4 text-green-700">
//       <strong className="text-green-700">High/Low Yield:</strong>{" "}
//       {result.high_yield === 1 ? "High" : "Low"}
//     </p>

//           {/* Accuracy Chart */}
//           <h3 className="text-xl font-semibold mt-4 text-black">Model Accuracy Comparison</h3>
//           <ResponsiveContainer width="100%" height={200}>
//             <BarChart data={result.model_accuracy}>
//               <XAxis dataKey="model" />
//               <YAxis />
//               <Tooltip />
//               <Bar dataKey="accuracy" fill="#22c55e" />
//             </BarChart>
//           </ResponsiveContainer>

//           {/* Probability Chart */}
//           <h3 className="text-xl font-semibold mt-4 text-black">High/Low Probability per Model</h3>
//           <ResponsiveContainer width="100%" height={200}>
//             <BarChart data={probChartData}margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
//               <XAxis dataKey="model" />
//               <YAxis />
//               <Tooltip />
//               <Bar dataKey="probability" fill="#60a5fa" />
//             </BarChart>
//           </ResponsiveContainer>
//         </div>
//       )}
//     </div>
//   );
// }
"use client";

import { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function Home() {
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:5000";

  const [formData, setFormData] = useState({
    Crop: "Cotton",
    Region: "West",
    Rainfall_mm: "",
    Temperature_Celsius: "",
    Irrigation_Used: "0", // Default to No (0)
    Fertilizer_Used: "0", // Default to No (0)
  });

  const crops = ["Cotton", "Rice", "Barley", "Soybean", "Wheat", "Maize"];
  const regions = ["West", "East", "North", "South"];

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.Rainfall_mm || !formData.Temperature_Celsius) {
      setError("Rainfall and Temperature values are required to run prediction models.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const res = await axios.post(`${apiBaseUrl}/predict`, {
        Crop: formData.Crop,
        Region: formData.Region,
        Rainfall_mm: parseFloat(formData.Rainfall_mm),
        Temperature_Celsius: parseFloat(formData.Temperature_Celsius),
        Irrigation_Used: parseInt(formData.Irrigation_Used, 10),
        Fertilizer_Used: parseInt(formData.Fertilizer_Used, 10),
      });
      setResult(res.data);
    } catch (err: any) {
      console.error(err);
      setError("Failed to communicate with prediction server. Verify backend is active on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  // Convert model probabilities into individual datasets
  const lowData = result && result.model_probabilities
    ? Object.entries(result.model_probabilities).map(([model, probs]: any) => ({
        model: model === "DecisionTree" ? "Decision Tree" : model === "RandomForest" ? "Random Forest" : model,
        probability: probs.Low !== null ? parseFloat((probs.Low * 100).toFixed(1)) : 0,
      }))
    : [];

  const mediumData = result && result.model_probabilities
    ? Object.entries(result.model_probabilities).map(([model, probs]: any) => ({
        model: model === "DecisionTree" ? "Decision Tree" : model === "RandomForest" ? "Random Forest" : model,
        probability: probs.Medium !== null ? parseFloat((probs.Medium * 100).toFixed(1)) : 0,
      }))
    : [];

  const highData = result && result.model_probabilities
    ? Object.entries(result.model_probabilities).map(([model, probs]: any) => ({
        model: model === "DecisionTree" ? "Decision Tree" : model === "RandomForest" ? "Random Forest" : model,
        probability: probs.High !== null ? parseFloat((probs.High * 100).toFixed(1)) : 0,
      }))
    : [];

  const modelAccuracyInfo = [
    { name: "Random Forest", accuracy: "81.94%", type: "Ensemble Classifier", color: "bg-emerald-600 dark:bg-emerald-500" },
    { name: "Decision Tree", accuracy: "77.07%", type: "Tree Classifier", color: "bg-emerald-400" },
    { name: "KNN", accuracy: "76.77%", type: "Instance Classifier", color: "bg-zinc-400" },
    { name: "SVM", accuracy: "63.78%", type: "Kernel Classifier", color: "bg-zinc-600" },
  ];

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors duration-200">
      
      {/* Sleek top border accent */}
      <div className="h-1 bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-700"></div>

      {/* Header */}
      <header className="border-b border-zinc-200 dark:border-zinc-900 bg-white dark:bg-zinc-900/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            {/* Geometric Leaf Sprout Vector */}
            <svg className="w-5.5 h-5.5 text-emerald-700 dark:text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22C12 22 20 18 20 12C20 9 18 7 15 7C13.5 7 12 8 12 9.5C12 8 10.5 7 9 7C6 7 4 9 4 12C4 18 12 22 12 22Z" />
              <path d="M12 12V22" />
            </svg>
            <span className="font-semibold text-lg tracking-tight text-zinc-900 dark:text-white">
              CropPredict
            </span>
          </div>
          <div className="text-[11px] bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 font-medium uppercase tracking-widest px-3 py-1 rounded border border-zinc-200 dark:border-zinc-700">
            SML Benchmarks
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Title area */}
        <div className="mb-10 max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
            Crop Yield Predictor
          </h1>
          <p className="mt-2 text-zinc-500 dark:text-zinc-400 text-sm sm:text-base">
            Enter local environmental parameters and farming inputs to compare classification probability arrays across multiple supervised ML algorithms.
          </p>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="mb-8 p-4 rounded-lg bg-red-50 border border-red-200/60 dark:bg-red-950/20 dark:border-red-900/60 text-red-700 dark:text-red-400 flex items-center space-x-3 text-xs sm:text-sm">
            <svg className="w-5 h-5 flex-shrink-0 text-red-600 dark:text-red-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT: Configuration & Details */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Input Form Card */}
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-sm p-6">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-6">
                Parameter Configuration
              </h2>

              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Crop & Region Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Crop Type
                    </label>
                    <select
                      name="Crop"
                      value={formData.Crop}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    >
                      {crops.map((crop) => (
                        <option key={crop} value={crop}>{crop}</option>
                      ))}
                    </select>
                  </div>

                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Region
                    </label>
                    <select
                      name="Region"
                      value={formData.Region}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    >
                      {regions.map((region) => (
                        <option key={region} value={region}>{region}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Rainfall & Temperature Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Rainfall (mm)
                    </label>
                    <input
                      type="number"
                      step="any"
                      name="Rainfall_mm"
                      placeholder="800"
                      value={formData.Rainfall_mm}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-300 dark:placeholder-zinc-700 focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    />
                  </div>

                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Temperature (°C)
                    </label>
                    <input
                      type="number"
                      step="any"
                      name="Temperature_Celsius"
                      placeholder="25"
                      value={formData.Temperature_Celsius}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white placeholder-zinc-300 dark:placeholder-zinc-700 focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    />
                  </div>
                </div>

                {/* Yes / No Dropdowns */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Irrigation Used?
                    </label>
                    <select
                      name="Irrigation_Used"
                      value={formData.Irrigation_Used}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>

                  <div className="flex flex-col">
                    <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Fertilizer Used?
                    </label>
                    <select
                      name="Fertilizer_Used"
                      value={formData.Fertilizer_Used}
                      onChange={handleChange}
                      className="border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-white focus:outline-none focus:border-zinc-400 dark:focus:border-zinc-600 text-sm transition-all"
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full mt-4 bg-zinc-900 dark:bg-zinc-100 hover:bg-zinc-800 dark:hover:bg-zinc-200 text-white dark:text-zinc-900 font-medium py-2.5 px-4 rounded-lg shadow-sm active:scale-[0.99] transition-all disabled:opacity-50 cursor-pointer flex items-center justify-center space-x-2 text-sm"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-4.5 w-4.5 text-zinc-400 dark:text-zinc-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Evaluating models...</span>
                    </>
                  ) : (
                    <span>Predict Yield</span>
                  )}
                </button>
              </form>
            </div>

            {/* Project Overview Card */}
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-3">
                Project Overview
              </h3>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed mb-4">
                CropPredict is a Supervised Machine Learning benchmarking platform. It trains and evaluates K-Nearest Neighbors, SVM, Decision Tree, and Random Forest algorithms on a 1-million row dataset. The backend calculates classification probability matrices and estimates exact crop yields using a linear regression model.
              </p>
              
              <div className="text-xs mb-4">
                <span className="text-zinc-400 dark:text-zinc-500 font-semibold uppercase tracking-wider text-[9px] block mb-1">
                  Dataset Source
                </span>
                <a 
                  href="https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield" 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="text-emerald-600 dark:text-emerald-400 hover:underline inline-flex items-center font-medium"
                >
                  Kaggle Agriculture Crop Yield
                  <svg className="w-3 h-3 ml-1 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </div>

              <div className="h-px bg-zinc-100 dark:bg-zinc-800 my-4"></div>

              <h4 className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                Baseline Model Accuracy
              </h4>
              <div className="space-y-3">
                {modelAccuracyInfo.map((model) => (
                  <div key={model.name} className="flex items-center justify-between text-xs">
                    <span className="font-medium text-zinc-700 dark:text-zinc-300">{model.name}</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-zinc-400 dark:text-zinc-500 text-[10px]">{model.type}</span>
                      <span className="font-bold text-zinc-900 dark:text-white px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800">
                        {model.accuracy}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>

          {/* RIGHT: Results & Graphs Panel */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* If no result yet, show dynamic placeholder */}
            {!result ? (
              <div className="border border-zinc-200 dark:border-zinc-800 rounded-xl p-12 text-center flex flex-col items-center justify-center h-full min-h-[580px] text-zinc-400 dark:text-zinc-600 bg-white dark:bg-zinc-900">
                {/* Minimalist Graphic Icon instead of Emoji */}
                <svg className="w-10 h-10 text-zinc-300 dark:text-zinc-700 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 3v18h18" />
                  <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
                </svg>
                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Ready to Predict
                </h3>
                <p className="max-w-xs mt-1.5 text-xs text-zinc-400 dark:text-zinc-500 leading-relaxed">
                  Enter local farming values on the left and trigger prediction to benchmark classifiers and view regression estimations.
                </p>
              </div>
            ) : (
              <>
                {/* 1. Main Yield Prediction Summary Card */}
                <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-6">
                  <div>
                    <h3 className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1">
                      Regression Estimation
                    </h3>
                    <div className="flex items-baseline space-x-2">
                      <span className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-white">
                        {result.exact_yield.toFixed(2)}
                      </span>
                      <span className="text-zinc-400 dark:text-zinc-500 text-xs font-medium">
                        tons / hectare
                      </span>
                    </div>
                  </div>

                  <div className="h-px sm:h-12 w-full sm:w-px bg-zinc-200 dark:bg-zinc-800"></div>

                  <div>
                    <h3 className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-1.5">
                      Classification Vote
                    </h3>
                    <div>
                      <span className={`px-3 py-1 rounded text-xs font-semibold uppercase tracking-wider border ${
                        result.predicted_class === "High"
                          ? "bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900"
                          : result.predicted_class === "Medium"
                          ? "bg-zinc-50 dark:bg-zinc-800/40 text-zinc-700 dark:text-zinc-300 border-zinc-200 dark:border-zinc-700"
                          : "bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900"
                      }`}>
                        {result.predicted_class} Yield
                      </span>
                    </div>
                  </div>
                </div>

                {/* 2. Low Yield Probability Chart */}
                <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-4">
                    Low Yield Probability
                  </h3>
                  <div className="h-[180px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={lowData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" className="dark:hidden" />
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" className="hidden dark:block" />
                        <XAxis dataKey="model" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#888888" fontSize={10} tickLine={false} axisLine={false} unit="%" />
                        <Tooltip contentStyle={{ borderRadius: '6px', background: '#1e293b', color: '#ffffff', border: 'none', fontSize: '11px' }} />
                        <Bar dataKey="probability" fill="#e11d48" radius={[2, 2, 0, 0]} name="Low Yield (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 3. Medium Yield Probability Chart */}
                <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-4">
                    Medium Yield Probability
                  </h3>
                  <div className="h-[180px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={mediumData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" className="dark:hidden" />
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" className="hidden dark:block" />
                        <XAxis dataKey="model" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#888888" fontSize={10} tickLine={false} axisLine={false} unit="%" />
                        <Tooltip contentStyle={{ borderRadius: '6px', background: '#1e293b', color: '#ffffff', border: 'none', fontSize: '11px' }} />
                        <Bar dataKey="probability" fill="#d97706" radius={[2, 2, 0, 0]} name="Medium Yield (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 4. High Yield Probability Chart */}
                <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-4">
                    High Yield Probability
                  </h3>
                  <div className="h-[180px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={highData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" className="dark:hidden" />
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" className="hidden dark:block" />
                        <XAxis dataKey="model" stroke="#888888" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#888888" fontSize={10} tickLine={false} axisLine={false} unit="%" />
                        <Tooltip contentStyle={{ borderRadius: '6px', background: '#1e293b', color: '#ffffff', border: 'none', fontSize: '11px' }} />
                        <Bar dataKey="probability" fill="#059669" radius={[2, 2, 0, 0]} name="High Yield (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

          </div>

        </div>
      </main>
    </div>
  );
}

