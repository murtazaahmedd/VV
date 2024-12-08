import React, { useState } from "react";
import "./MyLibrary.css";

const MyLibrary = () => {
  const [selectedClasses, setSelectedClasses] = useState([]); // Track selected models
  const [confidenceValue, setConfidenceValue] = useState(0.0);
  const [selectedInputType, setSelectedInputType] = useState("");
  const [file, setFile] = useState(null);
  const [responseMessage, setResponseMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false); // Track if processing is active

  const models = ["Crowd", "Queue", "Smoke", "Mask"]; // Define all models

  const handleClassToggle = (className) => {
    if (selectedClasses.includes(className)) {
      setSelectedClasses(selectedClasses.filter((cls) => cls !== className));
    } else {
      setSelectedClasses([...selectedClasses, className]);
    }
  };

  const selectAllModels = () => {
    setSelectedClasses(models); // Select all models
  };

  const handleSliderChange = (event) => {
    setConfidenceValue(event.target.value);
  };

  const handleInputTypeChange = (event) => {
    setSelectedInputType(event.target.value);
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleStartProcessing = async () => {
    if (selectedInputType === "livefeed" && selectedClasses.length === 0) {
      alert("Please select at least one model for live feed processing.");
      return;
    }

    if (selectedInputType === "video" && !file) {
      alert("Please upload a video file.");
      return;
    }

    setIsProcessing(true); // Set processing state to true

    const formData = new FormData();
    if (file) formData.append("file", file); // Add file if it's a video
    selectedClasses.forEach((cls) => formData.append("models", cls.toLowerCase()));

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setResponseMessage(`Processing completed: ${JSON.stringify(data.results)}`);
      } else {
        setResponseMessage(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error("Error connecting to the backend:", error);
      setResponseMessage("Error connecting to the backend.");
    }
    setIsProcessing(false); // Set processing state to false after completion
  };

  return (
    <div className="my-library-container">
      <div className="input-section">
        <div className="input-label">Input</div>
        <select
          className="dropdown"
          value={selectedInputType}
          onChange={handleInputTypeChange}
        >
          <option value="">Select Input</option>
          <option value="video">Video</option>
          <option value="image">Image</option>
          <option value="livefeed">Live Feed</option>
        </select>

        {selectedInputType === "video" && (
          <div className="file-upload">
            <input type="file" accept="video/*" onChange={handleFileChange} />
          </div>
        )}

        <div className="slider-section">
          <label className="slider-label">Confidence Interval</label>
          <input
            type="range"
            className="slider"
            min="0.0"
            max="1.0"
            step="0.01"
            value={confidenceValue}
            onChange={handleSliderChange}
          />
          <div className="slider-value">{confidenceValue}</div>
        </div>
      </div>

      <div className="classes-section">
        <label className="slider-label">Select Models:</label>
        <div className="class-buttons">
          {models.map((className) => (
            <button
              key={className}
              className={`class-button ${
                selectedClasses.includes(className) ? "selected" : ""
              }`}
              onClick={() => handleClassToggle(className)}
            >
              {className}
            </button>
          ))}
          <button className="select-all-button" onClick={selectAllModels}>
            Select All
          </button>
        </div>
      </div>

      {selectedInputType === "livefeed" && selectedClasses.length > 0 && (
  <div className="livefeed-container">
    <h3>Live Camera Feed with Detections</h3>
    <img
      src={`http://localhost:5000/webcam?models=${selectedClasses
        .map((cls) => cls.toLowerCase())
        .join(",")}`}
      alt="Live Feed"
      style={{ width: "100%", border: "1px solid black" }}
    />
  </div>
)}


      <button
        className="start-processing-button"
        onClick={handleStartProcessing}
        disabled={!selectedInputType || isProcessing}
      >
        {isProcessing ? "Processing..." : "Start Processing"}
      </button>

      {responseMessage && <div className="response-message">{responseMessage}</div>}
    </div>
  );
};

export default MyLibrary;
