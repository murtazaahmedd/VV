import React, { useState } from "react";
import "./MyLibrary.css";

const MyLibrary = () => {
  const [selectedClasses, setSelectedClasses] = useState([]); // Track selected models
  const [confidenceValue, setConfidenceValue] = useState(0.0);
  const [selectedInputType, setSelectedInputType] = useState("");
  const [file, setFile] = useState(null);
  const [responseMessage, setResponseMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false); // Track if processing is active
  const [startLiveFeed, setStartLiveFeed] = useState(false);
  const [isLiveFeedRunning, setIsLiveFeedRunning] = useState(false);
  const [detectionThreshold, setDetectionThreshold] = useState(5); // New state for detection threshold

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
    setStartLiveFeed(false);
    setIsLiveFeedRunning(false); // Reset state on input change
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

    if (selectedInputType === "video") {
      const formData = new FormData();
      formData.append("file", file); // Add file if it's a video
      selectedClasses.forEach((cls) =>
        formData.append("models", cls.toLowerCase())
      );
      formData.append("threshold", detectionThreshold);

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
    }

    if (selectedInputType === "livefeed") {
      setStartLiveFeed(true); // Activate live feed processing
    }    
    setIsProcessing(false); // Set processing state to false after completion
  };

  const handleLiveFeedToggle = async () => {
    const action = isLiveFeedRunning ? "stop" : "start";

    const response = await fetch("http://localhost:5000/webcam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        models: selectedClasses.map((cls) => cls.toLowerCase()),
        threshold: detectionThreshold,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      setIsLiveFeedRunning(!isLiveFeedRunning);
      alert(data.message);
    } else {
      alert(`Error: ${data.error}`);
    }
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
          <option> Select Input</option>
          <option value="video">Video</option>
          <option value="livefeed">Live Feed</option>
        </select>
      </div>
      <div className="threshold-section">
        <label>Detection Threshold for Crowd Counting:</label>
        <input
          type="number"
          value={detectionThreshold}
          onChange={(e) => setDetectionThreshold(e.target.value)}
          min="1"
          step="1"
        />
      </div>      
      {selectedInputType === "video" && (
          <div className="classes-section">
            <input type="file" accept="video/*" onChange={handleFileChange} />
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
        )}      
      {selectedInputType === "livefeed" && (
        <>
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
        </>
      )}

      {selectedInputType === "livefeed" && selectedClasses.length > 0 && (
      <div className="livefeed-container">
          <img
            src={`http://localhost:5000/webcam?models=${selectedClasses
            .map((cls) => cls.toLowerCase())
            .join(",")}`}
            style={{ width: "100%", border: "1px solid black" }}
          />
      </div>
)}


      <button
        className="start-processing-button"
        onClick={selectedInputType==="livefeed"? handleLiveFeedToggle : handleStartProcessing}
      >
        {isLiveFeedRunning ? "Processing..." : "Start Processing"}
      </button>

      {responseMessage && <div className="response-message">{responseMessage}</div>}
    </div>
  );
};

export default MyLibrary;