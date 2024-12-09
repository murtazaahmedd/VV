import React from "react";
import "./VideoLibrary.css";

const VideoLibrary = () => {
  const videos = [
    { id: 1, camera: "01", live: true },
    { id: 2, camera: "02", live: true },
    { id: 3, camera: "03", live: true },
    { id: 4, camera: "04", live: false },
  ];

  return (
    <div className="video-library">
      <h1>My Library</h1>
      <div className="video-grid">
        {videos.map((video) => (
          <div key={video.id} className="video-card">
            <p>Camera angle: {video.camera}</p>
            {video.live && <span className="live-badge">Live</span>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default VideoLibrary;
