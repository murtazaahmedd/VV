import React from "react";
import "./HomePage.css";  // Import the CSS file

const HomePage = () => {
  return (
    <div className="home-page">
      <h1>Vigilant Vision</h1>
      <p>Intelligent Surveillance and Anomaly Detection Proposal</p>
      <div className="footer">
        <p>Group Members: 
          <br></br>
          <a href="https://www.linkedin.com/in/sh-badar/" target="_blank" rel="noopener noreferrer">Shaheer Badar</a>
          <br></br>
          <a href="https://www.linkedin.com/in/maryam-sha-hid/" target="_blank" rel="noopener noreferrer">Maryam Shahid</a>
          <br></br>
          <a href="https://www.linkedin.com/in/murtazaahmed" target="_blank" rel="noopener noreferrer">Murtaza Ahmed</a>
        </p>
      </div>
    </div>
  );
};

export default HomePage;
