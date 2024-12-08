import React from "react";
import { Link } from "react-router-dom";
import "./Sidebar.css";
import logo from "./vv.jpeg"; // Import your logo image

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="logo-container">
        <img src={logo} alt="Vigilant Vision Logo" className="sidebar-logo" />
      </div>
      <div className="admin-panel">
        
      </div>
      <ul className="menu">
        <li><Link to="/" className="menu-link">Home</Link></li>
        <li><Link to="/my-library" className="menu-link">My Library</Link></li>
        <li><Link to="/alerts" className="menu-link">Alerts</Link></li>
        <li><Link to="/settings" className="menu-link">Settings</Link></li>
      </ul>
    </div>
  );
};

export default Sidebar;
