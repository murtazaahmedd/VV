import React from "react";
import "./Navbar.css";

const Navbar = () => {
  return (
    <div className="navbar">
      <h1 className="logo">VigilantVision</h1>
      <div className="nav-links">
        <button>Home</button>
        <button>My Library</button>
        <button>Alerts</button>
        <button>Settings</button>
      </div>
    </div>
  );
};

export default Navbar;
