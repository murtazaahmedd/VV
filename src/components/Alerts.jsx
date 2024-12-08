import React, { useEffect, useState } from "react";
import "./Alerts.css";

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/alerts"); // Ensure this is the correct API URL
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        setAlerts(data);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching alerts:", err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchAlerts();
  }, []);

  const getAlertStyle = (alertType) => {
    switch (alertType) {
      case "Crowd":
        return { backgroundColor: "rgba(255, 0, 0, 0.1)", color: "red" };
      case "Mask":
        return { backgroundColor: "rgba(255, 255, 0, 0.1)", color: "yellow" };
      case "Smoke":
        return { backgroundColor: "rgba(0, 0, 255, 0.1)", color: "blue" };
      case "Queue":
        return { backgroundColor: "rgba(0, 255, 0, 0.1)", color: "green" };
      default:
        return { backgroundColor: "rgba(0, 0, 0, 0.1)", color: "black" };
    }
  };

  if (loading) {
    return <div>Loading alerts...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="alerts">
      <h1>Alerts</h1>
      <div className="alerts-content">
        {alerts.length === 0 ? (
          <div>No alerts to show.</div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.camera_id}
              className="alert-item"
              style={getAlertStyle(alert.alert_type)} // Apply dynamic style based on alert type
            >
              <div className="alert-title">
                <strong>{alert.alert_type}</strong> alert for{" "}
                <strong>{alert.location_name}</strong>:{" "}
                <em>Detected {alert.detected_value} people</em>
              </div>
              <span className="alert-time">
                Timestamp: {new Date(alert.timestamp).toLocaleString()}
              </span>
              {alert.image && (
                <div className="alert-image">
                  <img
                    src={`data:image/jpeg;base64,${alert.image}`}
                    alt="Alert"
                    style={{ width: "200px", height: "auto" }}
                  />
                </div>
              )}
              
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Alerts;
