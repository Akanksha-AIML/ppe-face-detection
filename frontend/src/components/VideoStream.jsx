import React, { useEffect, useState } from "react";

// const BASE_URL = "http://192.168.1.232:8000";
const BASE_URL = "http://49.249.159.18:8000";



const VideoStream = () => {
  const [captures, setCaptures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCaptures();
    const interval = setInterval(fetchCaptures, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchCaptures = async () => {
    try {
      const response = await fetch(`${BASE_URL}/detection/get_image/`);
      const result = await response.json();

      let list = [];
      if (Array.isArray(result)) list = result;
      else if (Array.isArray(result.data)) list = result.data;
      else if (Array.isArray(result.images)) list = result.images;

      const formatted = list.map((item, index) => ({
        id: item.id || index,
        image_url:
          item.image_url ||
          (item.image ? `${BASE_URL}${item.image}` : "") ||
          (item.file ? `${BASE_URL}${item.file}` : ""),
        created_at: item.created_at || item.timestamp || ""
      }));

      setCaptures(formatted.reverse());
      setLoading(false);
    } catch (error) {
      console.error("Error fetching captures:", error);
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: "linear-gradient(135deg, #eef2f7, #f9fbfd)",
        minHeight: "100vh",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
      }}
    >
      {/* ================= RESPONSIVE STYLES ================= */}
      <style>
        {`
          .main-content {
            display: flex;
            gap: 30px;
            padding: 30px 40px;
          }

          @media (max-width: 1024px) {
            .main-content {
              flex-direction: column;
              padding: 20px;
            }

            .snapshot-panel {
              width: 100% !important;
            }

            .live-image {
              max-width: 100% !important;
            }
          }
        `}
      </style>

      {/* ================= HEADER ================= */}
      <div
        style={{
          background: "#0f172a",
          padding: "18px 40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          color: "#fff"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* 🔹 LOGO REPLACED HERE */}
          <img
            src="https://media.licdn.com/dms/image/v2/C4D0BAQEi4cevcx6xiw/company-logo_200_200/company-logo_200_200/0/1631309511072?e=1767830400&v=beta&t=66DAMI5BWCQZwlxNZu7WHIHzwUh7WdECTtSmWNky-7s"
            alt="Company Logo"
            style={{
              width: "42px",
              height: "42px",
              borderRadius: "10px",
              objectFit: "contain",
              background: "#ffffff",
              padding: "4px"
            }}
          />

          <div>
            <div style={{ fontSize: "18px", fontWeight: 600 }}>
              Smart Vision Monitoring
            </div>
            <div style={{ fontSize: "12px", color: "#94a3b8" }}>
              Real-Time Object Detection Dashboard
            </div>
          </div>
        </div>

        <div
          style={{
            padding: "6px 14px",
            background: "#16a34a",
            borderRadius: "20px",
            fontSize: "12px",
            fontWeight: 500
          }}
        >
          ● Live
        </div>
      </div>

      {/* ================= MAIN CONTENT ================= */}
      <div className="main-content">
        {/* ================= LIVE STREAM ================= */}
        <div
          style={{
            background: "#fff",
            borderRadius: "14px",
            boxShadow: "0 10px 25px rgba(0,0,0,0.12)",
            padding: "22px",
            flex: 1
          }}
        >
          <h2 style={{ marginBottom: "6px", color: "#1e293b" }}>
            Live Camera Feed
          </h2>
          <p style={{ marginBottom: "16px", color: "#64748b", fontSize: "14px" }}>
            Real-time YOLO based object detection with continuous monitoring
          </p>

          <img
            src={`${BASE_URL}/detection/frame`}
            alt="Live Stream"
            className="live-image"
            style={{
              width: "100%",
              maxWidth: "760px",
              borderRadius: "12px",
              border: "1px solid #e5e7eb"
            }}
          />
        </div>

        {/* ================= CAPTURE LIST ================= */}
        <div
          className="snapshot-panel"
          style={{
            background: "#fff",
            borderRadius: "14px",
            boxShadow: "0 10px 25px rgba(0,0,0,0.12)",
            padding: "22px",
            width: "340px",
            display: "flex",
            flexDirection: "column"
          }}
        >
          <h3 style={{ marginBottom: "6px", color: "#1e293b" }}>
            Detected Snapshots
          </h3>
          <p style={{ marginBottom: "14px", color: "#64748b", fontSize: "13px" }}>
            Automatically captured frames during detection events
          </p>

          {loading ? (
            <p style={{ color: "#94a3b8" }}>Loading snapshots...</p>
          ) : captures.length === 0 ? (
            <p style={{ color: "#94a3b8" }}>No snapshots available</p>
          ) : (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "14px",
                overflowY: "auto"
              }}
            >
              {captures.map((item) => (
                <div
                  key={item.id}
                  style={{
                    border: "1px solid #e5e7eb",
                    borderRadius: "10px",
                    padding: "8px"
                  }}
                >
                  <img
                    src={item.image_url}
                    alt="Capture"
                    style={{
                      width: "100%",
                      borderRadius: "8px",
                      marginBottom: "6px"
                    }}
                  />

                  {item.created_at && (
                    <div
                      style={{
                        fontSize: "11px",
                        color: "#64748b",
                        textAlign: "right"
                      }}
                    >
                      Captured at: {item.created_at}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoStream;
