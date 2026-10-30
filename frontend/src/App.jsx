import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Bell,
  Camera,
  Check,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cpu,
  Download,
  Eye,
  FileText,
  Filter,
  History,
  Info,
  LayoutDashboard,
  Menu,
  Monitor,
  Play,
  RefreshCw,
  RotateCcw,
  ScanEye,
  Search,
  Settings,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Trash2,
  Upload,
  Video,
  VideoOff,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import "./index.css";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  // Navigation & Layout
  const [activePage, setActivePage] = useState("Dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);

  // File Upload & Drag-and-Drop
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [fileDimensions, setFileDimensions] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Assessment Engine State
  const [assessment, setAssessment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("overlay"); // "overlay" | "original"

  // Samples & Cameras
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [cameras, setCameras] = useState([]);

  // History & Filters
  const [history, setHistory] = useState([]);
  const [historyFilter, setHistoryFilter] = useState("ALL");
  const [historySearch, setHistorySearch] = useState("");

  // Notifications System
  const [notifications, setNotifications] = useState(() => {
    try {
      const saved = localStorage.getItem("cqg_notifications");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [showNotifications, setShowNotifications] = useState(false);

  // Live Webcam Monitor State
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamLoading, setWebcamLoading] = useState(false);
  const [webcamDevices, setWebcamDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [continuousGuard, setContinuousGuard] = useState(false);
  const [guardLastScore, setGuardLastScore] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const guardTimerRef = useRef(null);

  // Configuration
  const [config, setConfig] = useState(null);
  const [configSavedToast, setConfigSavedToast] = useState(false);

  // Save notifications to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("cqg_notifications", JSON.stringify(notifications));
    } catch {
      // ignore
    }
  }, [notifications]);

  // Periodic Backend Health Check
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          setApiOnline(data.status === "ok");
        } else {
          setApiOnline(false);
        }
      } catch {
        setApiOnline(false);
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 12000);
    return () => clearInterval(interval);
  }, []);

  // Fetch initial data
  useEffect(() => {
    fetchSamples();
    fetchHistory();
    fetchCameras();
    fetchConfig();
    enumerateCameras();
  }, []);

  // Cleanup webcam stream on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  // Handle continuous guardian timer
  useEffect(() => {
    if (continuousGuard && webcamActive) {
      guardTimerRef.current = setInterval(() => {
        captureAndAssessWebcam(false); // background continuous audit
      }, 4500);
    } else {
      if (guardTimerRef.current) {
        clearInterval(guardTimerRef.current);
        guardTimerRef.current = null;
      }
    }
    return () => {
      if (guardTimerRef.current) clearInterval(guardTimerRef.current);
    };
  }, [continuousGuard, webcamActive]);

  async function fetchSamples() {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
      }
    } catch {
      // offline fallback
    }
  }

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch {
      // offline fallback
    }
  }

  async function fetchCameras() {
    try {
      const res = await fetch(`${API_BASE}/api/cameras`);
      if (res.ok) {
        const data = await res.json();
        setCameras(data.cameras || []);
      }
    } catch {
      // offline fallback
    }
  }

  async function fetchConfig() {
    try {
      const res = await fetch(`${API_BASE}/api/config`);
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
      }
    } catch {
      // offline fallback
    }
  }

  // Enumerate video devices
  async function enumerateCameras() {
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setWebcamDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      } catch (err) {
        console.warn("Could not enumerate media devices:", err);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Notification Helpers
  // ---------------------------------------------------------------------------
  function addNotification({ type, title, message, assessmentData }) {
    const newNotif = {
      id: "notif_" + Date.now() + "_" + Math.random().toString(36).substr(2, 5),
      timestamp: new Date().toISOString(),
      type, // "GOOD" | "WARNING" | "POOR" | "INFO"
      title,
      message,
      read: false,
      data: assessmentData || null,
    };
    setNotifications((prev) => [newNotif, ...prev.slice(0, 49)]);
  }

  function markAllNotificationsRead() {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }

  function clearNotifications() {
    setNotifications([]);
  }

  function inspectNotificationItem(item) {
    if (item.data) {
      setAssessment(item.data);
      setSelectedSampleId(null);
      setSelectedFile(null);
      setFilePreview(item.data.original_image || null);
      setActivePage("Dashboard");
      setShowNotifications(false);
    } else if (item.data?.id) {
      inspectAssessmentById(item.data.id);
    }
  }

  // ---------------------------------------------------------------------------
  // File Validation & Handling
  // ---------------------------------------------------------------------------
  function validateAndSetFile(file) {
    if (!file) return;

    setError("");
    const allowed = [".jpg", ".jpeg", ".png"];
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();

    if (!allowed.includes(ext)) {
      setError(`Unsupported format (${ext}). Please upload JPG, JPEG, or PNG.`);
      return;
    }

    if (file.size === 0) {
      setError("The selected file is empty (0 bytes).");
      return;
    }

    const maxSize = 25 * 1024 * 1024; // 25 MB limit
    if (file.size > maxSize) {
      setError("File exceeds maximum allowed size (25MB).");
      return;
    }

    setSelectedFile(file);
    setSelectedSampleId(null);

    const objectUrl = URL.createObjectURL(file);
    setFilePreview(objectUrl);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setFileDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    validateAndSetFile(file);
    // Reset file input value so re-selecting same file triggers onChange
    event.target.value = "";
  }

  function handleRemoveFile() {
    setSelectedFile(null);
    setFilePreview(null);
    setFileDimensions(null);
    setError("");
  }

  function handleReplaceFile() {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  // Drag and Drop handlers
  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }

  function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  }

  // ---------------------------------------------------------------------------
  // Assessment API Calls
  // ---------------------------------------------------------------------------
  async function handleAssessUpload() {
    if (!selectedFile) {
      setError("Please select or drop an image file first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/api/assess/image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Image assessment failed on server.");
      }

      setAssessment(data);
      setViewMode("overlay");
      fetchHistory();

      // Trigger genuine assessment notification
      const passCount = data.checks_summary?.passed ?? 0;
      const failCount = data.checks_summary?.failed ?? 0;
      const issuesPreview = data.all_issues?.length > 0 ? data.all_issues[0] : "All 5 quality checks passed";

      addNotification({
        type: data.overall_status,
        title: `Assessment Completed: ${data.overall_status} (${Math.round(data.overall_score)}/100)`,
        message: `${data.filename} (${data.image_width}x${data.image_height}) • ${passCount} Passed, ${failCount} Failed. ${issuesPreview}`,
        assessmentData: data,
      });
    } catch (err) {
      setError(err.message || "Could not connect to the Python Quality Guardian API.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAssessSample(sampleName) {
    setLoading(true);
    setError("");
    setSelectedSampleId(sampleName);
    setSelectedFile(null);
    setFilePreview(null);
    setFileDimensions(null);

    try {
      const response = await fetch(`${API_BASE}/api/assess/sample/${sampleName}`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Failed to assess sample ${sampleName}`);
      }

      setAssessment(data);
      setFilePreview(data.original_image);
      setViewMode("overlay");
      fetchHistory();

      const passCount = data.checks_summary?.passed ?? 0;
      const failCount = data.checks_summary?.failed ?? 0;

      addNotification({
        type: data.overall_status,
        title: `Sample Audit: ${sampleName} → ${data.overall_status} (${Math.round(data.overall_score)}/100)`,
        message: `Evaluation completed • ${passCount} Passed, ${failCount} Failed • ${data.all_issues[0] || "Optimal operational quality"}`,
        assessmentData: data,
      });
    } catch (err) {
      setError(err.message || "Failed to analyze sample image.");
    } finally {
      setLoading(false);
    }
  }

  async function inspectAssessmentById(id) {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/assess/${id}`);
      if (res.ok) {
        const data = await res.json();
        setAssessment(data);
        setSelectedSampleId(null);
        setSelectedFile(null);
        setFilePreview(data.original_image);
        setActivePage("Dashboard");
      }
    } catch (err) {
      console.warn("Could not load cached assessment:", err);
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Live Webcam Integration
  // ---------------------------------------------------------------------------
  async function startWebcam() {
    setWebcamLoading(true);
    setError("");

    try {
      const constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setWebcamActive(true);
      enumerateCameras();
    } catch (err) {
      setError(`Webcam access failed: ${err.message}. Ensure camera permissions are granted.`);
      setWebcamActive(false);
    } finally {
      setWebcamLoading(false);
    }
  }

  function stopWebcam() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
    setContinuousGuard(false);
  }

  async function captureAndAssessWebcam(switchToDashboard = true) {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const filename = `webcam_frame_${Date.now()}.jpg`;
      const file = new File([blob], filename, { type: "image/jpeg" });

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE}/api/assess/image`, {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (response.ok) {
          setGuardLastScore(data.overall_score);

          if (switchToDashboard) {
            setAssessment(data);
            setSelectedFile(file);
            setFilePreview(URL.createObjectURL(blob));
            setActivePage("Dashboard");
            setViewMode("overlay");
          }

          fetchHistory();

          // Push genuine notification
          addNotification({
            type: data.overall_status,
            title: `Webcam Snapshot: ${data.overall_status} (${Math.round(data.overall_score)}/100)`,
            message: `Live camera frame checked • ${data.checks_summary?.passed ?? 0} Passed, ${data.checks_summary?.failed ?? 0} Failed • ${data.all_issues[0] || "Live feed healthy"}`,
            assessmentData: data,
          });
        }
      } catch (err) {
        console.warn("Continuous guardian capture failed:", err);
      }
    }, "image/jpeg", 0.90);
  }

  // ---------------------------------------------------------------------------
  // History & Config Helpers
  // ---------------------------------------------------------------------------
  async function handleClearHistory() {
    try {
      await fetch(`${API_BASE}/api/history`, { method: "DELETE" });
      setHistory([]);
    } catch {
      // ignore
    }
  }

  function handleDownloadReport(item) {
    const reportData = assessment && assessment.id === item?.id ? assessment : item;
    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `camera_quality_report_${item?.filename || "assessment"}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleSaveConfig(event) {
    event.preventDefault();
    if (!config) return;

    try {
      const res = await fetch(`${API_BASE}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        setConfigSavedToast(true);
        setTimeout(() => setConfigSavedToast(false), 3500);
      }
    } catch {
      // offline
    }
  }

  async function handleResetConfig() {
    try {
      const res = await fetch(`${API_BASE}/api/config/reset`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setConfig(data.config);
        setConfigSavedToast(true);
        setTimeout(() => setConfigSavedToast(false), 3500);
      }
    } catch {
      // offline
    }
  }

  // Navigation Items
  const unreadCount = notifications.filter((n) => !n.read).length;

  const navigation = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Assessments", icon: History, count: history.length },
    { name: "Camera Monitor", icon: Video, live: webcamActive },
    { name: "Settings", icon: Settings },
  ];

  // Helper for check status determination
  function getCheckStatusBadge(checkObj) {
    if (!checkObj) return { text: "PENDING", className: "badge-pending" };
    if (checkObj.passed) return { text: "PASS", className: "badge-pass" };
    if (checkObj.uncertain) return { text: "UNCERTAIN", className: "badge-uncertain" };
    return { text: "FAIL", className: "badge-fail" };
  }

  function getObstructionBadge(obstructionObj) {
    if (!obstructionObj) return { text: "PENDING", className: "badge-pending" };
    if (obstructionObj.status === "CLEAR") return { text: "PASS (CLEAR)", className: "badge-pass" };
    if (obstructionObj.status === "UNCERTAIN") return { text: "UNCERTAIN", className: "badge-uncertain" };
    return { text: "FAIL (OBSTRUCTED)", className: "badge-fail" };
  }

  return (
    <div className="app">
      {/* Mobile Sidebar Backdrop */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Industrial Left Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">
            <ScanEye size={22} />
          </div>
          <div>
            <h2>Quality Guardian</h2>
            <span>CAMERA INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.name;

            return (
              <button
                key={item.name}
                className={`nav-item ${isActive ? "active" : ""}`}
                onClick={() => {
                  setActivePage(item.name);
                  setSidebarOpen(false);
                }}
              >
                <Icon size={18} />
                <span>{item.name}</span>
                {item.live && <span className="nav-pulse-dot" title="Live stream active" />}
                {item.count !== undefined && item.count > 0 && (
                  <span className="nav-badge">{item.count}</span>
                )}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className={`status-dot ${apiOnline ? "online" : "offline"}`} />
            <div>
              <strong>{apiOnline ? "FastAPI Online" : "Backend Disconnected"}</strong>
              <small>{apiOnline ? "Python CV Engine Ready" : "Port 8000 unreachable"}</small>
            </div>
          </div>
          <p>CAMERA QUALITY GUARDIAN • v1.0 INDUSTRIAL</p>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="hamburger-btn"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle menu"
            >
              <Menu size={20} />
            </button>
            <div className="breadcrumb">
              <span>Workspace</span>
              <span>/</span>
              <strong>{activePage}</strong>
            </div>
          </div>

          <div className="top-actions">
            <div className="engine-chip">
              <Cpu size={14} />
              <span>OpenCV • Rule-Based CV</span>
            </div>

            {/* Notification Bell with Badge */}
            <div className="notification-wrapper">
              <button
                className={`icon-button ${unreadCount > 0 ? "has-unread" : ""}`}
                aria-label="Notifications"
                title="Assessment alerts"
                onClick={() => {
                  setShowNotifications(!showNotifications);
                  if (unreadCount > 0) markAllNotificationsRead();
                }}
              >
                <Bell size={18} />
                {unreadCount > 0 && <span className="notification-count">{unreadCount}</span>}
              </button>

              {/* Notifications Dropdown Drawer */}
              {showNotifications && (
                <div className="notifications-dropdown">
                  <div className="notif-header">
                    <div>
                      <strong>Assessment Alerts</strong>
                      <span className="notif-total">{notifications.length} records</span>
                    </div>
                    <div className="notif-header-actions">
                      <button className="text-link-btn" onClick={clearNotifications}>
                        Clear
                      </button>
                      <button className="icon-close-btn" onClick={() => setShowNotifications(false)}>
                        <X size={15} />
                      </button>
                    </div>
                  </div>

                  <div className="notif-list">
                    {notifications.length > 0 ? (
                      notifications.map((notif) => (
                        <div
                          key={notif.id}
                          className={`notif-item notif-${notif.type.toLowerCase()}`}
                          onClick={() => inspectNotificationItem(notif)}
                        >
                          <div className="notif-icon-col">
                            {notif.type === "GOOD" ? (
                              <CheckCircle2 size={16} className="text-lime" />
                            ) : notif.type === "WARNING" ? (
                              <AlertTriangle size={16} className="text-amber" />
                            ) : (
                              <XCircle size={16} className="text-red" />
                            )}
                          </div>
                          <div className="notif-body">
                            <div className="notif-title-row">
                              <strong>{notif.title}</strong>
                              <small>
                                {new Date(notif.timestamp).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </small>
                            </div>
                            <p>{notif.message}</p>
                            {notif.data && (
                              <span className="notif-inspect-hint">Click to inspect in Dashboard →</span>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="notif-empty">
                        <Bell size={24} className="text-muted" />
                        <p>No assessment alerts recorded yet.</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="profile" title="Operator Console">CQG</div>
          </div>
        </header>

        {/* Dynamic Page Content */}
        <div className="page-content">
          {/* TAB 1: DASHBOARD */}
          {activePage === "Dashboard" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">PRE-DETECTION QUALITY VERIFICATION</div>
                  <h1>Camera Quality Dashboard</h1>
                  <p>
                    Evaluate camera lighting, sharpness, resolution, glare, and lens obstruction before trusting downstream PPE detection models.
                  </p>
                </div>

                <div className={`live-indicator ${apiOnline ? "live" : "warn"}`}>
                  <span className={`status-dot ${apiOnline ? "online" : "offline"}`} />
                  {apiOnline ? "OpenCV Engine Ready" : "Backend Disconnected"}
                </div>
              </div>

              {/* Complete Drag-and-Drop Image Upload Panel */}
              <section className="welcome-panel">
                <div className="welcome-copy">
                  <span className="welcome-tag">FRAME INGESTION &amp; AUDIT</span>
                  <h2>Upload or Select Camera Frame</h2>
                  <p>
                    Industrial vision pre-flight check. Ensure lighting, optical focus, sensor resolution, and unobstructed view before feeding images to PPE detection models.
                  </p>

                  {/* Drag-and-Drop Zone */}
                  <div
                    className={`dropzone-box ${isDragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
                    onDragOver={handleDragOver}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !selectedFile && handleReplaceFile()}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: "none" }}
                      accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                      onChange={handleFileChange}
                    />

                    {!selectedFile ? (
                      <div className="dropzone-empty-state">
                        <div className="drop-icon-circle">
                          <Upload size={22} />
                        </div>
                        <div className="drop-text">
                          <strong>Drag &amp; drop camera image here</strong>
                          <span>or click to browse local files</span>
                        </div>
                        <div className="format-tags">
                          <span className="format-tag">JPG</span>
                          <span className="format-tag">JPEG</span>
                          <span className="format-tag">PNG</span>
                          <span className="format-tag">Max 25MB</span>
                        </div>
                      </div>
                    ) : (
                      <div className="file-selected-card" onClick={(e) => e.stopPropagation()}>
                        <div className="file-thumbnail">
                          {filePreview ? (
                            <img src={filePreview} alt="Selected preview" />
                          ) : (
                            <FileText size={24} />
                          )}
                        </div>
                        <div className="file-info-col">
                          <strong>{selectedFile.name}</strong>
                          <div className="file-meta-row">
                            <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                            {fileDimensions && (
                              <span>• {fileDimensions.width} × {fileDimensions.height} px</span>
                            )}
                            <span className="file-ready-tag">• Ready for evaluation</span>
                          </div>
                        </div>
                        <div className="file-card-actions">
                          <button
                            type="button"
                            className="btn-replace-file"
                            title="Replace image"
                            onClick={handleReplaceFile}
                          >
                            Replace
                          </button>
                          <button
                            type="button"
                            className="btn-remove-file"
                            title="Remove image"
                            onClick={handleRemoveFile}
                          >
                            <X size={15} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Assessment Actions Bar */}
                  <div className="launcher-actions-bar">
                    <button
                      className="primary-action-btn"
                      onClick={handleAssessUpload}
                      disabled={!selectedFile || loading}
                    >
                      {loading ? (
                        <>
                          <RefreshCw size={16} className="spin" />
                          Evaluating Frame with OpenCV...
                        </>
                      ) : (
                        <>
                          <Zap size={16} />
                          Run Quality Assessment
                        </>
                      )}
                    </button>

                    <button
                      className="secondary-btn"
                      onClick={() => setActivePage("Camera Monitor")}
                      title="Switch to live camera capture"
                    >
                      <Camera size={15} />
                      Capture from Live Webcam
                    </button>
                  </div>

                  {/* Instant Sample Test Selector */}
                  <div className="sample-selector-strip">
                    <span className="sample-label">Or Instant Sample Test:</span>
                    <div className="sample-pills">
                      {samples.map((s) => (
                        <button
                          key={s.id}
                          className={`sample-pill ${selectedSampleId === s.id ? "active" : ""}`}
                          onClick={() => handleAssessSample(s.id)}
                          disabled={loading}
                          title={s.description}
                        >
                          <span className="sample-name">{s.id}</span>
                          <span className={`pill-badge badge-${s.expected_status.toLowerCase()}`}>
                            {s.expected_status}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {error && (
                    <div className="error-banner">
                      <AlertTriangle size={15} />
                      <span>{error}</span>
                    </div>
                  )}
                </div>

                {/* Industrial Graphic Widget */}
                <div className="welcome-graphic">
                  <div className="graphic-ring ring-one" />
                  <div className="graphic-ring ring-two" />
                  <div className="graphic-center">
                    <Camera size={38} />
                    <span>5-POINT<br />ENGINE</span>
                  </div>
                  <div className="graphic-chip chip-top">DARKNESS • BLUR</div>
                  <div className="graphic-chip chip-bottom">GLARE • OBSTRUCTION</div>
                </div>
              </section>

              {/* Assessment Summary Section */}
              <div className="section-heading">
                <div>
                  <h2>Assessment Summary</h2>
                  <p>Quality scorecard, status breakdown, and downstream PPE readiness</p>
                </div>
                {assessment && (
                  <span className="timestamp-badge">
                    Evaluated: {new Date(assessment.processed_at).toLocaleTimeString()}
                  </span>
                )}
              </div>

              <section className="metric-grid">
                {/* 1. Overall Score Card */}
                <div
                  className={`metric-card score-card ${
                    assessment ? `status-${assessment.overall_status.toLowerCase()}` : ""
                  }`}
                >
                  <div className="metric-top">
                    <span>OVERALL QUALITY SCORE</span>
                    <Activity size={18} />
                  </div>
                  <div className="empty-score">
                    {assessment ? Math.round(assessment.overall_score) : "--"}
                    <small>/100</small>
                  </div>
                  <div className="score-status-row">
                    {assessment ? (
                      <span className={`status-pill pill-${assessment.overall_status.toLowerCase()}`}>
                        {assessment.overall_status}
                      </span>
                    ) : (
                      <span className="status-pill pill-neutral">PENDING AUDIT</span>
                    )}
                    <span className="metric-hint">
                      {assessment ? "≥ 80 GOOD • ≥ 50 WARNING • < 50 POOR" : "Run assessment to evaluate"}
                    </span>
                  </div>
                </div>

                {/* 2. Passed / Failed / Uncertain Checks Count Card */}
                <div className="metric-card checks-summary-card">
                  <div className="metric-top">
                    <span>CHECKS EVALUATION</span>
                    <CheckCircle2 size={18} />
                  </div>
                  <div className="metric-value">
                    {assessment?.checks_summary
                      ? `${assessment.checks_summary.passed} / ${assessment.checks_summary.total} Passed`
                      : "0 / 5 Evaluated"}
                  </div>
                  <div className="checks-badge-row">
                    {assessment?.checks_summary ? (
                      <>
                        <span className="mini-badge badge-pass">
                          <Check size={11} /> {assessment.checks_summary.passed} Passed
                        </span>
                        {assessment.checks_summary.failed > 0 && (
                          <span className="mini-badge badge-fail">
                            <X size={11} /> {assessment.checks_summary.failed} Failed
                          </span>
                        )}
                        {assessment.checks_summary.uncertain > 0 && (
                          <span className="mini-badge badge-uncertain">
                            <AlertTriangle size={11} /> {assessment.checks_summary.uncertain} Uncertain
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="text-muted-sm">Darkness • Blur • Resolution • Glare • Obstruction</span>
                    )}
                  </div>
                </div>

                {/* 3. Image Dimensions & File Card */}
                <div className="metric-card file-summary-card">
                  <div className="metric-top">
                    <span>IMAGE SPECIFICATIONS</span>
                    <Camera size={18} />
                  </div>
                  <div className="metric-value font-mono">
                    {assessment
                      ? `${assessment.image_width} × ${assessment.image_height}`
                      : "-- × -- px"}
                  </div>
                  <p className="file-name-line">
                    {assessment ? (
                      <>
                        <span className="text-lime">{assessment.filename}</span>
                        <span>• Source: {assessment.source_type}</span>
                      </>
                    ) : (
                      "Awaiting image ingestion"
                    )}
                  </p>
                </div>

                {/* 4. PPE Detection Readiness Verdict */}
                <div className="metric-card ppe-card">
                  <div className="metric-top">
                    <span>PPE DETECTION READINESS</span>
                    <ShieldCheck size={18} />
                  </div>
                  <div className="metric-value">
                    {assessment?.ppe_readiness ? (
                      <span className={`ppe-badge ppe-${assessment.ppe_readiness.badge.toLowerCase()}`}>
                        {assessment.ppe_readiness.badge}
                      </span>
                    ) : (
                      <span className="ppe-badge ppe-neutral">AWAITING AUDIT</span>
                    )}
                  </div>
                  <p>
                    {assessment?.ppe_readiness
                      ? assessment.ppe_readiness.message
                      : "Automated pre-flight check confirms whether downstream PPE models can be safely trusted."}
                  </p>
                </div>
              </section>

              {/* Visual Dual-View Frame Inspection Panel */}
              {assessment && (
                <section className="frame-inspection-panel">
                  <div className="inspection-header">
                    <div className="inspection-title">
                      <ScanEye size={18} />
                      <strong>Computer Vision Inspection Overlay</strong>
                      <span className="frame-meta">
                        {assessment.image_width} × {assessment.image_height} px • {assessment.filename}
                      </span>
                    </div>

                    <div className="view-mode-toggles">
                      <button
                        className={`toggle-btn ${viewMode === "overlay" ? "active" : ""}`}
                        onClick={() => setViewMode("overlay")}
                        disabled={!assessment.annotated_image}
                        title="View OpenCV visual diagnostic masks"
                      >
                        <Eye size={14} /> Quality Overlay (OpenCV)
                      </button>
                      <button
                        className={`toggle-btn ${viewMode === "original" ? "active" : ""}`}
                        onClick={() => setViewMode("original")}
                        title="View unmodified input frame"
                      >
                        <Camera size={14} /> Original Frame
                      </button>
                    </div>
                  </div>

                  <div className="image-display-frame">
                    {viewMode === "overlay" && assessment.annotated_image ? (
                      <img
                        src={assessment.annotated_image}
                        alt="Quality Annotated Frame"
                        className="inspected-image"
                      />
                    ) : (
                      <img
                        src={assessment.original_image || filePreview}
                        alt="Original Frame"
                        className="inspected-image"
                      />
                    )}
                  </div>

                  <div className="inspection-footer">
                    <span>
                      {viewMode === "overlay"
                        ? "Visual overlay shows detected glare regions (yellow), dark regions (blue), and status banner generated directly by OpenCV."
                        : "Raw unprocessed camera frame as ingested into the quality inspection engine."}
                    </span>
                    <button
                      className="text-action-btn"
                      onClick={() => handleDownloadReport(assessment)}
                    >
                      <Download size={14} /> Download JSON Audit Report
                    </button>
                  </div>
                </section>
              )}

              {/* Five Quality Parameter Cards Section */}
              <div className="section-heading checks-heading">
                <div>
                  <h2>Five Quality Parameter Cards</h2>
                  <p>Individual rule-based computer vision checks performed by the Python engine</p>
                </div>
              </div>

              <section className="checks-grid">
                {/* 1. Darkness Card */}
                {(() => {
                  const check = assessment?.checks?.darkness;
                  const status = getCheckStatusBadge(check);
                  return (
                    <div className={`check-card ${check ? (check.passed ? "card-pass" : "card-fail") : ""}`}>
                      <div className="check-icon">
                        <Activity size={20} />
                      </div>
                      <div className="check-info">
                        <div className="check-title-row">
                          <strong>Darkness Detection</strong>
                          <span className={`check-badge ${status.className}`}>
                            {status.text}
                          </span>
                        </div>
                        <span className="check-desc">
                          {check
                            ? `Mean intensity: ${check.value.toFixed(1)} (Threshold ≥ ${check.threshold}) • Dark pixels: ${((check.details?.dark_pixel_fraction || 0) * 100).toFixed(1)}%`
                            : "Evaluates scene illumination and underexposed pixel fraction."}
                        </span>
                        {check && (
                          <div className="check-metrics-strip">
                            <span className="penalty-tag">
                              {check.passed ? "0 pts penalty" : "-30.0 pts penalty"}
                            </span>
                            {check.issues?.length > 0 && (
                              <span className="check-issue-note">
                                <AlertCircle size={12} /> {check.issues[0]}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* 2. Blur Card */}
                {(() => {
                  const check = assessment?.checks?.blur;
                  const status = getCheckStatusBadge(check);
                  return (
                    <div
                      className={`check-card ${
                        check
                          ? check.passed
                            ? "card-pass"
                            : check.uncertain
                            ? "card-uncertain"
                            : "card-fail"
                          : ""
                      }`}
                    >
                      <div className="check-icon">
                        <ScanEye size={20} />
                      </div>
                      <div className="check-info">
                        <div className="check-title-row">
                          <strong>Blur &amp; Focus Check</strong>
                          <span className={`check-badge ${status.className}`}>
                            {status.text}
                          </span>
                        </div>
                        <span className="check-desc">
                          {check
                            ? `Laplacian variance: ${check.value.toFixed(1)} (Threshold ≥ ${check.threshold}) ${
                                check.uncertain ? "• Low scene lighting prevents reliable focus evaluation" : ""
                              }`
                            : "Measures edge gradient variance to detect lens defocus or motion blur."}
                        </span>
                        {check && (
                          <div className="check-metrics-strip">
                            <span className="penalty-tag">
                              {check.passed || check.uncertain ? "0 pts penalty" : "-25.0 pts penalty"}
                            </span>
                            {check.uncertain && (
                              <span className="check-issue-note text-amber">
                                <AlertTriangle size={12} /> Visual detail insufficient (no penalty)
                              </span>
                            )}
                            {!check.passed && !check.uncertain && check.issues?.length > 0 && (
                              <span className="check-issue-note">
                                <AlertCircle size={12} /> {check.issues[0]}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* 3. Resolution Card */}
                {(() => {
                  const check = assessment?.checks?.resolution;
                  const status = getCheckStatusBadge(check);
                  return (
                    <div className={`check-card ${check ? (check.passed ? "card-pass" : "card-fail") : ""}`}>
                      <div className="check-icon">
                        <Camera size={20} />
                      </div>
                      <div className="check-info">
                        <div className="check-title-row">
                          <strong>Resolution Check</strong>
                          <span className={`check-badge ${status.className}`}>
                            {status.text}
                          </span>
                        </div>
                        <span className="check-desc">
                          {check
                            ? `Frame size: ${assessment.image_width} × ${assessment.image_height} px (Minimum required: 640 × 480 px)`
                            : "Confirms image dimensions are sufficient for small PPE feature detection."}
                        </span>
                        {check && (
                          <div className="check-metrics-strip">
                            <span className="penalty-tag">
                              {check.passed ? "0 pts penalty" : "-20.0 pts penalty"}
                            </span>
                            {check.issues?.length > 0 && (
                              <span className="check-issue-note">
                                <AlertCircle size={12} /> {check.issues[0]}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* 4. Glare Card */}
                {(() => {
                  const check = assessment?.checks?.glare;
                  const status = getCheckStatusBadge(check);
                  return (
                    <div className={`check-card ${check ? (check.passed ? "card-pass" : "card-fail") : ""}`}>
                      <div className="check-icon">
                        <AlertTriangle size={20} />
                      </div>
                      <div className="check-info">
                        <div className="check-title-row">
                          <strong>Glare &amp; Overexposure</strong>
                          <span className={`check-badge ${status.className}`}>
                            {status.text}
                          </span>
                        </div>
                        <span className="check-desc">
                          {check
                            ? `Overexposed pixels: ${(check.value * 100).toFixed(1)}% (Threshold ≤ ${(check.threshold * 100).toFixed(1)}%)`
                            : "Identifies washed-out highlights and sensor saturation from direct glare."}
                        </span>
                        {check && (
                          <div className="check-metrics-strip">
                            <span className="penalty-tag">
                              {check.passed ? "0 pts penalty" : "-15.0 pts penalty"}
                            </span>
                            {check.issues?.length > 0 && (
                              <span className="check-issue-note">
                                <AlertCircle size={12} /> {check.issues[0]}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* 5. Obstruction Card */}
                {(() => {
                  const obs = assessment?.obstruction;
                  const status = getObstructionBadge(obs);
                  return (
                    <div
                      className={`check-card full-width-card ${
                        obs
                          ? obs.status === "CLEAR"
                            ? "card-pass"
                            : obs.status === "UNCERTAIN"
                            ? "card-uncertain"
                            : "card-fail"
                          : ""
                      }`}
                    >
                      <div className="check-icon">
                        <ShieldCheck size={20} />
                      </div>
                      <div className="check-info">
                        <div className="check-title-row">
                          <strong>Obstruction &amp; Lens Cover</strong>
                          <span className={`check-badge ${status.className}`}>
                            {status.text}
                          </span>
                        </div>
                        <span className="check-desc">
                          {obs
                            ? `Status: ${obs.status} (Confidence: ${(obs.confidence * 100).toFixed(0)}%) • Edge density: ${((obs.details?.edge_density || 0) * 100).toFixed(3)}% • Intensity variance: ${(obs.details?.intensity_std || 0).toFixed(1)}.`
                            : "Conservative heuristic evaluating edge density and variance to identify covered lenses."}
                        </span>
                        {obs && (
                          <div className="check-metrics-strip">
                            <span className="penalty-tag">
                              {obs.status === "OBSTRUCTED" ? "-10.0 pts penalty" : "0 pts penalty"}
                            </span>
                            <span className="check-note-text">
                              {obs.status === "UNCERTAIN"
                                ? "UNCERTAIN results carry zero score penalty; surfaced as operational advice."
                                : obs.note}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </section>

              {/* Diagnostic Issues & Actionable Recommendations */}
              {assessment && (
                <section className="diagnostics-panel">
                  <div className="diag-column">
                    <div className="diag-heading">
                      <AlertTriangle size={17} className="text-amber" />
                      <strong>Issues Detected ({assessment.all_issues.length})</strong>
                    </div>
                    {assessment.all_issues.length > 0 ? (
                      <ul className="diag-list issues-list">
                        {assessment.all_issues.map((issue, idx) => (
                          <li key={idx}>
                            <AlertCircle size={14} />
                            <span>{issue}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="diag-empty">
                        <CheckCircle2 size={16} className="text-lime" />
                        <span>No quality defects detected. Camera feed is optimal.</span>
                      </div>
                    )}
                  </div>

                  <div className="diag-column">
                    <div className="diag-heading">
                      <Info size={17} className="text-lime" />
                      <strong>Actionable Recommendations ({assessment.all_recommendations.length})</strong>
                    </div>
                    {assessment.all_recommendations.length > 0 ? (
                      <ul className="diag-list recs-list">
                        {assessment.all_recommendations.map((rec, idx) => (
                          <li key={idx}>
                            <ArrowRight size={14} />
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="diag-empty">
                        <CheckCircle2 size={16} className="text-lime" />
                        <span>No maintenance actions needed. Ready for operational PPE scanning.</span>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Score Breakdown Table */}
              {assessment && assessment.score_breakdown && (
                <section className="breakdown-card">
                  <div className="breakdown-header">
                    <strong>Score Deduction Breakdown (Starting baseline: 100.0 pts)</strong>
                    <span className="disclaimer-text">{assessment.disclaimer}</span>
                  </div>
                  <div className="breakdown-grid">
                    {Object.entries(assessment.score_breakdown).map(([checkKey, data]) => (
                      <div key={checkKey} className="breakdown-item">
                        <span className="breakdown-name">{checkKey.toUpperCase()}</span>
                        <span
                          className={`breakdown-penalty ${
                            data.penalty_applied > 0 ? "penalty-deducted" : "penalty-zero"
                          }`}
                        >
                          {data.penalty_applied > 0 ? `-${data.penalty_applied} pts` : "0 pts"}
                        </span>
                        <small className="breakdown-note">
                          {data.uncertain
                            ? "Uncertain (0 penalty)"
                            : data.failed || data.status === "OBSTRUCTED"
                            ? "Check failed"
                            : "Passed"}
                        </small>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}

          {/* TAB 2: ASSESSMENTS (HISTORY) */}
          {activePage === "Assessments" && (
            <div className="history-page">
              <div className="page-heading">
                <div>
                  <div className="eyebrow">AUDIT TRAIL &amp; ARCHIVE</div>
                  <h1>Inspection Records</h1>
                  <p>Comprehensive historical quality assessments logged in this environment.</p>
                </div>

                <div className="history-actions">
                  <button className="secondary-btn" onClick={fetchHistory}>
                    <RefreshCw size={14} /> Refresh
                  </button>
                  <button
                    className="danger-btn"
                    onClick={handleClearHistory}
                    disabled={history.length === 0}
                  >
                    <Trash2 size={14} /> Clear Log
                  </button>
                </div>
              </div>

              {/* Filter and Search Bar */}
              <div className="filter-bar">
                <div className="filter-group">
                  <span className="filter-label">Filter Status:</span>
                  {["ALL", "GOOD", "WARNING", "POOR"].map((f) => (
                    <button
                      key={f}
                      className={`filter-btn ${historyFilter === f ? "active" : ""}`}
                      onClick={() => setHistoryFilter(f)}
                    >
                      {f}
                    </button>
                  ))}
                </div>

                <div className="search-input-wrapper">
                  <Search size={14} />
                  <input
                    type="text"
                    placeholder="Search by filename or source..."
                    value={historySearch}
                    onChange={(e) => setHistorySearch(e.target.value)}
                  />
                  {historySearch && (
                    <button className="clear-search-btn" onClick={() => setHistorySearch("")}>
                      <X size={12} />
                    </button>
                  )}
                </div>
              </div>

              {/* History Table */}
              {history.length > 0 ? (
                <div className="history-table-container">
                  <table className="audit-table">
                    <thead>
                      <tr>
                        <th>Processed At</th>
                        <th>File / Source</th>
                        <th>Dimensions</th>
                        <th>Quality Score</th>
                        <th>Status</th>
                        <th>Checks Breakdown</th>
                        <th>PPE Suitability</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history
                        .filter((item) => (historyFilter === "ALL" ? true : item.overall_status === historyFilter))
                        .filter((item) =>
                          historySearch
                            ? item.filename.toLowerCase().includes(historySearch.toLowerCase())
                            : true
                        )
                        .map((item) => (
                          <tr key={item.id}>
                            <td className="timestamp-cell">
                              {new Date(item.processed_at).toLocaleTimeString([], {
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit",
                              })}
                            </td>
                            <td>
                              <strong>{item.filename}</strong>
                              <span className="sub-text">{item.source_type}</span>
                            </td>
                            <td>{item.width} × {item.height}</td>
                            <td>
                              <span className="table-score">{Math.round(item.overall_score)}</span>
                              <small>/100</small>
                            </td>
                            <td>
                              <span className={`status-pill pill-${item.overall_status.toLowerCase()}`}>
                                {item.overall_status}
                              </span>
                            </td>
                            <td>
                              <div className="table-checks-summary">
                                <span className="text-lime">{item.passed_count ?? "?"} Pass</span>
                                {item.failed_count > 0 && (
                                  <span className="text-red">• {item.failed_count} Fail</span>
                                )}
                                {item.uncertain_count > 0 && (
                                  <span className="text-amber">• {item.uncertain_count} Unc</span>
                                )}
                              </div>
                            </td>
                            <td>
                              <span
                                className={`ppe-badge ppe-${item.ppe_suitable ? "suitable" : "degraded"}`}
                              >
                                {item.ppe_suitable ? "SUITABLE" : "UNSUITABLE"}
                              </span>
                            </td>
                            <td>
                              <div className="action-buttons-cell">
                                <button
                                  className="inspect-btn-compact"
                                  title="Inspect in Dashboard"
                                  onClick={() => inspectAssessmentById(item.id)}
                                >
                                  <Eye size={13} /> Inspect
                                </button>
                                <button
                                  className="icon-btn-compact"
                                  title="Download JSON"
                                  onClick={() => handleDownloadReport(item)}
                                >
                                  <Download size={13} />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-history-panel">
                  <History size={36} className="text-muted" />
                  <strong>No Quality Records Logged Yet</strong>
                  <p>Execute an assessment from the Dashboard or run a webcam audit to populate records.</p>
                  <button className="primary-action-btn" onClick={() => setActivePage("Dashboard")}>
                    Go to Dashboard
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: CAMERA MONITOR (LIVE WEBCAM & SURVEILLANCE GRID) */}
          {activePage === "Camera Monitor" && (
            <div className="monitor-page">
              <div className="page-heading">
                <div>
                  <div className="eyebrow">LIVE SURVEILLANCE &amp; WEBCAM CAPTURE</div>
                  <h1>Camera Stream Monitor</h1>
                  <p>
                    Live hardware camera capture with OpenCV quality pre-flight evaluation. Audit video frames before enabling downstream PPE models.
                  </p>
                </div>

                <div className="live-indicator live">
                  <span className={`status-dot ${webcamActive ? "online" : "offline"}`} />
                  {webcamActive ? "Hardware Camera Streaming" : "Awaiting Camera Activation"}
                </div>
              </div>

              {/* Live Hardware Webcam Ingestion Panel */}
              <section className="webcam-monitor-card">
                <div className="webcam-header">
                  <div className="webcam-title">
                    <Video size={18} />
                    <strong>Live Hardware Camera Stream</strong>
                    {webcamActive && (
                      <span className="live-tag">LIVE FEED</span>
                    )}
                  </div>

                  <div className="webcam-controls-row">
                    {webcamDevices.length > 1 && (
                      <select
                        className="device-select"
                        value={selectedDeviceId}
                        onChange={(e) => setSelectedDeviceId(e.target.value)}
                        disabled={webcamActive}
                      >
                        {webcamDevices.map((d, idx) => (
                          <option key={d.deviceId || idx} value={d.deviceId}>
                            {d.label || `Camera ${idx + 1}`}
                          </option>
                        ))}
                      </select>
                    )}

                    {!webcamActive ? (
                      <button className="primary-action-btn" onClick={startWebcam} disabled={webcamLoading}>
                        {webcamLoading ? <RefreshCw size={15} className="spin" /> : <Play size={15} />}
                        Start Live Camera
                      </button>
                    ) : (
                      <>
                        <button className="capture-action-btn" onClick={() => captureAndAssessWebcam(true)}>
                          <Camera size={15} />
                          Capture Frame &amp; Assess Quality
                        </button>
                        <button
                          className={`guardian-toggle-btn ${continuousGuard ? "guard-active" : ""}`}
                          onClick={() => setContinuousGuard(!continuousGuard)}
                          title="Audit frame quality automatically every 4.5s"
                        >
                          <Activity size={15} />
                          {continuousGuard ? "Guardian Running (4.5s)" : "Enable Continuous Guardian"}
                        </button>
                        <button className="stop-btn" onClick={stopWebcam}>
                          <VideoOff size={15} /> Stop Stream
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Webcam Viewport */}
                <div className="webcam-viewport-frame">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`webcam-video ${webcamActive ? "active" : "hidden"}`}
                  />
                  <canvas ref={canvasRef} style={{ display: "none" }} />

                  {!webcamActive && (
                    <div className="webcam-idle-placeholder">
                      <div className="idle-icon-wrap">
                        <Video size={40} className="text-muted" />
                      </div>
                      <strong>Local Webcam Inactive</strong>
                      <p>
                        Activate your camera to capture live snapshots, run automated OpenCV quality audits, and test real-time lens conditions.
                      </p>
                      <button className="primary-action-btn" onClick={startWebcam}>
                        <Play size={15} /> Initialize Camera Stream
                      </button>
                    </div>
                  )}

                  {webcamActive && (
                    <div className="webcam-hud-overlay">
                      <div className="hud-corner top-left" />
                      <div className="hud-corner top-right" />
                      <div className="hud-corner bottom-left" />
                      <div className="hud-corner bottom-right" />
                      <div className="hud-center-cross" />

                      <div className="hud-stats-badge">
                        <span>STATUS: LIVE</span>
                        <span>FPS: 30.0</span>
                        <span>RES: {videoRef.current?.videoWidth || 1280}×{videoRef.current?.videoHeight || 720}</span>
                      </div>

                      {guardLastScore !== null && (
                        <div className={`hud-score-badge ${guardLastScore >= 80 ? "hud-good" : guardLastScore >= 50 ? "hud-warn" : "hud-poor"}`}>
                          LAST AUDIT: {Math.round(guardLastScore)}/100
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="webcam-footer-info">
                  <Info size={14} />
                  <span>
                    Camera frames captured here are encoded directly to JPEG and transmitted to <code>POST /api/assess/image</code> using multipart FormData.
                  </span>
                </div>
              </section>

              {/* Plant Surveillance Simulated Feeds Section */}
              <div className="section-heading" style={{ marginTop: "36px" }}>
                <div>
                  <h2>Industrial Site Surveillance Feeds</h2>
                  <p>Fixed CCTV camera streams deployed across plant sectors</p>
                </div>
                <span className="empty-label">4 ZONES MONITORED</span>
              </div>

              <div className="camera-grid">
                {cameras.map((cam) => (
                  <div key={cam.id} className="camera-card">
                    <div className="cam-header">
                      <div>
                        <strong>{cam.name}</strong>
                        <span className="cam-loc">{cam.location}</span>
                      </div>
                      <span className={`status-pill pill-${cam.quality_status.toLowerCase()}`}>
                        {cam.status}
                      </span>
                    </div>

                    <div className="cam-preview-placeholder">
                      <Video size={28} className="cam-icon" />
                      <span className="stream-badge">{cam.stream_protocol}</span>
                      <span className="resolution-badge">{cam.resolution}</span>
                    </div>

                    <div className="cam-specs-grid">
                      <div>
                        <span className="spec-label">IP Address</span>
                        <strong>{cam.ip}</strong>
                      </div>
                      <div>
                        <span className="spec-label">Frame Rate</span>
                        <strong>{cam.fps} FPS</strong>
                      </div>
                      <div>
                        <span className="spec-label">Last Score</span>
                        <strong>{cam.last_score}/100</strong>
                      </div>
                      <div>
                        <span className="spec-label">PPE Pipeline</span>
                        <strong className={cam.ppe_active ? "text-lime" : "text-amber"}>
                          {cam.ppe_active ? "ENABLED" : "PAUSED"}
                        </strong>
                      </div>
                    </div>

                    {cam.active_alert && (
                      <div className="cam-alert-box">
                        <AlertTriangle size={14} />
                        <span>{cam.active_alert}</span>
                      </div>
                    )}

                    <button
                      className="cam-audit-btn"
                      onClick={() => {
                        handleAssessSample(cam.sample_ref);
                        setActivePage("Dashboard");
                      }}
                    >
                      <Zap size={14} /> Run Instant Quality Audit
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 4: SETTINGS */}
          {activePage === "Settings" && (
            <div className="settings-page">
              <div className="page-heading">
                <div>
                  <div className="eyebrow">ENGINE CONFIGURATION</div>
                  <h1>Assessment Parameters &amp; Thresholds</h1>
                  <p>
                    Adjust rule-based computer vision thresholds and penalty weights to match your plant's lighting conditions and camera hardware.
                  </p>
                </div>

                <button className="secondary-btn" onClick={handleResetConfig}>
                  <RefreshCw size={14} /> Restore System Defaults
                </button>
              </div>

              {configSavedToast && (
                <div className="success-toast">
                  <Check size={16} /> Configuration updated successfully in the Python backend!
                </div>
              )}

              {config ? (
                <form className="settings-form" onSubmit={handleSaveConfig}>
                  {/* Darkness Settings */}
                  <div className="settings-section">
                    <h3>Darkness Thresholds &amp; Penalty</h3>
                    <div className="fields-row">
                      <div className="field-group">
                        <label>Min Mean Intensity (0–255)</label>
                        <input
                          type="number"
                          step="0.5"
                          value={config.darkness_mean_threshold}
                          onChange={(e) =>
                            setConfig({ ...config, darkness_mean_threshold: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 60.0. Average grayscale intensity threshold.</small>
                      </div>

                      <div className="field-group">
                        <label>Max Dark Pixel Fraction (0.0–1.0)</label>
                        <input
                          type="number"
                          step="0.05"
                          value={config.darkness_dark_pixel_fraction}
                          onChange={(e) =>
                            setConfig({ ...config, darkness_dark_pixel_fraction: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 0.40 (40% max dark pixels allowed).</small>
                      </div>

                      <div className="field-group">
                        <label>Darkness Penalty (pts)</label>
                        <input
                          type="number"
                          value={config.darkness_penalty}
                          onChange={(e) =>
                            setConfig({ ...config, darkness_penalty: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 30.0 points deducted for confirmed darkness failure.</small>
                      </div>
                    </div>
                  </div>

                  {/* Blur Settings */}
                  <div className="settings-section">
                    <h3>Blur &amp; Optical Focus Thresholds</h3>
                    <div className="fields-row">
                      <div className="field-group">
                        <label>Min Laplacian Variance Threshold</label>
                        <input
                          type="number"
                          step="1.0"
                          value={config.blur_laplacian_variance_threshold}
                          onChange={(e) =>
                            setConfig({ ...config, blur_laplacian_variance_threshold: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 100.0. Second-derivative operator measuring sharp edge energy.</small>
                      </div>

                      <div className="field-group">
                        <label>Blur Score Penalty (pts)</label>
                        <input
                          type="number"
                          value={config.blur_penalty}
                          onChange={(e) => setConfig({ ...config, blur_penalty: parseFloat(e.target.value) })}
                        />
                        <small>Default: 25.0 points deducted for confirmed optical blur.</small>
                      </div>
                    </div>
                  </div>

                  {/* Resolution Settings */}
                  <div className="settings-section">
                    <h3>Resolution Requirements</h3>
                    <div className="fields-row">
                      <div className="field-group">
                        <label>Minimum Width (px)</label>
                        <input
                          type="number"
                          value={config.resolution_min_width}
                          onChange={(e) =>
                            setConfig({ ...config, resolution_min_width: parseInt(e.target.value) })
                          }
                        />
                        <small>Default: 640 px.</small>
                      </div>

                      <div className="field-group">
                        <label>Minimum Height (px)</label>
                        <input
                          type="number"
                          value={config.resolution_min_height}
                          onChange={(e) =>
                            setConfig({ ...config, resolution_min_height: parseInt(e.target.value) })
                          }
                        />
                        <small>Default: 480 px.</small>
                      </div>

                      <div className="field-group">
                        <label>Resolution Penalty (pts)</label>
                        <input
                          type="number"
                          value={config.resolution_penalty}
                          onChange={(e) =>
                            setConfig({ ...config, resolution_penalty: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 20.0 points deducted if resolution below minimum.</small>
                      </div>
                    </div>
                  </div>

                  {/* Glare Settings */}
                  <div className="settings-section">
                    <h3>Glare &amp; Overexposure Thresholds</h3>
                    <div className="fields-row">
                      <div className="field-group">
                        <label>Bright Pixel Fraction Threshold</label>
                        <input
                          type="number"
                          step="0.01"
                          value={config.glare_bright_pixel_fraction_threshold}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              glare_bright_pixel_fraction_threshold: parseFloat(e.target.value),
                            })
                          }
                        />
                        <small>Default: 0.05 (5% overexposed pixels).</small>
                      </div>

                      <div className="field-group">
                        <label>Glare Score Penalty (pts)</label>
                        <input
                          type="number"
                          value={config.glare_penalty}
                          onChange={(e) => setConfig({ ...config, glare_penalty: parseFloat(e.target.value) })}
                        />
                        <small>Default: 15.0 points deducted for excessive glare.</small>
                      </div>
                    </div>
                  </div>

                  {/* Obstruction Settings */}
                  <div className="settings-section">
                    <h3>Obstruction &amp; Lens Cover Settings</h3>
                    <div className="fields-row">
                      <div className="field-group">
                        <label>Obstruction Confirmed Penalty (pts)</label>
                        <input
                          type="number"
                          value={config.obstruction_penalty}
                          onChange={(e) =>
                            setConfig({ ...config, obstruction_penalty: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 10.0 points. Applied ONLY for OBSTRUCTED, never for UNCERTAIN.</small>
                      </div>

                      <div className="field-group">
                        <label>Good Status Threshold (pts)</label>
                        <input
                          type="number"
                          value={config.score_good_threshold}
                          onChange={(e) =>
                            setConfig({ ...config, score_good_threshold: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 80.0 points for GOOD rating.</small>
                      </div>

                      <div className="field-group">
                        <label>Warning Status Threshold (pts)</label>
                        <input
                          type="number"
                          value={config.score_warning_threshold}
                          onChange={(e) =>
                            setConfig({ ...config, score_warning_threshold: parseFloat(e.target.value) })
                          }
                        />
                        <small>Default: 50.0 points for WARNING rating.</small>
                      </div>
                    </div>
                  </div>

                  <div className="settings-actions">
                    <button type="submit" className="primary-action-btn">
                      <Check size={16} /> Save Thresholds
                    </button>
                  </div>
                </form>
              ) : (
                <div className="loading-card">
                  <RefreshCw size={24} className="spin" />
                  <span>Loading configuration from Python backend...</span>
                </div>
              )}

              {/* Architecture & Methodology Disclosure */}
              <section className="engine-disclosure-card">
                <div className="disclosure-header">
                  <Info size={18} className="text-lime" />
                  <strong>Architecture &amp; Algorithm Disclosure</strong>
                </div>
                <p>
                  <strong>No Machine Learning or AI Models:</strong> Camera Quality Guardian relies strictly on traditional, deterministic image processing algorithms via OpenCV and NumPy.
                </p>
                <ul>
                  <li><strong>Darkness:</strong> Grayscale channel intensity histogram and thresholding.</li>
                  <li><strong>Blur:</strong> Discrete Laplacian second-order differential variance operator σ²(∇²I).</li>
                  <li><strong>Resolution:</strong> Explicit frame dimension validation (W ≥ 640, H ≥ 480).</li>
                  <li><strong>Glare:</strong> HSV space thresholding (Value ≥ 230 and Saturation ≤ 30).</li>
                  <li><strong>Obstruction:</strong> Multi-heuristic decision tree combining Canny edge density with intensity standard deviation. Returns UNCERTAIN without score penalty whenever extreme lighting or blur explain missing detail.</li>
                </ul>
              </section>
            </div>
          )}

          {/* Footer */}
          <footer>
            <span>Camera Quality Guardian • Industrial Safety Vision Pre-Flight</span>
            <span>Deterministic OpenCV &amp; NumPy Pipeline • Python 3.12 &amp; FastAPI</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
