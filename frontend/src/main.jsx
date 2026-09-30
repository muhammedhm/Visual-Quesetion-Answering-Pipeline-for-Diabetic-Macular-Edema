import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart3,
  Camera,
  Eye,
  EyeOff,
  History,
  Loader2,
  LogOut,
  Plus,
  Send,
  Trash2,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function apiPath(path) {
  if (!path) return "";
  const normalized = path.replaceAll("\\", "/");
  const fileName = normalized.split("/").pop();
  return `${API_URL}/uploads/${fileName}`;
}

async function request(path, { token, method = "GET", body, isForm = false } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    data = {};
  }

  if (!res.ok) {
    throw new Error(data.detail || data.message || "Request failed");
  }
  return data;
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("deepeye_token"));
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem("deepeye_user");
    return saved ? JSON.parse(saved) : null;
  });
  const [booting, setBooting] = useState(Boolean(token));

  useEffect(() => {
    if (!token) {
      setBooting(false);
      return;
    }
    request("/auth/me", { token })
      .then((me) => {
        setUser(me);
        localStorage.setItem("deepeye_user", JSON.stringify(me));
      })
      .catch(() => {
        localStorage.removeItem("deepeye_token");
        localStorage.removeItem("deepeye_user");
        setToken(null);
        setUser(null);
      })
      .finally(() => setBooting(false));
  }, [token]);

  function handleLogin(payload) {
    localStorage.setItem("deepeye_token", payload.access_token);
    localStorage.setItem("deepeye_user", JSON.stringify(payload.user));
    setToken(payload.access_token);
    setUser(payload.user);
  }

  async function logout() {
    try {
      if (token) await request("/auth/logout", { token, method: "POST" });
    } catch {
      /* Token may already be expired; local logout still applies. */
    }
    localStorage.removeItem("deepeye_token");
    localStorage.removeItem("deepeye_user");
    setToken(null);
    setUser(null);
  }

  if (booting) return <LoadingScreen />;
  if (!token || !user) return <LoginScreen onLogin={handleLogin} />;
  if (user.is_admin) return <AdminDashboard token={token} user={user} onLogout={logout} />;
  return <ChatWorkspace token={token} user={user} onLogout={logout} />;
}

function LoadingScreen() {
  return (
    <div className="page login-page">
      <Loader2 className="spin" size={34} />
    </div>
  );
}

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await request("/auth/login", {
        method: "POST",
        body: { username, password },
      });
      onLogin(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page login-page">
      <section className="login-card">
        <h1>DeepEyeNet</h1>
        <h2>Login</h2>
        <form onSubmit={submit} className="login-form">
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Email"
            autoComplete="username"
          />
          <div className="password-field">
            <input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
            />
            <button type="button" aria-label="Toggle password" onClick={() => setShowPassword((v) => !v)}>
              {showPassword ? <EyeOff size={21} /> : <Eye size={21} />}
            </button>
          </div>
          {error && <p className="error-text">{error}</p>}
          <button className="primary-action" disabled={loading}>
            {loading ? <Loader2 className="spin" size={20} /> : "Login"}
          </button>
        </form>
      </section>
      <PartnerMarks />
    </main>
  );
}

function PartnerMarks() {
  return (
    <section className="partners" aria-label="Partnered with">
      <p>PARTNERED WITH</p>
      <div className="partner-row">
        <img className="partner-logo escv-logo" src="/assets/escv_logo.png" alt="ESCV" />
        <img className="partner-logo aeye-logo" src="/assets/a_eye_logo.png" alt="A-Eye Diagnostics" />
      </div>
    </section>
  );
}

function TopBar({ title, subtitle, user, onLogout }) {
  return (
    <header className="topbar">
      <div className="brand-line">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="topbar-actions">
        {user?.username && <span>{user.username}</span>}
        <button onClick={onLogout}>
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </header>
  );
}

function ChatWorkspace({ token, user, onLogout }) {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [architecture, setArchitecture] = useState("vqa");
  const [message, setMessage] = useState("");
  const [threads, setThreads] = useState([]);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    try {
      const rows = await request("/history/vqa", { token });
      const grouped = rows.map((row) => ({
        id: row.id,
        title: `Chat ${row.created_at?.slice(0, 10) || ""}`,
        count: "2 messages",
        time: row.created_at ? new Date(row.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "",
        image: apiPath(row.image_path),
        question: row.question,
        answer: row.answer,
        createdAt: row.created_at,
      }));
      setThreads(grouped);
    } catch {
      setThreads([]);
    }
  }

  function chooseFile(file) {
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setMessages([]);
    setError("");
  }

  function resetChat() {
    setImageFile(null);
    setImagePreview("");
    setMessage("");
    setMessages([]);
    setError("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function deleteConversation(threadId) {
    setError("");
    try {
      await request(`/history/vqa/${threadId}`, { token, method: "DELETE" });
      setThreads((prev) => prev.filter((thread) => thread.id !== threadId));
    } catch (err) {
      setError(err.message);
    }
  }

  function viewConversation(thread) {
    setError("");
    setImagePreview(thread.image || "");
    setImageFile(null);
    setMessage("");
    setMessages([
      { role: "user", text: thread.question || "Saved clinical question" },
      {
        role: "assistant",
        result: {
          status: "success",
          architecture_label: "Saved VQA",
          mode: "history",
          answers: [
            {
              question_type: "saved_conversation",
              matched_standard_question: thread.question,
              answer: { label: thread.answer, raw: thread.answer },
              pathological_feature_explanation:
                "This is the stored model answer from your previous analysis.",
            },
          ],
          disclaimer:
            "AI-generated research output only. Consult an ophthalmologist for clinical decisions.",
        },
      },
    ]);
  }

  async function askQuestion(e) {
    e.preventDefault();
    if (!imageFile || !message.trim()) return;
    setError("");
    setLoading(true);
    const userMessage = { role: "user", text: message.trim() };
    setMessages((prev) => [...prev, userMessage]);

    const form = new FormData();
    form.append("image", imageFile);
    form.append("message", message.trim());
    if (architecture === "multimodal_rag") {
      form.append("top_k", "5");
      form.append("alpha", "0.7");
    }

    try {
      const endpoint = architecture === "multimodal_rag" ? "/rag/chat" : "/vqa/chat";
      const data = await request(endpoint, { token, method: "POST", body: form, isForm: true });
      setMessages((prev) => [...prev, { role: "assistant", result: data }]);
      setMessage("");
      loadHistory();
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [...prev, { role: "assistant", error: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <TopBar title="DeepEyeNet" subtitle="Upload a fundus image and ask clinical questions" user={user} onLogout={onLogout} />
      <section className="chat-layout">
        <aside className="sidebar">
          <div className="sidebar-card">
            <div className="sidebar-head">
              <h2>
                <History size={24} />
                Conversations
              </h2>
              <button onClick={resetChat}>
                <Plus size={18} />
                New Chat
              </button>
            </div>
            <div className="thread-list">
              {threads.length === 0 && <p className="muted center">No conversations yet</p>}
              {threads.map((thread) => (
                <article className="thread-item" key={thread.id}>
                  <div className="thread-copy">
                    <strong>{thread.title}</strong>
                    <span>{thread.count}</span>
                    <small>{thread.time}</small>
                  </div>
                  {thread.image && <img src={thread.image} alt="" />}
                  <div className="thread-actions">
                    <button
                      className="thread-view"
                      onClick={() => viewConversation(thread)}
                      type="button"
                      aria-label="View conversation"
                      title="View conversation"
                    >
                      View
                    </button>
                    <button
                      className="thread-delete"
                      onClick={() => deleteConversation(thread.id)}
                      type="button"
                      aria-label="Delete conversation"
                      title="Delete conversation"
                    >
                      <Trash2 size={17} />
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </aside>

        <section className="upload-column">
          <div className="upload-card">
            <h2>
              <Camera size={24} />
              Upload Retinal Image
            </h2>
            <div className="architecture-switch" aria-label="Analysis architecture">
              <button
                className={architecture === "vqa" ? "active" : ""}
                onClick={() => setArchitecture("vqa")}
                type="button"
              >
                VQA
              </button>
              <button
                className={architecture === "multimodal_rag" ? "active" : ""}
                onClick={() => setArchitecture("multimodal_rag")}
                type="button"
              >
                Multimodal RAG
              </button>
            </div>
            <p className="architecture-note">
              {architecture === "vqa"
                ? "Direct trained VQA model answers the matched standard question."
                : "Image-text retrieval finds similar DME cases before producing the answer."}
            </p>
            <div
              className={`drop-zone ${imagePreview ? "has-preview" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                chooseFile(e.dataTransfer.files?.[0]);
              }}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
                onChange={(e) => chooseFile(e.target.files?.[0])}
              />
              {imagePreview ? (
                <img src={imagePreview} alt="Uploaded retina preview" />
              ) : (
                <>
                  <Camera size={48} />
                  <strong>Click or drag to upload</strong>
                  <span>JPG, PNG, BMP, TIFF</span>
                </>
              )}
            </div>
          </div>
        </section>

        <section className="chat-panel">
          <div className="messages">
            {messages.length === 0 ? (
              <p className="empty-state">No analysis yet. Upload an image and ask a question!</p>
            ) : (
              messages.map((item, index) =>
                item.role === "user" ? (
                  <div className="bubble user-bubble" key={index}>{item.text}</div>
                ) : (
                  <AnalysisCard key={index} item={item} />
                ),
              )
            )}
          </div>
          <form className="composer" onSubmit={askQuestion}>
            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              disabled={!imageFile || loading}
              placeholder={
                architecture === "vqa"
                  ? "Ask a VQA question about the retinal image..."
                  : "Ask using multimodal RAG over similar retinal cases..."
              }
            />
            <button disabled={!imageFile || !message.trim() || loading} aria-label="Send question">
              {loading ? <Loader2 className="spin" size={24} /> : <Send size={25} />}
            </button>
          </form>
          {!imageFile && (
            <p className="upload-hint">
              <Upload size={16} />
              Please upload an image first
            </p>
          )}
          {error && <p className="error-text chat-error">{error}</p>}
        </section>
      </section>
    </main>
  );
}

function AnalysisCard({ item }) {
  if (item.error) return <div className="bubble assistant-bubble error-card">{item.error}</div>;
  const result = item.result;
  return (
    <div className="bubble assistant-bubble">
      <div className="result-header">
        <strong>{result.mode === "explain_all" ? "Full Image Explanation" : "Clinical Answer"}</strong>
        <span>{result.architecture_label || result.status}</span>
      </div>
      <div className="answer-grid">
        {result.answers?.map((answer) => (
          <article className="answer-card" key={answer.question_type}>
            <small>{answer.question_type.replaceAll("_", " ")}</small>
            <h3>{answer.answer.label}</h3>
            <p>{answer.pathological_feature_explanation}</p>
            <span>{answer.matched_standard_question}</span>
          </article>
        ))}
      </div>
      {result.rag && (
        <div className="rag-section">
          <h4>Retrieved Clinical Context</h4>
          {result.rag.map((ragItem) => (
            <article className="rag-card" key={ragItem.question_type}>
              <strong>{ragItem.question_type.replaceAll("_", " ")}</strong>
              {ragItem.error ? (
                <p>{ragItem.error}</p>
              ) : (
                <>
                  <p>{ragItem.structured_answer?.primary_finding}</p>
                  <span>
                    {ragItem.structured_answer?.retrieval_quality} retrieval ·{" "}
                    {ragItem.retrieved_cases?.length || 0} similar cases
                  </span>
                </>
              )}
            </article>
          ))}
        </div>
      )}
      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}

function AdminDashboard({ token, user, onLogout }) {
  const [users, setUsers] = useState([]);
  const [newUser, setNewUser] = useState({ username: "", password: "", is_admin: false });
  const [deleteUsername, setDeleteUsername] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    refreshUsers();
  }, []);

  async function refreshUsers() {
    try {
      setUsers(await request("/admin/users", { token }));
    } catch (err) {
      setError(err.message);
    }
  }

  async function addUser(e) {
    e.preventDefault();
    setError("");
    setMessage("");
    try {
      await request("/admin/users", { token, method: "POST", body: newUser });
      setMessage(`Added ${newUser.username}`);
      setNewUser({ username: "", password: "", is_admin: false });
      refreshUsers();
    } catch (err) {
      setError(err.message);
    }
  }

  async function deleteUser(e) {
    e.preventDefault();
    if (!deleteUsername.trim()) return;
    setError("");
    setMessage("");
    try {
      await request(`/admin/users/${encodeURIComponent(deleteUsername.trim())}`, { token, method: "DELETE" });
      setMessage(`Deleted ${deleteUsername.trim()}`);
      setDeleteUsername("");
      refreshUsers();
    } catch (err) {
      setError(err.message);
    }
  }

  const activities = useMemo(() => {
    const userEvents = users.slice(0, 5).map((row) => ({
      who: row.username,
      what: row.is_admin ? "Admin account active" : "User account active",
      when: row.created_at ? new Date(row.created_at).toLocaleDateString() : "recently",
    }));
    return [
      { who: user.username, what: "Logged in", when: "now" },
      ...userEvents,
    ];
  }, [users, user.username]);

  return (
    <main className="admin-shell">
      <TopBar title="DeepEye - Admin Dashboard" subtitle="Manage users and monitor system activities" user={null} onLogout={onLogout} />
      <section className="admin-grid">
        <article className="admin-card">
          <h2>
            <Plus size={34} />
            Add New User
          </h2>
          <p>Register a new user account</p>
          <form onSubmit={addUser} className="admin-form">
            <label>Email Address</label>
            <input
              value={newUser.username}
              onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
              placeholder="Enter user email"
            />
            <label>Password</label>
            <input
              value={newUser.password}
              onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
              placeholder="Enter password"
              type="password"
            />
            <label className="check-row">
              <input
                checked={newUser.is_admin}
                onChange={(e) => setNewUser({ ...newUser, is_admin: e.target.checked })}
                type="checkbox"
              />
              Admin user
            </label>
            <button className="primary-action">Add User</button>
          </form>
        </article>

        <article className="admin-card">
          <h2>
            <Trash2 size={31} />
            Delete User
          </h2>
          <p>Remove a user account from the system</p>
          <form onSubmit={deleteUser} className="admin-form">
            <label>User Email</label>
            <input
              value={deleteUsername}
              onChange={(e) => setDeleteUsername(e.target.value)}
              placeholder="Enter email of user to delete"
            />
            <button className="danger-action">Delete User</button>
          </form>
          <div className="user-list">
            {users.map((row) => (
              <span key={row.id}>{row.username}{row.is_admin ? " · admin" : ""}</span>
            ))}
          </div>
        </article>

        <article className="admin-card activities-card">
          <h2>
            <BarChart3 size={32} />
            Recent Activities
          </h2>
          <p>System activity log</p>
          <div className="activities-list">
            {activities.map((activity, index) => (
              <div className="activity" key={`${activity.who}-${index}`}>
                <strong>{activity.who}</strong>
                <span>{activity.what}</span>
                <time>{activity.when}</time>
              </div>
            ))}
          </div>
        </article>
      </section>
      {(message || error) && (
        <div className={`toast ${error ? "toast-error" : ""}`}>{error || message}</div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
