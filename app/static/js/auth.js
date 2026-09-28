/* Shared JS helpers: token storage, logout, API wrapper */

const API_BASE = "";

function getToken() { return localStorage.getItem("token"); }
function setToken(t) { localStorage.setItem("token", t); }
function clearToken() {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
}
function setUser(u) { localStorage.setItem("user", JSON.stringify(u)); }
function getUser() {
    try { return JSON.parse(localStorage.getItem("user")); } catch { return null; }
}

async function apiFetch(path, options = {}) {
    const token = getToken();
    const headers = Object.assign(
        { "Content-Type": "application/json" },
        options.headers || {}
    );
    if (token) headers["Authorization"] = "Bearer " + token;

    const resp = await fetch(API_BASE + path, { ...options, headers });
    let data = null;
    try { data = await resp.json(); } catch {}
    if (!resp.ok) {
        throw new Error((data && data.error) || `HTTP ${resp.status}`);
    }
    return data;
}

document.addEventListener("DOMContentLoaded", () => {
    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", (e) => {
            e.preventDefault();
            clearToken();
            window.location.href = "/login";
        });
    }
});
