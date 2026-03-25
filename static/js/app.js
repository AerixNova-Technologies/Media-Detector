/**
 * Mission Control - Core Application Logic
 * Handles Camera management, Live stream, SSE, and Stats
 */

let activeCamId = null;
let statsInterval = null;
let activeStatsId = null;
let allCameras = [];
let selectedCamForRoles = null;

// --- Camera Management ---

async function loadCameras() {
    const list = document.getElementById('camera-list');
    const grid = document.getElementById('camera-grid');
    
    if (list) list.innerHTML = '<div style="padding: 20px; text-align: center;"><div class="saas-spinner"></div></div>';
    if (grid) grid.innerHTML = '<div style="padding: 40px; text-align: center;"><div class="saas-spinner"></div></div>';

    try {
        const res = await fetch('/api/cameras?refresh=1&t=' + Date.now());
        const cams = await res.json();
        allCameras = cams;
        
        if (list) renderCameraList(cams);
        if (grid) renderCameraGrid(cams);

        if (activeCamId) {
            const active = cams.find(c => String(c.id) == String(activeCamId));
            if (active) selectCamera(active.id, active.name, true);
        }
    } catch (e) {
        console.error("Failed to load cameras", e);
    }
}

function renderCameraList(cams) {
    const list = document.getElementById('camera-list');
    if (!list) return;
    
    if (!cams || cams.length === 0) {
        list.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-tertiary);">No cameras detected</div>';
        return;
    }

    list.innerHTML = cams.map(c => {
        const isOnline = c.status && c.status.includes('Online');
        const isActive = (c.id == activeCamId);
        const ip = c.ip_address || (c.type === 'webcam' ? 'Built-in / USB' : 'Network Device');
        const roles = c.roles || [];
        const roleText = roles.length > 0 ? roles.map(r => r.charAt(0).toUpperCase() + r.slice(1)).join(', ') : 'No Roles';
        
        return `
            <div class="camera-item ${isActive ? 'active' : ''}" onclick="selectCamera('${c.id}','${c.name}')">
                <i data-lucide="${c.type === 'webcam' ? 'laptop' : 'video'}" style="width: 18px; color: ${isOnline ? 'var(--text-secondary)' : 'var(--text-tertiary)'}"></i>
                <div style="flex: 1; min-width: 0;">
                    <div style="font-size: 13px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary);">${c.name}</div>
                    <div style="font-size: 11px; color: ${isOnline ? 'var(--success)' : 'var(--text-tertiary)'}; display: flex; align-items: center; gap: 6px; margin-top: 2px;">
                        <span style="width: 5px; height: 5px; border-radius: 50%; background: ${isOnline ? 'var(--success)' : 'var(--text-tertiary)'}"></span>
                        ${ip}
                    </div>
                    <div class="camera-role-label" style="${roles.length === 0 ? 'opacity: 0.5; color: var(--text-tertiary);' : ''}">
                        ${roleText}
                    </div>
                </div>
                <button class="options-btn" onclick="event.stopPropagation(); showCameraOptions('${c.id}', '${c.name.replace(/'/g, "\\'")}', event)">
                    <i data-lucide="more-vertical" style="width: 14px;"></i>
                </button>
            </div>
        `;
    }).join('');
    lucide.createIcons();
}

function renderCameraGrid(cams) {
    const grid = document.getElementById('camera-grid');
    if (!grid) return;

    if (!cams || cams.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-tertiary);">No cameras available</div>';
        return;
    }

    grid.innerHTML = cams.map(c => {
        const isOnline = c.status && c.status.includes('Online');
        const streamUrl = `/api/camera_preview_stream/${c.id}?t=${Date.now()}`;
        
        return `
            <div class="saas-card" style="padding: 0; overflow: hidden; cursor: pointer; border-color: ${isOnline ? 'var(--border-main)' : 'var(--error)'}" onclick="selectCamera('${c.id}','${c.name}')">
                <div style="aspect-ratio: 16/9; background: #020617; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center;">
                    <!-- Premium Lens Grid Loader -->
                    <div style="position: absolute; opacity: 0.8; transform: scale(0.6);">
                        <svg class="bb-eye-svg colorful" viewBox="0 0 120 70">
                            <defs>
                                <linearGradient id="eye-grad-grid-${c.id}" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#FF0080" />
                                    <stop offset="40%" style="stop-color:#7928CA" />
                                    <stop offset="100%" style="stop-color:#0070F3" />
                                </linearGradient>
                                <radialGradient id="iris-grad-grid-${c.id}" cx="50%" cy="50%" r="50%">
                                    <stop offset="0%" style="stop-color:#ffffff; stop-opacity:0.2" />
                                    <stop offset="100%" style="stop-color:#7928CA; stop-opacity:0.1" />
                                </radialGradient>
                            </defs>
                            <path class="bb-eye-outline colorful" d="M10,35 C10,35 40,5 60,5 C80,5 110,35 110,35 C110,35 80,65 60,65 C40,65 10,35 10,35 Z" stroke="url(#eye-grad-grid-${c.id})" />
                            <circle class="bb-eye-iris colorful" cx="60" cy="35" r="22" fill="url(#iris-grad-grid-${c.id})" stroke="url(#eye-grad-grid-${c.id})" />
                            
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(0 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(45 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(90 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(135 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(180 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(225 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(270 60 35)" />
                            <line class="bb-aperture-line" x1="60" y1="13" x2="60" y2="18" transform="rotate(315 60 35)" />

                            <circle class="bb-eye-pupil colorful" cx="60" cy="35" r="8" fill="url(#eye-grad-grid-${c.id})" />
                            <circle class="bb-lens-reflection" cx="54" cy="29" r="3" />
                            <circle class="bb-lens-reflection" cx="65" cy="42" r="1.5" />
                        </svg>
                    </div>

                    <img src="${streamUrl}" style="width: 100%; height: 100%; object-fit: cover; position: relative; z-index: 2;" onerror="this.src='/static/img/hero.png'; this.style.opacity=0.3;">
                    
                    <div style="position: absolute; top: 12px; left: 12px; background: rgba(239, 68, 68, 0.8); z-index: 10; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 800; color: white; display: flex; align-items: center; gap: 4px;">
                        <span style="width: 4px; height: 4px; background: white; border-radius: 50%; animation: pulse 2s infinite;"></span>
                        LIVE
                    </div>
                    ${!isOnline ? '<div style="position: absolute; inset: 0; background: rgba(15, 23, 42, 0.6); z-index: 15; display: flex; align-items: center; justify-content: center; color: var(--error); font-weight: 700; letter-spacing: 0.1em; font-size: 11px;">OFFLINE</div>' : ''}
                </div>
                <div style="padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;">
                    <div class="heading" style="font-size: 14px;">${c.name}</div>
                    <i data-lucide="maximize-2" style="width: 14px; color: var(--text-tertiary);"></i>
                </div>
            </div>
        `;
    }).join('');
    lucide.createIcons();
}

async function selectCamera(id, name, force = false) {
    if (activeCamId === id && !force) {
        showFocused();
        return;
    }

    showFocused();
    activeCamId = id;
    
    // Update active state in list
    document.querySelectorAll('.camera-item').forEach(el => el.classList.remove('active'));
    
    const img = document.getElementById('stream-img');
    const ph = document.getElementById('placeholder');
    const label = document.getElementById('cam-label');
    const badge = document.getElementById('live-badge');
    const stopBtn = document.getElementById('stop-btn');

    if (ph) ph.style.display = 'flex';
    if (img) img.style.display = 'none';
    if (label) { label.innerText = name; label.style.display = 'block'; }

    try {
        await fetch('/api/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id }),
        });

        setTimeout(() => {
            if (img) {
                const newSrc = '/video_feed?cam=' + encodeURIComponent(id) + '&t=' + Date.now();
                if (img.src !== newSrc) {
                    img.src = newSrc;
                    img.style.display = 'block';
                }
            }
            if (ph) ph.style.display = 'none';
            if (badge) badge.style.display = 'flex';
            if (stopBtn) stopBtn.disabled = false;
        }, 100);

        startStats();
    } catch (e) {
        console.error("Select error", e);
    }
}

async function stopCamera() {
    await fetch('/api/stop', { method: 'POST' });
    activeCamId = null;
    stopStats();

    const img = document.getElementById('stream-img');
    const ph = document.getElementById('placeholder');
    const badge = document.getElementById('live-badge');
    const label = document.getElementById('cam-label');
    const stopBtn = document.getElementById('stop-btn');

    if (img) img.style.display = 'none';
    if (ph) ph.style.display = 'flex';
    if (badge) badge.style.display = 'none';
    if (label) label.style.display = 'none';
    if (stopBtn) stopBtn.disabled = true;

    document.querySelectorAll('.camera-item').forEach(el => el.classList.remove('active'));
    showGrid();
}

// --- Camera Roles / Options ---

function showCameraOptions(id, name, event) {
    const modal = document.getElementById('camera-roles-modal');
    if (!modal) return;
    
    selectedCamForRoles = id;
    document.getElementById('modal-cam-name').innerText = 'CONFIGURING: ' + name;
    
    // Fetch current roles from allCameras
    const cam = allCameras.find(c => String(c.id) == String(id));
    if (!cam) return;
    
    const roles = cam.roles || [];
    document.querySelectorAll('#roles-form input').forEach(input => {
        input.checked = roles.includes(input.value);
    });
    
    modal.style.display = 'flex';
}

function closeRolesModal() {
    const modal = document.getElementById('camera-roles-modal');
    if (modal) modal.style.display = 'none';
    selectedCamForRoles = null;
}

async function saveRoles() {
    if (!selectedCamForRoles) return;
    
    const checkboxes = document.querySelectorAll('#roles-form input');
    const roles = Array.from(checkboxes)
        .filter(i => i.checked)
        .map(i => i.value);
    
    const btn = document.querySelector('.saas-modal-footer .btn-primary');
    const original = btn.innerText;
    btn.innerText = 'Syncing...';
    btn.disabled = true;

    try {
        const res = await fetch(`/api/cameras/${selectedCamForRoles}/roles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roles })
        });
        const data = await res.json();
        if (data.success) {
            btn.innerText = 'Synchronized!';
            btn.style.background = 'var(--success)';
            
            // Update local state
            const cam = allCameras.find(c => String(c.id) == String(selectedCamForRoles));
            if (cam) cam.roles = roles;
            
            // Refresh list to show new roles
            renderCameraList(allCameras);
            
            setTimeout(() => {
                closeRolesModal();
                btn.innerText = original;
                btn.style.background = '';
                btn.disabled = false;
            }, 1000);
        }
    } catch (e) {
        console.error("Save roles failed", e);
        btn.innerText = 'Sync Failed';
        btn.style.background = 'var(--error)';
        setTimeout(() => {
            btn.innerText = original;
            btn.disabled = false;
        }, 2000);
    }
}

// --- Stats & Metrics ---

function startStats() {
    if (statsInterval) clearInterval(statsInterval);
    statsInterval = setInterval(fetchStats, 1000);
}

function stopStats() {
    if (statsInterval) clearInterval(statsInterval);
    statsInterval = null;
    resetStats();
}

async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        updateStatsUI(data);
    } catch (e) {}
}

function updateStatsUI(d) {
    const elPersons = document.getElementById('stat-persons');
    const elEntries = document.getElementById('stat-entries');
    const elExits = document.getElementById('stat-exits');
    const elFps = document.getElementById('stat-fps');
    const elMotion = document.getElementById('stat-motion');
    const container = document.getElementById('person-cards');

    if (elPersons) elPersons.innerText = d.persons;
    if (elEntries) elEntries.innerText = d.entries || 0;
    if (elExits) elExits.innerText = d.exits || 0;
    if (elFps) elFps.innerText = d.fps + ' FPS';
    if (elMotion) {
        elMotion.innerText = d.motion ? 'YES' : 'NO';
        elMotion.style.color = d.motion ? 'var(--success)' : 'var(--error)';
    }

    if (container) {
        if (!d.tracks || d.tracks.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-tertiary);">No active detections</div>';
            return;
        }

        container.innerHTML = d.tracks.map(t => {
            const isAlert = t.action && (t.action.includes('▲') || t.action.includes('RUN'));
            const identity = (t.identity && t.identity !== "Unknown") ? t.identity : "Subject #" + t.id;
            
            return `
                <div class="saas-card" style="padding: 12px; margin-bottom: 8px; border-left: 3px solid ${isAlert ? 'var(--error)' : 'var(--accent)'}; transition: none !important;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 700; font-size: 13px;">${identity}</span>
                        <span style="font-size: 10px; color: var(--text-tertiary); font-family: monospace;">ID: ${t.id}</span>
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        ${t.emotion ? `<span class="status-badge status-success" style="font-size: 10px; padding: 1px 6px;">${t.emotion}</span>` : ''}
                        ${t.action ? `<span class="status-badge ${isAlert ? 'status-error' : 'status-warning'}" style="font-size: 10px; padding: 1px 6px;">${t.action}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }
}

function resetStats() {
    ['stat-persons', 'stat-entries', 'stat-exits'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = '0';
    });
    const fps = document.getElementById('stat-fps');
    if (fps) fps.innerText = '—';
    const motion = document.getElementById('stat-motion');
    if (motion) motion.innerText = '—';
}

// --- View Toggles ---

function showGrid() {
    const grid = document.getElementById('camera-grid-view');
    const focused = document.getElementById('focused-view');
    if (grid) grid.style.display = 'block';
    if (focused) focused.style.display = 'none';
}

function showFocused() {
    const grid = document.getElementById('camera-grid-view');
    const focused = document.getElementById('focused-view');
    if (grid) grid.style.display = 'none';
    if (focused) focused.style.display = 'flex';
}
function toggleGridFullscreen() {
    const area = document.querySelector('.video-area');
    if (!document.fullscreenElement) {
        area.requestFullscreen().catch(err => console.error(err));
    } else {
        document.exitFullscreen();
    }
}

// --- Toggles & Alerts ---

async function sendManualAlert() {
    const btn = document.querySelector('.btn-primary');
    if (!btn) return;
    const original = btn.innerHTML;
    btn.innerHTML = 'Sending...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/notify_manual', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            btn.innerHTML = 'Sent Successfully!';
            btn.style.color = 'var(--success)';
        } else {
            btn.innerHTML = 'Failed';
        }
    } catch (e) {
        btn.innerHTML = 'Error';
    }

    setTimeout(() => {
        btn.innerHTML = original;
        btn.disabled = false;
        btn.style.color = '';
    }, 2000);
}
