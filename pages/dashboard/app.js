/**
 * LivingMemory Dashboard - 主入口
 * 使用模块化架构，保持主文件简洁清晰
 */

import {
  ApiClient,
  PeekPanel,
  MemoryPage,
  TimelinePage,
  RecallPage,
  SystemPage,
  PromptPage,
  esc,
  statusPill,
  nodeBadge,
} from "./modules/index.js";

(() => {
  "use strict";

  /* ================================================================
     State
     ================================================================ */
  const state = {
    page: "graph",
    memory: {
      items: [],
      total: 0,
      page: 1,
      pageSize: 20,
      hasMore: false,
      keyword: "",
      session: "",
      status: "all",
      type: "all",
      sort: "created_desc",
    },
    selectedMemory: null,
    isEditing: false,
    _detailCache: null,
    _nodeDetailCache: null,
    _recallCache: null,
    _systemCache: null,
    pendingSearch: null,
  };

  /* ================================================================
     Initialize Modules
     ================================================================ */
  const api = new ApiClient();
  const peekPanel = new PeekPanel(state, api);
  const memoryPage = new MemoryPage(state, api, peekPanel);
  const timelinePage = new TimelinePage(state, api, peekPanel);
  const recallPage = new RecallPage(state, api, peekPanel);
  const systemPage = new SystemPage(state, api);
  const promptPage = new PromptPage(state, api);

  /* ================================================================
     Theme Management（三档：自动 / 白天 / 黑夜）
     自动模式按本地时间切换，默认 7:00–19:00 为白天，
     可用 localStorage 覆盖：lmem_theme_day_start / lmem_theme_day_end
     ================================================================ */
  const THEME_MODE_KEY = "lmem_theme_mode";
  const DAY_START_KEY = "lmem_theme_day_start";
  const DAY_END_KEY = "lmem_theme_day_end";

  /* 内存中的主题模式（唯一事实源）：
     沙箱环境下 localStorage 可能写入失败，若切换后回读存储会导致
     永远卡在 auto——所以 setThemeMode 直接更新内存变量。 */
  let themeMode = "auto";

  function loadThemeMode() {
    try {
      const saved = localStorage.getItem(THEME_MODE_KEY);
      if (saved === "light" || saved === "dark" || saved === "auto") {
        themeMode = saved;
        return true; // 本地存储命中（普通浏览器环境）
      }
    } catch (e) { /* ignore */ }
    // 插件页 iframe 是沙箱 origin（sandbox 无 allow-same-origin），
    // localStorage 读写会抛异常 → 返回 false，由 init 从后端补拉
    return false;
  }

  function getThemeMode() {
    return themeMode;
  }

  function getDayWindow() {
    let start = 7;
    let end = 19;
    try {
      const s = parseInt(localStorage.getItem(DAY_START_KEY), 10);
      const e = parseInt(localStorage.getItem(DAY_END_KEY), 10);
      if (s >= 0 && s <= 23) start = s;
      if (e >= 1 && e <= 24) end = e;
    } catch (e) { /* ignore */ }
    return { start, end };
  }

  function resolveAutoTheme() {
    const { start, end } = getDayWindow();
    const hour = new Date().getHours();
    // 支持跨零点窗口（如 20-7）
    const isDay = start <= end ? (hour >= start && hour < end) : (hour >= start || hour < end);
    return isDay ? "light" : "dark";
  }

  function resolveTheme() {
    const mode = getThemeMode();
    return mode === "auto" ? resolveAutoTheme() : mode;
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const darkIcon = document.getElementById("theme-icon-dark");
    const lightIcon = document.getElementById("theme-icon-light");
    if (darkIcon && lightIcon) {
      darkIcon.classList.toggle("hidden", theme === "light");
      lightIcon.classList.toggle("hidden", theme === "dark");
    }
  }

  function applyThemeMode() {
    applyTheme(resolveTheme());
    updateThemeMenu();
  }

  /* 沙箱 iframe 里 localStorage 不可用：经桥接存到插件后端持久化
     （本地存储可用时双写，下次加载本地优先） */
  function persistThemeMode(mode) {
    try {
      localStorage.setItem(THEME_MODE_KEY, mode);
    } catch (e) {
      console.warn("[LM] localStorage unavailable, persist via backend:", e.message);
    }
    if (api && typeof api.post === "function") {
      api.post("ui_pref/update", { theme_mode: mode }).catch(() => {});
    }
  }

  function setThemeMode(mode) {
    themeMode = mode;
    persistThemeMode(mode);
    applyTheme(mode === "auto" ? resolveAutoTheme() : mode);
    updateThemeMenu();
    const toastKey = mode === "auto" ? "theme.autoToast"
      : mode === "light" ? "theme.lightToast" : "theme.darkToast";
    showToast(window.t(toastKey));
  }

  function updateThemeMenu() {
    const mode = getThemeMode();
    document.querySelectorAll("#theme-menu .lang-option[data-mode]").forEach(option => {
      const active = option.dataset.mode === mode;
      option.classList.toggle("active", active);
      option.setAttribute("aria-current", active ? "true" : "false");
    });
  }

  /* ================================================================
     Toast Notification
     ================================================================ */
  let toastTimer;
  function showToast(msg, isError = false) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.remove("visible", "error");
    if (isError) el.classList.add("error");
    void el.offsetWidth;
    el.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.classList.remove("visible");
    }, 2500);
  }

  /* ================================================================
     Sidebar / Routing
     ================================================================ */
  function switchPage(name) {
    state.page = name;

    document.querySelectorAll(".nav-item[data-page]").forEach(item => {
      item.classList.toggle("active", item.dataset.page === name);
    });

    document.querySelectorAll(".page").forEach(p => {
      p.classList.toggle("active", p.id === "page-" + name);
    });

    if (name === "graph") {
      fetchGraphStats();
      if (window.ensureGraphScene) window.ensureGraphScene();
    }
    if (name === "timeline") timelinePage.fetch();
    if (name === "memory") memoryPage.fetch();
    if (name === "recall") { /* 召回页面按需加载 */ }
    if (name === "system") systemPage.fetch();
    if (name === "prompts") promptPage.fetch();
  }

  function normalizeLocale(locale) {
    const lang = String(locale || "").split("-")[0];
    return ["zh", "en", "ru"].includes(lang) ? lang : "zh";
  }

  function getCurrentLanguage() {
    if (typeof window.getLanguage === "function") {
      return window.getLanguage();
    }
    return normalizeLocale(api.bridge?.getLocale?.());
  }

  function initSidebar() {
    document.querySelectorAll(".nav-item[data-page]").forEach(item => {
      item.addEventListener("click", () => {
        switchPage(item.dataset.page);
      });
    });

    // 主题三档菜单
    const themeMenu = document.getElementById("theme-menu");
    document.querySelectorAll("#theme-menu .lang-option[data-mode]").forEach(option => {
      option.addEventListener("click", () => {
        const mode = option.dataset.mode;
        if (!mode) return;
        setThemeMode(mode);
        if (themeMenu) themeMenu.removeAttribute("open");
      });
    });
    if (themeMenu) {
      themeMenu.addEventListener("toggle", (e) => {
        if (e.newState === "open") updateThemeMenu();
      });
    }

    const langMenu = document.getElementById("lang-menu");
    document.querySelectorAll(".lang-option[data-lang]").forEach(option => {
      option.addEventListener("click", () => {
        const lang = option.dataset.lang;
        if (!lang) return;

        if (typeof window.setLanguage === "function") {
          window.setLanguage(lang, { persist: true, source: "user" });
        } else {
          try {
            localStorage.setItem("lmem_lang", lang);
          } catch (e) {
            console.warn("[LM] Failed to save language to localStorage:", e);
          }
        }

        if (langMenu) {
          langMenu.removeAttribute("open");
        }
        if (typeof window.setLanguage !== "function") {
          refreshDynamicI18n();
        }
        showToast(window.t("language.toast", option.textContent.trim()));
      });
    });

    if (langMenu) {
      langMenu.addEventListener("toggle", (e) => {
        if (e.newState === "open") {
          updateLanguageMenu();
        }
      });
    }
  }

  function updateLanguageMenu() {
    const currentLang = getCurrentLanguage();

    document.querySelectorAll(".lang-option[data-lang]").forEach(option => {
      const active = option.dataset.lang === currentLang;
      option.classList.toggle("active", active);
      option.setAttribute("aria-current", active ? "true" : "false");
    });
  }

  function refreshDynamicI18n() {
    updateLanguageMenu();
    updateThemeMenu();

    if (state.page === "memory") {
      memoryPage.renderVirtual();
      memoryPage.updatePagination();
    }
    if (state.page === "timeline" && timelinePage.hasData()) {
      timelinePage.render();
    }
    if (state.page === "recall" && state._recallCache) {
      recallPage.renderResults(state._recallCache.data, state._recallCache.elapsed);
    }
    if (state.page === "system" && state._systemCache) {
      systemPage.render(state._systemCache.data);
    }

    const peekPanelEl = document.getElementById("peek-panel");
    const peekVisible = peekPanelEl && peekPanelEl.classList.contains("visible");
    if (peekVisible && !state.isEditing) {
      if (state._detailCache) {
        peekPanel.renderDetailView(state._detailCache);
      } else if (state._nodeDetailCache) {
        peekPanel.renderNode(state._nodeDetailCache);
      }
    }
  }

  /* ================================================================
     Graph Page (依赖 graph-ui.js)
     ================================================================ */
  async function fetchGraphStats() {
    try {
      const data = await api.get("stats");

      document.getElementById("gs-total").textContent = data.total_memories || 0;
      document.getElementById("gs-nodes").textContent = data.graph_nodes || 0;
      document.getElementById("gs-edges").textContent = data.graph_edges || 0;

      const sessions = data.sessions || {};
      const sessionCount = typeof sessions === "object" ? Object.keys(sessions).length : 0;
      document.getElementById("gs-sessions").textContent = sessionCount;
    } catch (e) {
      showToast(e.message || window.t("misc.statsFail"), true);
    }
  }

  /* ================================================================
     Initialization
     ================================================================ */
  async function init() {
    const context = await api.ready();

    if (api.bridge && typeof api.bridge.onContext === "function") {
      api.bridge.onContext((ctx) => {
        if (ctx && ctx.locale) {
          updateLanguageMenu();
        }
      });
    }

    // 主题：三档模式（默认自动昼夜），仪表盘推送的 isDark 不再直接覆盖用户偏好
    const hasLocalTheme = loadThemeMode();
    applyThemeMode();

    // 主题守卫：仪表盘侧 plugin_page_bridge.js 收到 context 推送（含 isDark）
    // 时会把 data-theme 直接强写成仪表盘自身的明暗，覆盖页面设置的主题。
    // 监听属性变化，一旦被外部改掉就立刻改回用户选择的主题；
    // 页面自身 applyTheme 写入的值与 resolveTheme() 一致，不会形成循环。
    if (typeof MutationObserver === "function") {
      new MutationObserver(() => {
        const current = document.documentElement.getAttribute("data-theme");
        if (current && current !== resolveTheme()) {
          applyTheme(resolveTheme());
        }
      }).observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });
    }

    // 沙箱 iframe 里 localStorage 读不到已存模式：从后端补拉（本地命中则本地优先）
    if (!hasLocalTheme && typeof api.get === "function") {
      api.get("ui_pref").then((data) => {
        const saved = data && data.theme_mode;
        if (
          (saved === "light" || saved === "dark" || saved === "auto") &&
          saved !== themeMode
        ) {
          themeMode = saved;
          applyThemeMode();
        }
      }).catch(() => { /* 后端不可用时保持默认 auto */ });
    }

    // 自动模式下每 5 分钟复查一次（页面久开时跨过昼夜分界自动切换）
    setInterval(() => {
      if (getThemeMode() === "auto") applyTheme(resolveTheme());
    }, 5 * 60 * 1000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && getThemeMode() === "auto") applyTheme(resolveTheme());
    });

    initSidebar();

    memoryPage.initEventListeners();
    timelinePage.initEventListeners();
    recallPage.initEventListeners();
    systemPage.initEventListeners();

    document.getElementById("peek-close").addEventListener("click", () => peekPanel.close());
    document.getElementById("peek-overlay").addEventListener("click", () => peekPanel.close());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        peekPanel.close();
      }
    });
    window.addEventListener("languagechange", refreshDynamicI18n);

    fetchGraphStats();
    switchPage("graph");
  }

  /* ================================================================
     Global Exports (for graph-ui.js and other dependencies)
     ================================================================ */
  window.lmState = state;
  window.lmShowToast = showToast;
  window.lmApiRequest = (path, options) => {
    // 兼容旧 API，转发到 ApiClient
    if (options && options.method === "POST") {
      return api.request(path, options);
    }
    return api.request(path, options || {});
  };
  window.lmOpenPeekNode = (nodeData) => peekPanel.renderNode(nodeData);
  window.lmOpenPeekMemory = (memory) => peekPanel.renderMemory(memory);
  window.lmClosePeek = () => peekPanel.close();
  window.lmSwitchPage = (name) => switchPage(name);
  window.lmFetchGraphStats = fetchGraphStats;
  window.lmRefreshMemories = async () => {
    await memoryPage.fetch();
    timelinePage.invalidate();
  };
  window.lmEsc = esc;
  window.lmStatusPill = statusPill;
  window.lmNodeBadge = nodeBadge;

  // 图谱小视图绘制函数（如果需要）
  window.lmDrawMiniGraph = (canvas, nodes, edges) => {
    if (!canvas || !nodes || !nodes.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // 简单布局算法
    const positions = nodes.map((node, i) => {
      const angle = (i / nodes.length) * Math.PI * 2;
      const r = Math.min(W, H) * 0.3;
      return {
        x: W / 2 + r * Math.cos(angle),
        y: H / 2 + r * Math.sin(angle),
        node
      };
    });

    // 绘制边
    if (edges && edges.length) {
      ctx.strokeStyle = "rgba(100, 100, 100, 0.3)";
      ctx.lineWidth = 1;
      edges.forEach(edge => {
        const source = positions.find(p => p.node.id === edge.source || p.node.id === edge.from);
        const target = positions.find(p => p.node.id === edge.target || p.node.id === edge.to);
        if (source && target) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }
      });
    }

    // 绘制节点
    positions.forEach(pos => {
      ctx.fillStyle = "#4a90e2";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  // 启动应用
  init();
})();
