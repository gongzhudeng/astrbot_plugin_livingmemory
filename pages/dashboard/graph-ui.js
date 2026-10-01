(() => {
  "use strict";

  /* ================================================================
     Graph UI — 2D Knowledge Graph Controller
     Bridges page_api data → Graph2D renderer
     ================================================================ */

  const state = {
    payload: null,
    graphIndex: null,
    selectedNodeId: null,
    selectedMemoryId: null,
    isLoading: false,
    hasLoadedOverview: false,
    isGraphReady: false,
  };

  const dom = {};

  /* Node type config */
  var NODE_TYPE_LABELS = {};
  function initLabels() {
    NODE_TYPE_LABELS = {
      get topic() { return window.t("graph.nodeTopic"); },
      get person() { return window.t("graph.nodePerson"); },
      get fact() { return window.t("graph.nodeFact"); },
      get summary() { return window.t("graph.nodeSummary"); },
    };
  }

  const NODE_TYPE_COLORS = {
    topic: "#6f9df0", person: "#7d8bff", fact: "#4fd8ce",
    summary: "#a88fff", other: "#9aa3b8",
  };

  /* ================================================================
     Bridge helpers
     ================================================================ */
  function buildEndpoint(path) {
    var cleanPath = String(path).replace(/^\/+/, "");
    if (cleanPath.startsWith("page/")) return cleanPath;
    return ("page/" + cleanPath).replace(/\/+/g, "/");
  }

  function numericId(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === "string" && !value.trim()) return null;
    var id = Number(value);
    return Number.isFinite(id) ? id : null;
  }

  async function requestGraph(path, options) {
    options = options || {};
    var bridge = window.AstrBotPluginPage;
    if (!bridge) throw new Error(window.t("graph.bridgeError"));

    var method = (options.method || "GET").toUpperCase();
    if (method === "GET") {
      var qi = path.indexOf("?");
      if (qi !== -1) {
        var base = path.substring(0, qi);
        var qs = path.substring(qi + 1);
        var params = {};
        new URLSearchParams(qs).forEach(function(v, k) { params[k] = v; });
        return unwrapGraphData(await bridge.apiGet(buildEndpoint(base), params));
      }
      return unwrapGraphData(await bridge.apiGet(buildEndpoint(path), {}));
    }
    return unwrapGraphData(await bridge.apiPost(buildEndpoint(path), options.body || {}));
  }

  function unwrapGraphData(response) {
    if (response && response.status === "ok" && Object.prototype.hasOwnProperty.call(response, "data")) {
      return response.data || {};
    }
    if (response && response.status === "error") {
      throw new Error(response.message || window.t("misc.requestFailed"));
    }
    return response || {};
  }

  /* ================================================================
     记忆碎片层：碎片开关 + 文字池（来自真实记忆摘要/条目/节点名）
     ================================================================ */
  var FRAG_KEY = "lmem_fragments";
  var fragmentsOn = true;

  function updateFragmentsBtn(btn) {
    btn.classList.toggle("on", fragmentsOn);
    btn.textContent = window.t(fragmentsOn ? "graph.fragmentsOn" : "graph.fragmentsOff");
    btn.title = window.t("graph.fragmentsTip");
  }

  function initFragmentsToggle() {
    var btn = document.getElementById("graph-fragments-btn");
    if (!btn) return;
    try {
      fragmentsOn = localStorage.getItem(FRAG_KEY) !== "off";
    } catch (e) { /* ignore */ }
    window.Graph2D.setFragmentsEnabled(fragmentsOn);
    updateFragmentsBtn(btn);

    btn.addEventListener("click", function() {
      fragmentsOn = !fragmentsOn;
      try {
        localStorage.setItem(FRAG_KEY, fragmentsOn ? "on" : "off");
      } catch (e) { /* ignore */ }
      window.Graph2D.setFragmentsEnabled(fragmentsOn);
      updateFragmentsBtn(btn);
    });
  }

  function buildFragmentPool(payload) {
    var snapshot = (payload && payload.snapshot) || {};
    /* 优先用真实记忆摘要/条目内容，节点名（常为英文实体词）只少量补充 */
    var pool = [];
    (snapshot.memories || []).forEach(function(m) {
      var t = m.summary || m.text || m.content || "";
      if (t) pool.push(t);
    });
    var memoCount = pool.length;
    (snapshot.entries || []).forEach(function(en) {
      var t = en.summary || en.text || en.content || "";
      if (t) pool.push(t);
    });
    var contentCount = pool.length;
    (snapshot.nodes || []).forEach(function(n) {
      if (n.label && pool.length - contentCount < 20) pool.push(n.label);
    });
    /* 若记忆内容太少，用节点名垫底；否则截断在内容类文本之后 */
    var cap = Math.max(contentCount + 20, memoCount + 40);
    return pool.slice(0, Math.min(cap, 220));
  }

  /* 供阅读层「图谱定位」调用：按记忆 ID 聚焦关系子图 */
  window.lmGotoGraphFocus = function(memoryId) {
    var id = Number(memoryId);
    if (!Number.isFinite(id)) return;
    if (dom.memoryInput) dom.memoryInput.value = String(id);
    focusMemory();
  };

  /* ================================================================
     Init
     ================================================================ */
  function init() {
    initLabels();
    initFragmentsToggle();
    dom.queryInput = document.getElementById("graph-query-input");
    dom.sessionInput = document.getElementById("graph-session-filter");
    dom.memoryInput = document.getElementById("graph-memory-id");
    dom.searchButton = document.getElementById("graph-search-btn");
    dom.focusButton = document.getElementById("graph-focus-btn");
    dom.overviewButton = document.getElementById("graph-overview-btn");
    dom.legend = document.getElementById("graph-legend");
    dom.canvas = document.getElementById("graph-canvas");
    dom.canvasState = document.getElementById("graph-canvas-state");

    if (!dom.canvas) return;

    dom.searchButton.addEventListener("click", runQuery);
    dom.focusButton.addEventListener("click", focusMemory);
    dom.overviewButton.addEventListener("click", fetchOverview);

    dom.queryInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter") { e.preventDefault(); runQuery(); }
    });
    dom.memoryInput.addEventListener("keydown", function(e) {
      if (e.key === "Enter") { e.preventDefault(); focusMemory(); }
    });

    /* Init Graph2D */
    if (window.Graph2D && typeof window.Graph2D.init === "function") {
      window.Graph2D.init(dom.canvas, {
        onNodeClick: function(nodeId) {
          selectNode(nodeId, false);
        },
        onNodeDblClick: function(nodeId) {
          selectNode(nodeId, true);
        },
        onNodeHover: function(nodeId) {
          /* Tooltip could go here */
        },
        onBackgroundClick: function() {
          clearSelection();
        },
      });
      state.isGraphReady = true;
      /* 渲染器刚创建，重新应用碎片开关的存储偏好 */
      window.Graph2D.setFragmentsEnabled(fragmentsOn);
      setCanvasMessage(window.t("graph.canvasDefault"), false);
    } else {
      setCanvasMessage(window.t("graph2d.moduleFail"), false);
    }

    window.addEventListener("languagechange", function() {
      initLabels();
      if (state.payload) renderLegend(state.payload);
      var fragBtn = document.getElementById("graph-fragments-btn");
      if (fragBtn) updateFragmentsBtn(fragBtn);
      if (!state.payload && dom.canvasState && dom.canvasState.textContent) {
        setCanvasMessage(window.t("graph.canvasDefault"), false);
      }
    });

    /* Auto-load overview */
    setTimeout(function() {
      if (!state.hasLoadedOverview && !state.isLoading) {
        fetchOverview();
      }
    }, 100);
  }

  /* Expose for app.js lazy-load */
  window.ensureGraphScene = function() {
    if (!state.hasLoadedOverview && !state.isLoading) fetchOverview();
  };

  /* ================================================================
     Data Fetching
     ================================================================ */
  function getFilters() {
    return {
      session_id: dom.sessionInput ? dom.sessionInput.value.trim() || null : null,
    };
  }

  function addOptionalFilter(body, key, value) {
    if (value !== null && value !== undefined && String(value).trim()) {
      body[key] = String(value).trim();
    }
    return body;
  }

  async function fetchOverview() {
    setLoading(true);
    try {
      var filters = getFilters();
      var params = new URLSearchParams();
      if (filters.session_id) params.set("session_id", filters.session_id);
      var qs = params.toString();
      var payload = await requestGraph("/graph/overview" + (qs ? "?" + qs : ""));
      state.hasLoadedOverview = true;
      renderPayload(payload, true);
      if (window.lmFetchGraphStats) window.lmFetchGraphStats();
    } catch (e) {
      setCanvasMessage(e.message || window.t("graph.errorFetch"), false);
    } finally {
      setLoading(false);
    }
  }

  async function runQuery() {
    var query = dom.queryInput.value.trim();
    if (!query) { fetchOverview(); return; }

    setLoading(true);
    try {
      var filters = getFilters();
      var body = addOptionalFilter({ query: query }, "session_id", filters.session_id);
      var payload = await requestGraph("/graph/query", {
        method: "POST",
        body: body,
      });
      renderPayload(payload, true);
    } catch (e) {
      setCanvasMessage(e.message || window.t("graph.queryFail"), false);
    } finally {
      setLoading(false);
    }
  }

  async function focusMemory() {
    var text = dom.memoryInput.value.trim();
    if (!text) { setCanvasMessage(window.t("graph.focusEmpty"), false); return; }
    var memoryId = Number.parseInt(text, 10);
    if (Number.isNaN(memoryId)) { setCanvasMessage(window.t("graph.focusNotInt"), false); return; }

    setLoading(true);
    try {
      var filters = getFilters();
      var body = addOptionalFilter({ memory_id: memoryId }, "session_id", filters.session_id);
      var payload = await requestGraph("/graph/query", {
        method: "POST",
        body: body,
      });
      renderPayload(payload, true);
    } catch (e) {
      setCanvasMessage(e.message || window.t("graph.focusFail"), false);
    } finally {
      setLoading(false);
    }
  }

  /* ================================================================
     Render Pipeline
     ================================================================ */
  function setLoading(loading) {
    state.isLoading = loading;
    if (dom.searchButton) dom.searchButton.disabled = loading;
    if (dom.focusButton) dom.focusButton.disabled = loading;
    if (dom.overviewButton) dom.overviewButton.disabled = loading;
  }

  function renderPayload(payload, focusSelection) {
    state.payload = payload;
    state.graphIndex = buildGraphIndex(payload.snapshot || {});

    /* 喂给碎片层一批真实记忆文字 */
    if (window.Graph2D && window.Graph2D.setFragmentPool) {
      window.Graph2D.setFragmentPool(buildFragmentPool(payload));
    }

    if (!payload.enabled) {
      setCanvasMessage(window.t("graph.disabledCanvas"), false);
      renderLegend(payload);
      return;
    }

    var snapshot = payload.snapshot || {};
    var hasGraphData = Boolean((snapshot.nodes || []).length);
    var shouldFocusSelection = hasGraphData &&
      focusSelection !== false &&
      payload.mode !== "overview";

    if (!shouldFocusSelection) {
      state.selectedNodeId = null;
      state.selectedMemoryId = null;
      if (state.isGraphReady && window.Graph2D) {
        window.Graph2D.selection = null;
        if (window.Graph2D.renderer) window.Graph2D.renderer._selection = null;
      }
    }

    /* Delegate to Graph2D renderer */
    if (state.isGraphReady) {
      window.Graph2D.loadData(payload);
      setCanvasMessage(hasGraphData ? "" : window.t("graph.canvasEmpty"), false);
    }

    /* Auto-select based on payload mode */
    if (shouldFocusSelection) {
      ensureSelection(payload);
    } else if (!hasGraphData) {
      state.selectedNodeId = null;
      state.selectedMemoryId = null;
    }

    renderLegend(payload);

    /* Apply selection to Graph2D */
    if (state.isGraphReady) {
      if (state.selectedNodeId !== null) {
        window.Graph2D.selectNode(state.selectedNodeId);
      } else if (state.selectedMemoryId !== null) {
        window.Graph2D.selectMemory(state.selectedMemoryId);
      }
    }

    if (window.lmFetchGraphStats) window.lmFetchGraphStats();
  }

  /* ================================================================
     Selection Logic
     ================================================================ */
  function ensureSelection(payload) {
    var selectedNodeId = numericId(state.selectedNodeId);
    if (state.graphIndex && selectedNodeId !== null && state.graphIndex.nodeMap.has(selectedNodeId)) {
      state.selectedNodeId = selectedNodeId;
      state.selectedMemoryId = null;
      return;
    }
    var selectedMemoryId = numericId(state.selectedMemoryId);
    if (state.graphIndex && selectedMemoryId !== null && state.graphIndex.memoryMap.has(selectedMemoryId)) {
      state.selectedNodeId = null;
      state.selectedMemoryId = selectedMemoryId;
      return;
    }

    state.selectedNodeId = null;
    state.selectedMemoryId = null;

    var matchedNodeIds = payload.matched_node_ids || [];
    var firstNode = matchedNodeIds.find(function(id) {
      var nodeId = numericId(id);
      return nodeId !== null && state.graphIndex && state.graphIndex.nodeMap.has(nodeId);
    });
    if (firstNode !== undefined) {
      state.selectedNodeId = numericId(firstNode);
      return;
    }

    var firstRetrieved = payload.retrieval && payload.retrieval.items && payload.retrieval.items[0];
    var firstRetrievedId = firstRetrieved ? numericId(firstRetrieved.memory_id) : null;
    if (firstRetrievedId !== null && state.graphIndex && state.graphIndex.memoryMap.has(firstRetrievedId)) {
      state.selectedMemoryId = firstRetrievedId;
      return;
    }

    var topNodes = payload.top_nodes || [];
    var topNodeId = topNodes.length ? numericId(topNodes[0].id) : null;
    if (topNodeId !== null && state.graphIndex && state.graphIndex.nodeMap.has(topNodeId)) {
      state.selectedNodeId = topNodeId;
      return;
    }

    var snapMemories = (payload.snapshot && payload.snapshot.memories) || [];
    var snapMemoryId = snapMemories.length ? numericId(snapMemories[0].memory_id) : null;
    if (snapMemoryId !== null && state.graphIndex && state.graphIndex.memoryMap.has(snapMemoryId)) {
      state.selectedMemoryId = snapMemoryId;
    }
  }

  function clearSelection() {
    if (!state.payload) return;
    state.selectedNodeId = null;
    state.selectedMemoryId = null;
    if (state.isGraphReady) window.Graph2D.clearSelection();
    if (window.lmClosePeek) window.lmClosePeek();
  }

  function selectNode(nodeId, focusCamera) {
    nodeId = numericId(nodeId);
    if (nodeId === null) return;
    if (!state.graphIndex || !state.graphIndex.nodeMap.has(nodeId)) return;
    state.selectedNodeId = nodeId;
    state.selectedMemoryId = null;

    if (state.isGraphReady) {
      window.Graph2D.selectNode(nodeId);
    }

    /* Show in peek panel */
    var node = state.graphIndex.nodeMap.get(nodeId);
    if (window.lmOpenPeekNode && node) window.lmOpenPeekNode(node);
  }

  function selectMemory(memoryId) {
    memoryId = numericId(memoryId);
    if (memoryId === null) return;
    if (!state.graphIndex || !state.graphIndex.memoryMap.has(memoryId)) return;
    state.selectedMemoryId = memoryId;
    state.selectedNodeId = null;

    if (state.isGraphReady) {
      window.Graph2D.selectMemory(memoryId);
    }

    /* Show in peek panel */
    var memory = state.graphIndex.memoryMap.get(memoryId);
    if (window.lmState && memory) {
      var item = {
        memory_id: memoryId,
        summary: memory.summary || memory.content || "",
        content: memory.content || memory.summary || "",
        memory_type: memory.memory_type || "",
        importance: memory.importance,
        status: memory.status || "active",
        raw: memory,
      };
      if (window.lmOpenPeekMemory) window.lmOpenPeekMemory(item);
    }
  }

  /* ================================================================
     Legend
     ================================================================ */
  function renderLegend(payload) {
    if (!dom.legend) return;
    var summary = payload.summary || {};
    var nodeTypes = summary.node_type_breakdown || {};
    var relTypes = summary.relation_breakdown || {};

    var chips = Object.entries(nodeTypes).sort(function(a, b) { return b[1] - a[1]; })
      .map(function(e) {
        return '<span class="legend-chip"><span class="dot" style="background:' +
          (NODE_TYPE_COLORS[e[0]] || NODE_TYPE_COLORS.other) + '"></span>' +
          typeLabel(e[0]) + ' &middot; ' + e[1] + '</span>';
      });

    var rchips = Object.entries(relTypes).sort(function(a, b) { return b[1] - a[1]; }).slice(0, 4)
      .map(function(e) {
        return '<span class="legend-chip">' + relationLabel(e[0]) + ' &middot; ' + e[1] + '</span>';
      });

    dom.legend.innerHTML = chips.concat(rchips).join("") ||
      '<span class="legend-chip">' + window.t("graph.legendEmpty") + '</span>';
  }

  /* ================================================================
     Graph Index
     ================================================================ */
  function buildGraphIndex(snapshot) {
    var nodes = snapshot.nodes || [];
    var edges = snapshot.edges || [];
    var entries = snapshot.entries || [];
    var memories = snapshot.memories || [];

    var nodeMap = new Map(nodes.map(function(n) { return [Number(n.id), n]; }));
    var memoryMap = new Map(memories.map(function(m) { return [Number(m.memory_id), m]; }));
    var memoryToEntries = new Map();
    var memoryToNodes = new Map();
    var nodeToMemories = new Map();
    var nodeToEntries = new Map();
    var neighborMap = new Map();

    function ensureSet(map, key) {
      if (!map.has(key)) map.set(key, new Set());
      return map.get(key);
    }

    entries.forEach(function(entry) {
      var mId = Number(entry.memory_id);
      if (!memoryToEntries.has(mId)) memoryToEntries.set(mId, []);
      memoryToEntries.get(mId).push(entry);
      (entry.node_ids || []).forEach(function(nId) {
        var nodeId = Number(nId);
        ensureSet(memoryToNodes, mId).add(nodeId);
        ensureSet(nodeToMemories, nodeId).add(mId);
        if (!nodeToEntries.has(nodeId)) nodeToEntries.set(nodeId, []);
        nodeToEntries.get(nodeId).push(entry);
      });
    });

    edges.forEach(function(edge) {
      var s = Number(edge.source);
      var t = Number(edge.target);
      var mId = Number(edge.memory_id);
      ensureSet(memoryToNodes, mId).add(s);
      ensureSet(memoryToNodes, mId).add(t);
      ensureSet(nodeToMemories, s).add(mId);
      ensureSet(nodeToMemories, t).add(mId);
      ensureSet(neighborMap, s).add(t);
      ensureSet(neighborMap, t).add(s);
    });

    return {
      nodeMap: nodeMap, memoryMap: memoryMap,
      memoryToEntries: memoryToEntries, memoryToNodes: memoryToNodes,
      nodeToMemories: nodeToMemories, nodeToEntries: nodeToEntries,
      neighborMap: neighborMap, edges: edges, entries: entries,
    };
  }

  /* ================================================================
     Canvas Messages
     ================================================================ */
  function setCanvasMessage(msg, loading) {
    if (!dom.canvasState) return;
    dom.canvasState.textContent = msg || "";
    dom.canvasState.style.display = msg ? "block" : "none";
  }

  /* ================================================================
     Labels
     ================================================================ */
  function typeLabel(t) { return NODE_TYPE_LABELS[t] || t || window.t("graph.nodeUnknown"); }
  function relationLabel(t) { return String(t || "related").replace(/_/g, " ").replace(/\b\w/g, function(c) { return c.toUpperCase(); }); }

  document.addEventListener("DOMContentLoaded", init);
})();
