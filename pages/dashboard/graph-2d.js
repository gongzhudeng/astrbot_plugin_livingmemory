(() => {
  "use strict";

  /* ================================================================
     Graph2D — Centered Knowledge Graph
     Center node anchors an organic force-directed canvas
     ================================================================ */

  /* ── Configuration ─────────────────────────────────────────── */
  const CFG = {
    /* 恒星节点：三层画法（白热内核→类型色冕→透明渐变），核心半径收紧 */
    NODE_RADIUS_MIN: 2.6,
    NODE_RADIUS_MAX: 6.5,
    NODE_RADIUS_BASE: 2.6,
    CENTER_SCALE: 1.5,
    CENTER_MAX_RADIUS: 9.5,
    NODE_FONT_SIZE: 11,
    NODE_META_SIZE: 9,
    EDGE_WIDTH_DEFAULT: 0.55,
    EDGE_WIDTH_ACTIVE: 0.9,
    EDGE_WIDTH_HIGHLIGHT: 1.5,
    EDGE_OPACITY_DEFAULT: 0.13,
    EDGE_OPACITY_ACTIVE: 0.4,
    EDGE_OPACITY_HIGHLIGHT: 0.7,
    /* 流光脉冲（沿连线游走的光点） */
    PULSE_MAX: 16,
    PULSE_SPAWN_MIN: 0.4,
    PULSE_SPAWN_MAX: 1.0,
    PULSE_DUR_MIN: 2.2,
    PULSE_DUR_MAX: 3.8,
    PULSE_TRAIL: 0.09,
    /* 星尘背景 */
    STAR_COUNT: 170,
    /* 全图缓慢漂移（世界坐标振幅） */
    DRIFT_AMP_X: 6,
    DRIFT_AMP_Y: 4.5,
    /* 点击/聚焦涟漪 */
    RIPPLE_LIFE: 1.1,
    RIPPLE_MAX: 6,
    /* 记忆碎片 */
    FRAG_NEAR_MAX: 6,
    FRAG_NEAR_SPAWN_MIN: 0.5,
    FRAG_NEAR_SPAWN_MAX: 1.3,
    FRAG_FADE: 0.7,
    FRAG_HOLD: 2.8,
    FRAG_RISE: 9,
    FRAG_DRIFT_MAX: 3,
    FRAG_DRIFT_SPAWN_MIN: 3,
    FRAG_DRIFT_SPAWN_MAX: 5,
    FRAG_DRIFT_EDGE_FADE: 170,
    FRAG_MAX_CHARS: 18,
    /* Force-directed layout - optimized for natural clustering */
    FORCE_ITERATIONS: 400,
    FORCE_REPULSION: 1800,
    FORCE_LINK_DISTANCE: 120,
    FORCE_LINK_STRENGTH: 0.025,
    FORCE_GRAVITY: 0.008,
    FORCE_DAMPING: 0.82,
    FORCE_MAX_SPEED: 15,
    /* Animation */
    ANIM_SPEED: 0.075,
    IDLE_DAMPING: 0.05,
    ZOOM_MIN: 0.2,
    ZOOM_MAX: 3.5,
    ZOOM_STEP: 0.001,
    DPR_MAX: 2,
    HOVER_RADIUS: 8,
  };

  /* 恒星类型微着色（冷色系、低饱和，与「星夜」单强调色族一致） */
  const TYPE_COLORS = {
    topic: "#6f9df0",   // 主题 · 蓝
    person: "#7d8bff",  // 人物 · 靛
    fact: "#4fd8ce",    // 事实 · 青
    summary: "#a88fff", // 记忆 · 紫
    other: "#9aa3b8",   // 其他 · 中性
  };

  /* ── CSS helpers ───────────────────────────────────────────── */
  function isDark() {
    return (document.documentElement.getAttribute("data-theme") || "light") === "dark";
  }

  /* ── Math helpers ──────────────────────────────────────────── */
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  function themeColor(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function hexToRgba(h, alpha) {
    var v = String(h || "#000").replace("#", "").trim();
    v = v.length === 3 ? v.split("").map(function(c) { return c + c; }).join("") : v.padEnd(6, "0").slice(0, 6);
    var r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16), b = parseInt(v.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + clamp(alpha, 0, 1) + ")";
  }

  /* ── Event helpers ─────────────────────────────────────────── */
  function getPos(e, el) {
    var rect = el.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  /* ═══════════════════════════════════════════════════════════════
     ForceDirectedLayout — true force-directed graph layout
     No fixed center, no BFS rings — nodes repel, edges spring,
     gentle gravity keeps the graph centered.
     ═══════════════════════════════════════════════════════════════ */
  function ForceDirectedLayout() {
    this.centerId = null;   // focus node id (for viewport + visual emphasis)
    this.positions = {};    // id → {tx, ty}
    this.rings = {};        // id → 0 for focus, 1 for others (backward compat)
  }

  ForceDirectedLayout.prototype._hashUnit = function(value, salt) {
    var str = String(value) + ":" + String(salt || 0);
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return ((h >>> 0) % 100000) / 100000;
  };

  ForceDirectedLayout.prototype._layoutRadius = function(node) {
    var w = clamp(Number(node.weight || 0), 0, 20);
    var mr = clamp(Number(node.memory_count || 0), 0, 15);
    var radius = CFG.NODE_RADIUS_BASE + Math.sqrt(w) * 0.45 + Math.sqrt(mr) * 0.24;
    return clamp(radius, CFG.NODE_RADIUS_MIN, CFG.NODE_RADIUS_MAX);
  };

  /* Compute force-directed positions — all nodes are free */
  ForceDirectedLayout.prototype.compute = function(nodes, edges, focusId) {
    var self = this;
    this.positions = {};
    this.rings = {};

    var n = nodes.length;
    if (n === 0) return;
    if (n === 1) {
      this.rings[nodes[0].id] = 0;
      this.positions[nodes[0].id] = { tx: 0, ty: 0 };
      this.centerId = nodes[0].id;
      return;
    }

    /* Seed positions — pseudo-random spiral from node id (deterministic) */
    var sim = nodes.map(function(nd, i) {
      var angle = self._hashUnit(nd.id, 13) * Math.PI * 2;
      var dist = Math.sqrt(self._hashUnit(nd.id, 17)) * 180 + 20;
      return {
        id: nd.id,
        node: nd,
        x: Math.cos(angle) * dist,
        y: Math.sin(angle) * dist,
        vx: 0,
        vy: 0,
        radius: self._layoutRadius(nd),
      };
    });

    var indexById = {};
    sim.forEach(function(s, i) { indexById[s.id] = i; });

    /* Build edge simulation list */
    var simEdges = [];
    edges.forEach(function(edge) {
      var si = indexById[edge.source];
      var ti = indexById[edge.target];
      if (si == null || ti == null) return;
      var weight = clamp(Number(edge.weight || 1), 0.4, 12);
      var confidence = clamp(Number(edge.confidence || 0.8), 0.2, 1);
      simEdges.push({
        source: si,
        target: ti,
        weight: weight,
        confidence: confidence,
        distanceJitter: self._hashUnit(String(edge.id) + ":" + edge.source + ":" + edge.target, 61),
      });
    });

    /* Mark focus node (treated gently in gravity, not locked) */
    var focusIndex = focusId != null ? indexById[focusId] : -1;
    if (focusIndex >= 0) {
      sim[focusIndex].isFocus = true;
      this.centerId = focusId;
    } else {
      this.centerId = null;
    }

    /* ── N-body force simulation ── */
    var iterations = n > 200 ? 300 : n > 100 ? 350 : CFG.FORCE_ITERATIONS;
    for (var step = 0; step < iterations; step++) {
      var alpha = 1 - step / iterations;
      var cooled = 0.3 + alpha * 0.7;

      /* Repulsion between all node pairs with distance-based falloff */
      for (var i = 0; i < sim.length; i++) {
        var a = sim[i];
        for (var j = i + 1; j < sim.length; j++) {
          var b = sim[j];
          var dx = a.x - b.x;
          var dy = a.y - b.y;
          var distSq = dx * dx + dy * dy;
          if (distSq < 0.01) {
            var kick = self._hashUnit(a.id + ":" + b.id, 43) * Math.PI * 2;
            dx = Math.cos(kick) * 0.1;
            dy = Math.sin(kick) * 0.1;
            distSq = dx * dx + dy * dy;
          }
          var dist = Math.sqrt(distSq);
          var effectiveRange = 280 + Math.min(120, n * 1.2);

          var minSep = (a.radius + b.radius) * 2.2 + 16;
          var repulse = CFG.FORCE_REPULSION * cooled / Math.max(distSq, minSep * minSep * 0.25);

          /* Smoother distance falloff - linear instead of sharp cutoff */
          if (dist < effectiveRange) {
            var falloff = 1 - (dist / effectiveRange);
            repulse *= falloff * falloff;
          } else {
            repulse *= 0.05;
          }

          /* Stronger push when nodes are too close */
          if (dist < minSep) {
            repulse += (minSep - dist) * 0.35;
          }

          var fx = dx / dist * repulse;
          var fy = dy / dist * repulse;
          a.vx += fx; a.vy += fy;
          b.vx -= fx; b.vy -= fy;
        }
      }

      /* Spring attraction along edges with adaptive strength */
      simEdges.forEach(function(edge) {
        var s = sim[edge.source];
        var t = sim[edge.target];
        var dx = t.x - s.x;
        var dy = t.y - s.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 0.001;

        /* Base distance varies by edge weight */
        var baseDistance = CFG.FORCE_LINK_DISTANCE + edge.distanceJitter * 40;
        var weightFactor = Math.min(1.5, Math.sqrt(edge.weight || 1) * 0.3);
        var desired = baseDistance - weightFactor * 15;

        /* Adaptive spring strength - weaker for long edges */
        var lengthRatio = dist / desired;
        var adaptiveStrength = CFG.FORCE_LINK_STRENGTH * edge.confidence * cooled;
        if (lengthRatio > 2) {
          adaptiveStrength *= 0.5;
        }

        var force = (dist - desired) * adaptiveStrength;
        var fx = (dx / dist) * force;
        var fy = (dy / dist) * force;
        s.vx += fx; s.vy += fy;
        t.vx -= fx; t.vy -= fy;
      });

      /* Gentle centering gravity with mass-based scaling */
      for (var k = 0; k < sim.length; k++) {
        var sn = sim[k];
        /* Gravity scales with node degree/weight to keep important nodes more central */
        var massFactor = 1 + Math.sqrt(sn.node.weight || 0) * 0.1 + Math.sqrt(sn.node.degree || 0) * 0.05;
        var gravity = CFG.FORCE_GRAVITY * cooled / massFactor;
        if (sn.isFocus) gravity *= 1.8;
        sn.vx -= sn.x * gravity;
        sn.vy -= sn.y * gravity;
      }

      /* Damping + position update */
      sim.forEach(function(sn) {
        sn.vx *= CFG.FORCE_DAMPING;
        sn.vy *= CFG.FORCE_DAMPING;
        var speed = Math.sqrt(sn.vx * sn.vx + sn.vy * sn.vy);
        if (speed > CFG.FORCE_MAX_SPEED) {
          sn.vx = sn.vx / speed * CFG.FORCE_MAX_SPEED;
          sn.vy = sn.vy / speed * CFG.FORCE_MAX_SPEED;
        }
        sn.x += sn.vx;
        sn.y += sn.vy;
      });
    }

    /* Assign rings: 0 = focus node, 1 = others (for backward compat) */
    sim.forEach(function(sn) {
      self.rings[sn.id] = sn.isFocus ? 0 : 1;
      self.positions[sn.id] = { tx: sn.x, ty: sn.y };
    });
  };

  /* Get target position for a node */
  ForceDirectedLayout.prototype.getTarget = function(nodeId) {
    var p = this.positions[nodeId];
    return p || { tx: 0, ty: 0 };
  };

  /* Get ring (0 = focus, 1 = normal) */
  ForceDirectedLayout.prototype.getRing = function(nodeId) {
    return this.rings[nodeId] != null ? this.rings[nodeId] : 1;
  };

  /* ═══════════════════════════════════════════════════════════════
     Renderer — Canvas 2D drawing
     ═══════════════════════════════════════════════════════════════ */
  function Renderer(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.viewport = { ox: 0, oy: 0, scale: 1 };
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this._drawnNodes = [];
    this._drawnEdges = [];
    this._labelBoxes = [];
    this._selection = null;
    /* 星图动态元素 */
    this._stars = null;
    this.pulses = [];
    this.ripples = [];
    this._lastPulseSpawn = 0;
    /* 记忆碎片层（默认开启，graph-ui 会按存储偏好覆盖） */
    this.fragmentsEnabled = true;
    this.fragPool = [];
    this.nearFrags = [];
    this.driftFrags = [];
    this._fragSpawnAt = 0;
    this._driftSpawnAt = 0;
  }

  Renderer.prototype.resize = function() {
    var rect = this.canvas.parentElement.getBoundingClientRect();
    var w = Math.max(1, Math.floor(rect.width || this.canvas.parentElement.clientWidth || 1));
    var h = Math.max(320, Math.floor(rect.height || this.canvas.parentElement.clientHeight || 320));
    this.dpr = Math.min(window.devicePixelRatio || 1, CFG.DPR_MAX);
    this.width = w;
    this.height = h;
    this.canvas.width = w * this.dpr;
    this.canvas.height = h * this.dpr;
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  };

  Renderer.prototype.clear = function() {
    this.ctx.clearRect(0, 0, this.width, this.height);
  };

  Renderer.prototype.drawBackground = function(dark, now) {
    /* 透明画布：透出 CSS 星夜/晨雾渐变与极光光斑 */
    this.ctx.clearRect(0, 0, this.width, this.height);
    this._drawStardust(dark, now);
  };

  /* ── 星尘背景：缓慢闪烁 + 极慢漂移 ── */
  Renderer.prototype._initStardust = function() {
    if (this._stars) return;
    var stars = [];
    for (var i = 0; i < CFG.STAR_COUNT; i++) {
      stars.push({
        x: Math.random(),
        y: Math.random(),
        r: 0.5 + Math.random() * 1.2,
        base: 0.28 + Math.random() * 0.55,
        speed: 0.25 + Math.random() * 0.65,
        phase: Math.random() * Math.PI * 2,
      });
    }
    this._stars = stars;
  };

  Renderer.prototype._drawStardust = function(dark, now) {
    this._initStardust();
    var ctx = this.ctx;
    var w = this.width;
    var h = this.height;
    var dx = Math.sin(now * 0.02) * 9;
    var dy = Math.cos(now * 0.015) * 7;
    for (var i = 0; i < this._stars.length; i++) {
      var s = this._stars[i];
      var tw = 0.62 + 0.38 * Math.sin(now * s.speed + s.phase);
      var a = s.base * tw * (dark ? 1 : 0.55);
      if (a <= 0.01) continue;
      var par = 0.4 + s.r * 0.4;
      ctx.beginPath();
      ctx.arc(s.x * w + dx * par, s.y * h + dy * par, s.r, 0, Math.PI * 2);
      ctx.fillStyle = dark
        ? "rgba(210,222,255," + a.toFixed(3) + ")"
        : "rgba(110,130,200," + (a * 0.8).toFixed(3) + ")";
      ctx.fill();
    }
  };

  Renderer.prototype.worldToScreen = function(wx, wy) {
    return {
      x: (wx + this.viewport.ox) * this.viewport.scale + this.width / 2,
      y: (wy + this.viewport.oy) * this.viewport.scale + this.height / 2,
    };
  };

  Renderer.prototype.screenToWorld = function(sx, sy) {
    return {
      x: (sx - this.width / 2) / this.viewport.scale - this.viewport.ox,
      y: (sy - this.height / 2) / this.viewport.scale - this.viewport.oy,
    };
  };

  Renderer.prototype.nodeWorldRadius = function(nodeData, isCenter) {
    var w = clamp(Number(nodeData.weight || 0), 0, 20);
    var mr = clamp(Number(nodeData.memory_count || 0), 0, 15);
    var r = CFG.NODE_RADIUS_BASE + Math.sqrt(w) * 0.45 + Math.sqrt(mr) * 0.24;
    if (isCenter) {
      r = Math.min(CFG.CENTER_MAX_RADIUS, r * CFG.CENTER_SCALE);
    }
    if (nodeData.isSelected) r += 1.2;
    return clamp(r, CFG.NODE_RADIUS_MIN, isCenter ? CFG.CENTER_MAX_RADIUS : CFG.NODE_RADIUS_MAX);
  };

  Renderer.prototype.nodeScreenRadius = function(nodeData, isCenter) {
    return this.nodeWorldRadius(nodeData, isCenter) * this.viewport.scale;
  };

  Renderer.prototype.render = function(nodes, edges, nodeMap, selection, hoverId, layout, animProgress) {
    var ctx = this.ctx;
    var scale = this.viewport.scale;
    var dark = isDark();
    var now = Date.now() / 1000;
    var selNodeId = (selection && selection.type === "node") ? selection.id : null;
    var selMemId = (selection && selection.type === "memory") ? selection.id : null;

    /* 全图缓慢漂移（所有星点一起轻轻浮动） */
    var driftX = Math.sin(now * 0.05) * CFG.DRIFT_AMP_X;
    var driftY = Math.cos(now * 0.043) * CFG.DRIFT_AMP_Y;

    /* Build highlight sets */
    var highlightNodes = new Set();
    var highlightEdges = new Set();
    var adjacency = {};
    nodes.forEach(function(nd) { adjacency[nd.id] = []; });
    edges.forEach(function(e) {
      if (adjacency[e.source]) adjacency[e.source].push(e.target);
      if (adjacency[e.target]) adjacency[e.target].push(e.source);
    });

    if (selNodeId !== null) {
      highlightNodes.add(selNodeId);
      (adjacency[selNodeId] || []).forEach(function(nid) { highlightNodes.add(nid); });
    }
    if (selMemId !== null) {
      edges.forEach(function(edge) {
        if (edge.memory_id === selMemId) {
          highlightNodes.add(edge.source);
          highlightNodes.add(edge.target);
          highlightEdges.add(edge.id);
        }
      });
    }
    /* 悬停点亮关联子图（无选中焦点时） */
    if (selNodeId === null && selMemId === null && hoverId != null && adjacency[hoverId]) {
      highlightNodes.add(hoverId);
      (adjacency[hoverId] || []).forEach(function(nid) { highlightNodes.add(nid); });
      edges.forEach(function(edge) {
        if (edge.source === hoverId || edge.target === hoverId) highlightEdges.add(edge.id);
      });
    }

    var centerId = layout ? layout.centerId : null;

    this.drawBackground(dark, now);

    /* Compute animated positions */
    var ap = animProgress == null ? 1 : animProgress;

    /* Draw edges first (under nodes) */
    this._drawnEdges = [];
    this._labelBoxes = [];
    ctx.save();
    for (var e = 0; e < edges.length; e++) {
      var edge = edges[e];
      var src = nodeMap[edge.source];
      var tgt = nodeMap[edge.target];
      if (!src || !tgt) continue;

      var sAnim = { x: lerp(src._prevX || src.x, src.x, ap), y: lerp(src._prevY || src.y, src.y, ap) };
      var tAnim = { x: lerp(tgt._prevX || tgt.x, tgt.x, ap), y: lerp(tgt._prevY || tgt.y, tgt.y, ap) };

      var ssp = this.worldToScreen(sAnim.x + driftX, sAnim.y + driftY);
      var tsp = this.worldToScreen(tAnim.x + driftX, tAnim.y + driftY);

      var hasFocus = highlightNodes.size > 0 || highlightEdges.size > 0;
      var isActive = !hasFocus || (highlightNodes.has(edge.source) && highlightNodes.has(edge.target));
      var isMemHl = highlightEdges.has(edge.id);
      var isMuted = hasFocus && !isActive && !isMemHl;

      var de = {
        id: edge.id, sx: ssp.x, sy: ssp.y, tx: tsp.x, ty: tsp.y,
        sourceId: edge.source, targetId: edge.target,
        relationType: edge.relation_type || "related",
        memoryId: edge.memory_id, weight: edge.weight || 1,
        confidence: edge.confidence || 0.8,
        isActive: isActive, isHighlighted: isMemHl,
        isMuted: isMuted, hasFocus: hasFocus,
        isHovered: edge.id === hoverId,
        color: edge.__color || TYPE_COLORS.other,
      };
      this._drawnEdges.push(de);

      if (de.isMuted) continue;
      this._drawEdge(ctx, de, dark);
    }
    ctx.restore();

    /* 流光脉冲：沿连线游走的光点 */
    this._updatePulses(now);
    this._drawPulses(ctx, now, dark);

    /* Draw nodes */
    this._drawnNodes = [];
    ctx.save();
    for (var i = 0; i < nodes.length; i++) {
      var nd = nodes[i];
      /* Animated position */
      var px = lerp(nd._prevX || nd.x, nd.x, ap);
      var py = lerp(nd._prevY || nd.y, nd.y, ap);
      var sp = this.worldToScreen(px + driftX, py + driftY);

      var isCenter = centerId != null && nd.id === centerId;
      var isSel = nd.id === selNodeId;
      var isHl = highlightNodes.has(nd.id);
      var hasNodeFocus = highlightNodes.size > 0 || highlightEdges.size > 0;
      var isMuted = hasNodeFocus && !isHl && !isSel;
      var sr = this.nodeScreenRadius(nd, isCenter);

      var drawInfo = {
        id: nd.id, sx: sp.x, sy: sp.y, sr: sr,
        isSelected: isSel, isHighlighted: isHl, isMuted: isMuted,
        isHovered: nd.id === hoverId, isCenter: isCenter, hasFocus: hasNodeFocus,
        type: nd.type || "other", label: nd.label || "Unnamed",
        memoryCount: nd.memory_count || 0, degree: nd.degree || 0,
        labelScore: nd.labelScore || 0,
        color: TYPE_COLORS[nd.type] || TYPE_COLORS.other, fixed: nd.fixed,
      };
      this._drawnNodes.push(drawInfo);

      if (drawInfo.isMuted && !drawInfo.isHovered) {
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        ctx.arc(drawInfo.sx, drawInfo.sy, Math.max(1.4, drawInfo.sr * 0.6), 0, Math.PI * 2);
        ctx.fillStyle = dark ? "#5c6370" : "#c7ccd4";
        ctx.fill();
        ctx.globalAlpha = 1;
        continue;
      }
      this._drawNode(ctx, drawInfo, scale, dark, now);
    }
    ctx.restore();

    /* 涟漪 + 记忆碎片 */
    this._drawRipples(ctx, now, dark);
    this._updateFragments(now);
    this._drawFragments(ctx, now, dark);
  };

  /* Draw a single edge as a straight link */
  Renderer.prototype._drawEdge = function(ctx, de, dark) {
    var opacity = de.isHighlighted ? CFG.EDGE_OPACITY_HIGHLIGHT
      : de.hasFocus && de.isActive ? CFG.EDGE_OPACITY_ACTIVE : CFG.EDGE_OPACITY_DEFAULT;
    var width = de.isHighlighted ? CFG.EDGE_WIDTH_HIGHLIGHT
      : de.hasFocus && de.isActive ? CFG.EDGE_WIDTH_ACTIVE : CFG.EDGE_WIDTH_DEFAULT;
    var strength = clamp(Math.sqrt(Number(de.weight || 1)) / 3.6, 0, 1);

    if (de.isMuted) opacity *= 0.35;
    if (!de.isMuted) {
      width += strength * (de.hasFocus ? 0.35 : 0.8);
      opacity = clamp(opacity + strength * (de.hasFocus ? 0.04 : 0.1), 0, 0.84);
    }

    ctx.beginPath();
    ctx.moveTo(de.sx, de.sy);
    ctx.lineTo(de.tx, de.ty);
    ctx.strokeStyle = de.isHighlighted || (de.hasFocus && de.isActive)
      ? hexToRgba(de.color, opacity)
      : dark ? "rgba(168,180,225," + opacity + ")" : "rgba(96,112,150," + opacity + ")";
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.stroke();
  };

  /* ── 流光脉冲：光点沿连线游走 ── */
  Renderer.prototype._updatePulses = function(now) {
    var interval = CFG.PULSE_SPAWN_MIN + Math.random() * (CFG.PULSE_SPAWN_MAX - CFG.PULSE_SPAWN_MIN);
    if (now - this._lastPulseSpawn > interval) {
      this._lastPulseSpawn = now;
      if (this.pulses.length < CFG.PULSE_MAX && this._drawnEdges.length) {
        var candidates = this._drawnEdges.filter(function(de) { return !de.isMuted; });
        if (candidates.length) {
          var de = candidates[(Math.random() * candidates.length) | 0];
          this.pulses.push({
            edgeId: de.id,
            t0: now,
            dur: CFG.PULSE_DUR_MIN + Math.random() * (CFG.PULSE_DUR_MAX - CFG.PULSE_DUR_MIN),
            fwd: Math.random() < 0.5,
            color: de.color,
            hl: de.isHighlighted,
          });
        }
      }
    }
    if (this.pulses.length) {
      var self = this;
      this.pulses = this.pulses.filter(function(p) {
        return (now - p.t0) < p.dur && self._edgeById(p.edgeId);
      });
    }
  };

  Renderer.prototype._edgeById = function(edgeId) {
    for (var i = 0; i < this._drawnEdges.length; i++) {
      if (this._drawnEdges[i].id === edgeId) return this._drawnEdges[i];
    }
    return null;
  };

  Renderer.prototype._drawPulses = function(ctx, now, dark) {
    if (!this.pulses.length) return;
    ctx.save();
    if (dark) ctx.globalCompositeOperation = "lighter";
    for (var i = 0; i < this.pulses.length; i++) {
      var p = this.pulses[i];
      var de = this._edgeById(p.edgeId);
      if (!de) continue;

      var t = clamp((now - p.t0) / p.dur, 0, 1);
      t = t * t * (3 - 2 * t); /* smoothstep */
      var head = p.fwd ? t : 1 - t;
      var tail = p.fwd
        ? clamp(head - CFG.PULSE_TRAIL, 0, 1)
        : clamp(head + CFG.PULSE_TRAIL, 0, 1);

      var hx = lerp(de.sx, de.tx, head);
      var hy = lerp(de.sy, de.ty, head);
      var tx2 = lerp(de.sx, de.tx, tail);
      var ty2 = lerp(de.sy, de.ty, tail);

      /* 光尾 */
      var grad = ctx.createLinearGradient(tx2, ty2, hx, hy);
      grad.addColorStop(0, hexToRgba(p.color, 0));
      grad.addColorStop(1, hexToRgba(p.color, p.hl ? 0.75 : 0.45));
      ctx.beginPath();
      ctx.moveTo(tx2, ty2);
      ctx.lineTo(hx, hy);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.4;
      ctx.lineCap = "round";
      ctx.stroke();

      /* 亮点头 */
      ctx.beginPath();
      ctx.arc(hx, hy, p.hl ? 2.1 : 1.6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255," + (p.hl ? 0.95 : 0.8) + ")";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(hx, hy, p.hl ? 4.6 : 3.6, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(p.color, 0.22);
      ctx.fill();
    }
    ctx.restore();
  };

  /* ── 涟漪：点击/聚焦时扩散环 ── */
  Renderer.prototype.spawnRipple = function(nodeId) {
    for (var i = 0; i < this._drawnNodes.length; i++) {
      var dn = this._drawnNodes[i];
      if (dn.id === nodeId) {
        this.ripples.push({ x: dn.sx, y: dn.sy, t0: Date.now() / 1000 });
        if (this.ripples.length > CFG.RIPPLE_MAX) this.ripples.shift();
        return;
      }
    }
  };

  Renderer.prototype._drawRipples = function(ctx, now, dark) {
    if (!this.ripples.length) return;
    var self = this;
    this.ripples = this.ripples.filter(function(r) { return (now - r.t0) < CFG.RIPPLE_LIFE; });
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (var i = 0; i < this.ripples.length; i++) {
      var r = this.ripples[i];
      var p = clamp((now - r.t0) / CFG.RIPPLE_LIFE, 0, 1);
      var ease = 1 - Math.pow(1 - p, 2);
      var fade = 1 - p;
      var rad = 10 + ease * 62;

      ctx.beginPath();
      ctx.arc(r.x, r.y, rad, 0, Math.PI * 2);
      ctx.strokeStyle = dark ? "rgba(160,190,255," + (fade * 0.5).toFixed(3) + ")" : "rgba(111,157,240," + (fade * 0.45).toFixed(3) + ")";
      ctx.lineWidth = 1.3;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(r.x, r.y, rad * 0.62, 0, Math.PI * 2);
      ctx.strokeStyle = dark ? "rgba(160,190,255," + (fade * 0.3).toFixed(3) + ")" : "rgba(111,157,240," + (fade * 0.26).toFixed(3) + ")";
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }
    ctx.restore();
  };

  /* ── 记忆碎片层：①星点旁浮现（带引导线） ②横穿星域的飘过碎片 ── */
  Renderer.prototype.setFragmentPool = function(pool) {
    this.fragPool = Array.isArray(pool)
      ? pool.filter(function(s) { return s && String(s).trim(); }).map(String)
      : [];
  };

  Renderer.prototype.setFragmentsEnabled = function(on) {
    this.fragmentsEnabled = !!on;
    if (!on) {
      this.nearFrags = [];
      this.driftFrags = [];
    }
  };

  Renderer.prototype._pickFragText = function() {
    if (!this.fragPool.length) return null;
    var t = this.fragPool[(Math.random() * this.fragPool.length) | 0];
    t = String(t).replace(/\s+/g, " ").trim();
    if (t.length > CFG.FRAG_MAX_CHARS) t = t.substring(0, CFG.FRAG_MAX_CHARS) + "…";
    return t;
  };

  Renderer.prototype._updateFragments = function(now) {
    if (!this.fragmentsEnabled || !this.fragPool.length) {
      if (this.nearFrags.length) this.nearFrags = [];
      if (this.driftFrags.length) this.driftFrags = [];
      return;
    }

    /* ① 星点旁浮现碎片 */
    var spawnIn = CFG.FRAG_NEAR_SPAWN_MIN + Math.random() * (CFG.FRAG_NEAR_SPAWN_MAX - CFG.FRAG_NEAR_SPAWN_MIN);
    if (now - this._fragSpawnAt > spawnIn && this.nearFrags.length < CFG.FRAG_NEAR_MAX) {
      this._fragSpawnAt = now;
      var cands = this._drawnNodes.filter(function(n) {
        return !n.isMuted && n.sx > 60 && n.sx < this.width - 180 && n.sy > 60 && n.sy < this.height - 70;
      }, this);
      if (cands.length) {
        var n = cands[(Math.random() * cands.length) | 0];
        var text = this._pickFragText();
        if (text) {
          var ang = -Math.PI / 2 + (Math.random() - 0.5) * 1.7;
          var dist = n.sr + 16 + Math.random() * 22;
          this.nearFrags.push({
            nx: n.id,
            text: text,  /* 必须带上，否则画成 "undefined" */
            x: n.sx + Math.cos(ang) * dist,
            y: n.sy + Math.sin(ang) * dist,
            vx: (Math.random() - 0.5) * 2.4,
            vy: -CFG.FRAG_RISE * (0.6 + Math.random() * 0.5),
            t0: now,
            dur: CFG.FRAG_FADE + CFG.FRAG_HOLD + CFG.FRAG_FADE,
          });
        }
      }
    }
    var life = CFG.FRAG_FADE + CFG.FRAG_HOLD + CFG.FRAG_FADE;
    this.nearFrags = this.nearFrags.filter(function(f) { return (now - f.t0) < life; });

    /* ② 飘过碎片 */
    var driftIn = CFG.FRAG_DRIFT_SPAWN_MIN + Math.random() * (CFG.FRAG_DRIFT_SPAWN_MAX - CFG.FRAG_DRIFT_SPAWN_MIN);
    if (now - this._driftSpawnAt > driftIn && this.driftFrags.length < CFG.FRAG_DRIFT_MAX) {
      this._driftSpawnAt = now;
      var text2 = this._pickFragText();
      if (text2) {
        var fromLeft = Math.random() < 0.5;
        var speed = (this.width + 380) / (12 + Math.random() * 4);
        this.driftFrags.push({
          x: fromLeft ? -200 : this.width + 200,
          y: this.height * (0.08 + Math.random() * 0.38),
          text: text2,  /* 必须带上，否则画成 "undefined" */
          vx: (fromLeft ? 1 : -1) * speed,
          t0: now,
        });
      }
    }
    this.driftFrags = this.driftFrags.filter(function(f) {
      return f.x > -260 && f.x < this.width + 260;
    }, this);
  };

  Renderer.prototype._drawFragments = function(ctx, now, dark) {
    if (!this.fragmentsEnabled) return;
    var fragFade = CFG.FRAG_FADE;
    ctx.save();

    /* ① 星点旁浮现：细引导线 + 短文字 */
    for (var i = 0; i < this.nearFrags.length; i++) {
      var f = this.nearFrags[i];
      var age = now - f.t0;
      var alpha = age < fragFade
        ? age / fragFade
        : age > f.dur - fragFade ? Math.max(0, (f.dur - age) / fragFade) : 1;
      if (alpha <= 0.01) continue;

      var node = null;
      for (var j = 0; j < this._drawnNodes.length; j++) {
        if (this._drawnNodes[j].id === f.nx) { node = this._drawnNodes[j]; break; }
      }
      if (!node || node.isMuted) continue;

      var fx = f.x + f.vx * age;
      var fy = f.y + f.vy * age;

      /* 引导线：从星点边缘指向文字 */
      var dx = fx - node.sx;
      var dy = fy - node.sy;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      var sx2 = node.sx + (dx / len) * (node.sr * 1.5);
      var sy2 = node.sy + (dy / len) * (node.sr * 1.5);
      var ex2 = fx - (dx / len) * 4;
      var ey2 = fy - (dy / len) * 4;
      ctx.beginPath();
      ctx.moveTo(sx2, sy2);
      ctx.lineTo(ex2, ey2);
      ctx.strokeStyle = dark
        ? "rgba(190,205,245," + (alpha * 0.34).toFixed(3) + ")"
        : "rgba(90,110,170," + (alpha * 0.4).toFixed(3) + ")";
      ctx.lineWidth = 0.7;
      ctx.stroke();

      ctx.font = "11.5px -apple-system, BlinkMacSystemFont, 'PingFang SC', sans-serif";
      var textW = ctx.measureText(f.text).width;

      /* 文字胶囊底：让短句在任何背景下都可读、更醒目 */
      var padX = 8, padY = 4.5, capH = 20, capR = 10;
      var capX = fx >= node.sx ? fx - padX : fx - textW - padX;
      ctx.beginPath();
      ctx.moveTo(capX + capR, fy - capH / 2);
      ctx.lineTo(capX + textW + padX * 2 - capR, fy - capH / 2);
      ctx.arcTo(capX + textW + padX * 2, fy - capH / 2, capX + textW + padX * 2, fy - capH / 2 + capR, capR);
      ctx.lineTo(capX + textW + padX * 2, fy + capH / 2 - capR);
      ctx.arcTo(capX + textW + padX * 2, fy + capH / 2, capX + textW + padX * 2 - capR, fy + capH / 2, capR);
      ctx.lineTo(capX + capR, fy + capH / 2);
      ctx.arcTo(capX, fy + capH / 2, capX, fy + capH / 2 - capR, capR);
      ctx.lineTo(capX, fy - capH / 2 + capR);
      ctx.arcTo(capX, fy - capH / 2, capX + capR, fy - capH / 2, capR);
      ctx.closePath();
      ctx.fillStyle = dark
        ? "rgba(24,33,62," + (alpha * 0.62).toFixed(3) + ")"
        : "rgba(252,253,255," + (alpha * 0.68).toFixed(3) + ")";
      ctx.fill();
      ctx.strokeStyle = dark
        ? "rgba(140,160,220," + (alpha * 0.3).toFixed(3) + ")"
        : "rgba(120,140,200," + (alpha * 0.32).toFixed(3) + ")";
      ctx.lineWidth = 0.8;
      ctx.stroke();

      ctx.textAlign = fx >= node.sx ? "left" : "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = dark
        ? "rgba(232,239,255," + (alpha * 0.95).toFixed(3) + ")"
        : "rgba(44,56,92," + (alpha * 0.9).toFixed(3) + ")";
      ctx.fillText(f.text, fx, fy);
    }

    /* ② 飘过碎片：斜体、低透明度、缓慢横穿星域 */
    ctx.font = "italic 12px -apple-system, BlinkMacSystemFont, 'PingFang SC', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (var k = 0; k < this.driftFrags.length; k++) {
      var d = this.driftFrags[k];
      var dAge = now - d.t0;
      var edgeFade = Math.min(1, Math.min(d.x, this.width - d.x) / CFG.FRAG_DRIFT_EDGE_FADE);
      if (edgeFade <= 0) continue;
      var dAlpha = Math.min(1, dAge / 1.6) * edgeFade * 0.55;
      if (dAlpha <= 0.01) continue;
      ctx.fillStyle = dark
        ? "rgba(214,224,250," + dAlpha.toFixed(3) + ")"
        : "rgba(60,74,116," + (dAlpha * 1.15).toFixed(3) + ")";
      ctx.fillText(d.text, d.x, d.y);
      d.x += d.vx * 0.016; /* 帧步进近似 */
    }

    ctx.restore();
  };

  /* Draw a single circular node */
  Renderer.prototype._drawNode = function(ctx, dn, scale, dark, now) {
    var x = dn.sx, y = dn.sy, r = dn.sr;
    var col = dn.color;

    ctx.save();
    /* 夜间用叠加发光，白天普通合成（additive 会发白） */
    if (dark) ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = dn.isMuted ? 0.26 : 1;

    /* 呼吸（幅度减半）：光晕 ±4% */
    var breath = 1 + Math.sin(now * 0.9 + (dn.id % 17) * 0.37) * 0.04;

    /* ① 透明渐变光晕（收紧） */
    var haloBoost = (dn.isSelected ? 8 : dn.isHovered ? 6 : dn.isCenter ? 4 : 0) * scale;
    var haloR = r * 2.4 * breath + haloBoost;
    var grad = ctx.createRadialGradient(x, y, r * 0.2, x, y, haloR);
    grad.addColorStop(0, hexToRgba(col, dark ? 0.34 : 0.30));
    grad.addColorStop(0.45, hexToRgba(col, dark ? 0.17 : 0.14));
    grad.addColorStop(1, hexToRgba(col, 0));
    ctx.beginPath();
    ctx.arc(x, y, haloR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    /* ② 类型色冕 */
    ctx.beginPath();
    ctx.arc(x, y, r * 0.92, 0, Math.PI * 2);
    ctx.fillStyle = hexToRgba(col, dark ? 0.85 : 0.92);
    ctx.fill();

    /* ③ 白热内核 */
    ctx.beginPath();
    ctx.arc(x, y, r * 0.45, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255," + (dark ? 0.96 : 0.9) + ")";
    ctx.fill();
    if (!dark) {
      ctx.beginPath();
      ctx.arc(x, y, r * 0.92, 0, Math.PI * 2);
      ctx.lineWidth = 1;
      ctx.strokeStyle = hexToRgba(col, 0.8);
      ctx.stroke();
    }

    /* 选中/悬停外环 */
    if (dn.isSelected || dn.isHovered) {
      ctx.beginPath();
      ctx.arc(x, y, r + 3.5 * scale, 0, Math.PI * 2);
      ctx.lineWidth = dn.isSelected ? 1.6 : 1.1;
      ctx.strokeStyle = hexToRgba(col, dn.isSelected ? 0.9 : 0.6);
      ctx.stroke();
    }

    /* 大节点：倾斜细环（缓慢旋转）+ 十字星芒 */
    var grand = dn.isCenter || dn.isSelected || dn.degree >= 6 || dn.memoryCount >= 6;
    if (grand && !dn.isMuted) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(now * 0.15 + (dn.id % 13));
      ctx.beginPath();
      ctx.ellipse(0, 0, r * 1.7, r * 0.55, 0, 0, Math.PI * 2);
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = hexToRgba(col, dark ? 0.32 : 0.4);
      ctx.stroke();
      ctx.restore();

      var sr2 = r * (1.9 + Math.sin(now * 0.7 + dn.id) * 0.08);
      ctx.save();
      if (dark) ctx.globalCompositeOperation = "lighter";
      ctx.lineWidth = 0.9;
      ctx.strokeStyle = "rgba(255,255,255," + (dark ? 0.5 : 0.35) + ")";
      ctx.beginPath();
      ctx.moveTo(x - sr2, y);
      ctx.lineTo(x + sr2, y);
      ctx.moveTo(x, y - sr2);
      ctx.lineTo(x, y + sr2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();

    /* 标签：只在悬停/选中/焦点节点显示，平时保持纯净星空 */
    var labelVisible = dn.isHovered || dn.isSelected || dn.isCenter;
    if (!labelVisible || dn.isMuted) {
      return;
    }

    ctx.save();
    var fontSize = Math.max(10, CFG.NODE_FONT_SIZE * scale);
    ctx.fillStyle = dark ? "#e9ecef" : "#2f343a";
    ctx.font = (dn.isSelected || dn.isCenter ? "600 " : "500 ") + fontSize + "px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    var maxChars = dn.isCenter ? 28 : 24;
    var label = dn.label.length > maxChars ? dn.label.substring(0, maxChars - 1) + "…" : dn.label;
    var labelX = x + r + 7 * scale;
    var labelWidth = ctx.measureText(label).width;
    var labelHeight = fontSize + 4;
    var box = {
      x1: labelX - 3 * scale,
      y1: y - labelHeight / 2 - 2,
      x2: labelX + labelWidth + 3 * scale,
      y2: y + labelHeight / 2 + 2,
    };
    var forceLabel = dn.isHovered || dn.isSelected || dn.isCenter;
    if (!forceLabel && (!this._labelInView(box) || this._labelIntersects(box))) {
      ctx.restore();
      return;
    }
    this._labelBoxes.push(box);
    ctx.fillText(label, labelX, y);

    if (dn.isHovered || dn.isSelected) {
      var metaFs = Math.max(8, CFG.NODE_META_SIZE * scale);
      ctx.fillStyle = dark ? "#a6abb4" : "#6b7280";
      ctx.font = metaFs + "px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textBaseline = "top";
      ctx.fillText(dn.memoryCount + "M / " + dn.degree + " links", labelX, y + 8 * scale);
    }

    ctx.restore();
  };

  Renderer.prototype._labelInView = function(box) {
    return box.x2 >= 0 && box.x1 <= this.width && box.y2 >= 0 && box.y1 <= this.height;
  };

  Renderer.prototype._labelIntersects = function(box) {
    for (var i = 0; i < this._labelBoxes.length; i++) {
      var other = this._labelBoxes[i];
      if (box.x1 <= other.x2 && box.x2 >= other.x1 && box.y1 <= other.y2 && box.y2 >= other.y1) {
        return true;
      }
    }
    return false;
  };

  Renderer.prototype.hitTestNode = function(sx, sy) {
    var best = null, bestDist = Infinity;
    for (var i = this._drawnNodes.length - 1; i >= 0; i--) {
      var dn = this._drawnNodes[i];
      if (dn.isMuted) continue;
      var d = Math.sqrt((sx - dn.sx) ** 2 + (sy - dn.sy) ** 2);
      if (d < dn.sr + CFG.HOVER_RADIUS && d < bestDist) { best = dn; bestDist = d; }
    }
    return best;
  };

  Renderer.prototype.hitTestEdge = function(sx, sy) {
    for (var i = 0; i < this._drawnEdges.length; i++) {
      var de = this._drawnEdges[i];
      if (de.isMuted) continue;
      var dist = pointToSegmentDistance(sx, sy, de.sx, de.sy, de.tx, de.ty);
      if (dist < 8) return de;
    }
    return null;
  };

  function pointToSegmentDistance(px, py, x1, y1, x2, y2) {
    var dx = x2 - x1;
    var dy = y2 - y1;
    var len2 = dx * dx + dy * dy;
    if (!len2) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    var t = clamp(((px - x1) * dx + (py - y1) * dy) / len2, 0, 1);
    var x = x1 + t * dx;
    var y = y1 + t * dy;
    return Math.sqrt((px - x) ** 2 + (py - y) ** 2);
  }

  /* ═══════════════════════════════════════════════════════════════
     Interaction — mouse / touch
     ═══════════════════════════════════════════════════════════════ */
  function Interaction(container, canvas, renderer, callbacks) {
    this.container = container;
    this.canvas = canvas;
    this.renderer = renderer;
    this.cb = callbacks || {};
    this._dragging = false;
    this._panning = false;
    this._dragNode = null;
    this._dragStart = { x: 0, y: 0 };
    this._panStart = { ox: 0, oy: 0, mx: 0, my: 0 };
    this._hoverId = null;
    this._hoverType = null;
    this._pinchDist = 0;
    this._pinchScale = 1;
    this._bind();
  }

  Interaction.prototype._bind = function() {
    var self = this;
    var el = this.canvas;
    el.addEventListener("mousedown", function(e) { self._onMouseDown(e); });
    el.addEventListener("mousemove", function(e) { self._onMouseMove(e); });
    window.addEventListener("mouseup", function(e) { self._onMouseUp(e); });
    el.addEventListener("mouseleave", function(e) { self._onMouseUp(e); });
    el.addEventListener("wheel", function(e) { self._onWheel(e); }, { passive: false });
    el.addEventListener("dblclick", function(e) { self._onDblClick(e); });
    el.addEventListener("touchstart", function(e) { self._onTouchStart(e); }, { passive: false });
    el.addEventListener("touchmove", function(e) { self._onTouchMove(e); }, { passive: false });
    el.addEventListener("touchend", function(e) { self._onTouchEnd(e); });
    el.addEventListener("contextmenu", function(e) { e.preventDefault(); });
  };

  Interaction.prototype._onMouseDown = function(e) {
    var pos = getPos(e, this.canvas);
    var hit = this.renderer.hitTestNode(pos.x, pos.y);
    if (hit && e.button === 0) {
      this._dragging = true;
      this._dragNode = hit;
      this._dragStart = { x: pos.x, y: pos.y };
      e.preventDefault();
      return;
    }
    if (e.button === 0 || e.button === 2) {
      this._panning = true;
      this._panStart = {
        ox: this.renderer.viewport.ox, oy: this.renderer.viewport.oy,
        mx: pos.x, my: pos.y,
      };
      e.preventDefault();
    }
  };

  Interaction.prototype._onMouseMove = function(e) {
    var pos = getPos(e, this.canvas);
    var vr = this.renderer.viewport;

    if (this._dragging && this._dragNode) {
      var world = this.renderer.screenToWorld(pos.x, pos.y);
      var simNode = this.renderer._nodesMap && this.renderer._nodesMap[this._dragNode.id];
      if (simNode) {
        simNode.x = simNode._prevX = world.x;
        simNode.y = simNode._prevY = world.y;
        simNode.fixed = true;
      }
      return;
    }

    if (this._panning) {
      vr.ox = this._panStart.ox + (pos.x - this._panStart.mx) / vr.scale;
      vr.oy = this._panStart.oy + (pos.y - this._panStart.my) / vr.scale;
      return;
    }

    var hit = this.renderer.hitTestNode(pos.x, pos.y);
    if (hit) {
      if (this._hoverId !== hit.id || this._hoverType !== "node") {
        this._hoverId = hit.id; this._hoverType = "node";
        if (this.cb.onNodeHover) this.cb.onNodeHover(hit.id);
      }
      this.canvas.style.cursor = "pointer";
      return;
    }

    var hitE = this.renderer.hitTestEdge(pos.x, pos.y);
    if (hitE) {
      this._hoverId = hitE.id; this._hoverType = "edge";
      this.canvas.style.cursor = "pointer";
      return;
    }

    if (this._hoverId !== null) {
      this._hoverId = null; this._hoverType = null;
      if (this.cb.onNodeHover) this.cb.onNodeHover(null);
    }
    this.canvas.style.cursor = this._panning ? "grabbing" : "grab";
  };

  Interaction.prototype._onMouseUp = function(e) {
    if (this._dragging && this._dragNode) {
      var pos = getPos(e, this.canvas);
      var dx = pos.x - this._dragStart.x, dy = pos.y - this._dragStart.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        if (this.cb.onNodeClick) this.cb.onNodeClick(this._dragNode.id);
      }
      this._dragging = false; this._dragNode = null;
    }
    if (this._panning) {
      var pos2 = getPos(e, this.canvas);
      if (Math.sqrt((pos2.x - this._panStart.mx) ** 2 + (pos2.y - this._panStart.my) ** 2) < 3) {
        if (this.cb.onBackgroundClick) this.cb.onBackgroundClick();
      }
      this._panning = false;
    }
    this.canvas.style.cursor = "grab";
  };

  Interaction.prototype._onWheel = function(e) {
    e.preventDefault();
    var vr = this.renderer.viewport;
    var delta = e.deltaY > 0 ? -CFG.ZOOM_STEP * 60 : CFG.ZOOM_STEP * 60;
    var newScale = clamp(vr.scale + delta, CFG.ZOOM_MIN, CFG.ZOOM_MAX);
    var pos = getPos(e, this.canvas);
    var before = this.renderer.screenToWorld(pos.x, pos.y);
    vr.scale = newScale;
    var after = this.renderer.screenToWorld(pos.x, pos.y);
    vr.ox += before.x - after.x;
    vr.oy += before.y - after.y;
  };

  Interaction.prototype._onDblClick = function(e) {
    var pos = getPos(e, this.canvas);
    var hit = this.renderer.hitTestNode(pos.x, pos.y);
    if (hit && this.cb.onNodeDblClick) this.cb.onNodeDblClick(hit.id);
  };

  Interaction.prototype._onTouchStart = function(e) {
    if (e.touches.length === 2) {
      var t0 = e.touches[0], t1 = e.touches[1];
      this._pinchDist = Math.sqrt((t1.clientX - t0.clientX) ** 2 + (t1.clientY - t0.clientY) ** 2);
      this._pinchScale = this.renderer.viewport.scale;
      return;
    }
    if (e.touches.length === 1) {
      this._onMouseDown({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY, button: 0 });
    }
    e.preventDefault();
  };

  Interaction.prototype._onTouchMove = function(e) {
    if (e.touches.length === 2 && this._pinchDist > 0) {
      var t0 = e.touches[0], t1 = e.touches[1];
      var d = Math.sqrt((t1.clientX - t0.clientX) ** 2 + (t1.clientY - t0.clientY) ** 2);
      this.renderer.viewport.scale = clamp(this._pinchScale * (d / this._pinchDist), CFG.ZOOM_MIN, CFG.ZOOM_MAX);
      return;
    }
    if (e.touches.length === 1) {
      this._onMouseMove({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
    }
    e.preventDefault();
  };

  Interaction.prototype._onTouchEnd = function(e) {
    if (e.touches.length < 2) this._pinchDist = 0;
    var t = e.changedTouches[0] || {};
    this._onMouseUp({ clientX: t.clientX || 0, clientY: t.clientY || 0 });
  };

  Interaction.prototype.getHoverId = function() { return this._hoverId; };
  Interaction.prototype.getHoverType = function() { return this._hoverType; };

  /* ═══════════════════════════════════════════════════════════════
     Animator — RAF loop with layout position tweening
     ═══════════════════════════════════════════════════════════════ */
  function Animator(renderer, interaction) {
    this.renderer = renderer;
    this.interaction = interaction;
    this._running = false;
    this._rafId = null;
    this._nodes = [];
    this._edges = [];
    this._nodeMap = {};
    this._mem2node = {};
    this._layout = new ForceDirectedLayout();
    this._animProgress = 1; // 0→1 for position transitions
    this._needsRender = true;
  }

  Animator.prototype.fitViewport = function(options) {
    options = options || {};
    if (!this._nodes.length || !this.renderer.width || !this.renderer.height) return;

    var centerId = options.centerId != null ? options.centerId : this._layout.centerId;
    var centerTarget = centerId != null ? this._layout.getTarget(centerId) : null;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (var i = 0; i < this._nodes.length; i++) {
      var nd = this._nodes[i];
      var target = this._layout.getTarget(nd.id);
      var isCenter = this._layout.centerId != null && nd.id === this._layout.centerId;
      var pad = this.renderer.nodeWorldRadius(nd, isCenter) + 28;
      minX = Math.min(minX, target.tx - pad);
      maxX = Math.max(maxX, target.tx + pad);
      minY = Math.min(minY, target.ty - pad);
      maxY = Math.max(maxY, target.ty + pad);
    }

    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return;

    var boundsW = Math.max(1, maxX - minX);
    var boundsH = Math.max(1, maxY - minY);
    var padding = this.renderer.width < 520 ? 0.86 : 0.74;
    var fitScale = Math.min(
      (this.renderer.width * padding) / boundsW,
      (this.renderer.height * padding) / boundsH
    );
    /* 上限放宽到 2.8：节点少时星群也铺满画布，而不是缩在中间一小坨 */
    var scale = clamp(fitScale, 0.35, 2.8);
    var cx = (minX + maxX) / 2;
    var cy = (minY + maxY) / 2;

    if (centerTarget) {
      var centerBias = this.renderer.width < 520 ? 0.36 : 0.62;
      cx = lerp(cx, centerTarget.tx, centerBias);
      cy = lerp(cy, centerTarget.ty, centerBias * 0.78);
    }

    this.renderer.viewport.scale = scale;
    this.renderer.viewport.ox = -cx;
    this.renderer.viewport.oy = -cy;
    this._needsRender = true;
  };

  Animator.prototype.start = function() {
    if (this._running) return;
    this._running = true;
    this._tick();
  };

  Animator.prototype.stop = function() {
    this._running = false;
    if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
  };

  Animator.prototype.setData = function(nodes, edges) {
    var self = this;
    this._nodes = nodes;
    this._edges = edges;
    this._nodeMap = {};
    nodes.forEach(function(n) { self._nodeMap[n.id] = n; });
    this.renderer._nodesMap = this._nodeMap;
  };

  Animator.prototype.layoutGraph = function(centerId) {
    var self = this;
    /* Save previous positions for animation */
    this._nodes.forEach(function(n) {
      n._prevX = n.x;
      n._prevY = n.y;
    });
    this._layout.compute(this._nodes, this._edges, centerId);
    this.fitViewport({ centerId: centerId });
    this._animProgress = 0;
    this._needsRender = true;
    this.start();
  };

  Animator.prototype.recenter = function(centerId) {
    this.layoutGraph(centerId);
  };

  Animator.prototype._tick = function() {
    if (!this._running) return;
    var self = this;
    this._rafId = requestAnimationFrame(function() { self._tick(); });

    /* Animate positions toward layout targets */
    var dirty = true;
    if (this._animProgress < 1) {
      this._animProgress = Math.min(1, this._animProgress + CFG.ANIM_SPEED);
      var ap = easeInOutCubic(this._animProgress);

      for (var i = 0; i < this._nodes.length; i++) {
        var nd = this._nodes[i];
        if (nd.fixed) continue;
        var target = this._layout.getTarget(nd.id);
        if (nd._prevX == null) { nd._prevX = nd.x; nd._prevY = nd.y; }
        nd.x = lerp(nd._prevX, target.tx, ap);
        nd.y = lerp(nd._prevY, target.ty, ap);
      }
      if (this._animProgress >= 1) {
        /* Lock to exact targets */
        for (var j = 0; j < this._nodes.length; j++) {
          var nd2 = this._nodes[j];
          if (nd2.fixed) continue;
          var tgt = this._layout.getTarget(nd2.id);
          nd2.x = tgt.tx; nd2.y = tgt.ty;
          nd2._prevX = null; nd2._prevY = null;
        }
      }
    } else {
      var now = Date.now() / 1000;
      for (var k = 0; k < this._nodes.length; k++) {
        var floatNode = this._nodes[k];
        if (floatNode.fixed) continue;
        var home = this._layout.getTarget(floatNode.id);
        var ring = this._layout.getRing(floatNode.id);
        var weight = clamp(Number(floatNode.weight || 0), 0, 20);
        var amp = ring === 0 ? 1.2 : 2.0 + Math.sqrt(weight) * 0.4;
        var phase = (floatNode.id % 17) * 0.37;
        floatNode.x = lerp(floatNode.x, home.tx + Math.sin(now * 0.65 + phase) * amp, CFG.IDLE_DAMPING);
        floatNode.y = lerp(floatNode.y, home.ty + Math.cos(now * 0.55 + phase) * amp, CFG.IDLE_DAMPING);
      }
    }

    if (dirty || this._needsRender) {
      this.renderer.clear();
      var sel = this.renderer._selection;
      var hoverId = this.interaction.getHoverId();
      this.renderer.render(this._nodes, this._edges, this._nodeMap, sel, hoverId, this._layout, this._animProgress);
      this._needsRender = false;
    }
  };

  Animator.prototype.wake = function() {
    if (!this._running) this.start();
    this._needsRender = true;
  };

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  /* ═══════════════════════════════════════════════════════════════
     Graph2D — Public API
     ═══════════════════════════════════════════════════════════════ */
  function Graph2D() {
    this.container = null;
    this.canvas = null;
    this.renderer = null;
    this.interaction = null;
    this.animator = null;
    this.selection = null;
    this.callbacks = {};
    this._initialized = false;
  }

  Graph2D.prototype.init = function(containerEl, callbacks) {
    if (this._initialized) return;
    var self = this;
    this.container = containerEl;
    this.callbacks = callbacks || {};

    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.display = "block";
    this.canvas.style.cursor = "grab";
    this.container.innerHTML = "";
    this.container.appendChild(this.canvas);

    this.renderer = new Renderer(this.canvas);
    this.renderer._selection = this.selection;

    this.interaction = new Interaction(this.container, this.canvas, this.renderer, {
      onNodeClick: function(nodeId) {
        self.selectNode(nodeId);
        if (self.callbacks.onNodeClick) self.callbacks.onNodeClick(nodeId);
      },
      onNodeDblClick: function(nodeId) {
        if (self.callbacks.onNodeDblClick) self.callbacks.onNodeDblClick(nodeId);
      },
      onNodeHover: function(nodeId) {
        if (self.callbacks.onNodeHover) self.callbacks.onNodeHover(nodeId);
      },
      onBackgroundClick: function() {
        self.clearSelection();
        if (self.callbacks.onBackgroundClick) self.callbacks.onBackgroundClick();
      },
    });

    this.animator = new Animator(this.renderer, this.interaction);
    this.renderer.resize();
    this.animator.start();

    /* Resize observer */
    if (typeof window.ResizeObserver === "function") {
      var ro = new ResizeObserver(function() {
        self.resize();
      });
      ro.observe(this.container);
    }
    window.addEventListener("resize", function() {
      self.resize();
    }, { passive: true });

    /* Theme observer */
    if (typeof window.MutationObserver === "function") {
      var mo = new MutationObserver(function() { self.animator.wake(); });
      mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    }

    this._initialized = true;
  };

  Graph2D.prototype.loadData = function(payload) {
    var snapshot = payload.snapshot || {};
    var rawNodes = snapshot.nodes || [];
    var rawEdges = snapshot.edges || [];

    /* Convert to internal format.
       注意：id 必须收敛为有限数字——非数字 id 经 Number() 会变 NaN，
       而 seenIds/edgeSeen 以对象为容器时所有 NaN 是同一个键，会把
       全部节点/边错误去重成 1 条（画布只剩一条线的根因之一）。 */
    var seenIds = {};
    var nodes = [];
    rawNodes.forEach(function(node, idx) {
      var id = Number(node.id);
      if (!Number.isFinite(id)) id = idx + 1;
      if (seenIds[id]) return;
      seenIds[id] = true;
      nodes.push({
        id: id, type: node.type || "other",
        label: node.label || node.canonical_value || "Node",
        canonicalValue: node.canonical_value || "",
        x: 0, y: 0, _prevX: null, _prevY: null, fixed: false,
        weight: Number(node.weight || 0),
        memory_count: Number(node.memory_count || 0),
        degree: Number(node.degree || 0),
        entry_count: Number(node.entry_count || 0),
        labelScore: Number(node.degree || 0) * 2 +
          Number(node.memory_count || 0) * 3 +
          Number(node.entry_count || 0) +
          Number(node.weight || 0),
        color: TYPE_COLORS[node.type] || TYPE_COLORS.other,
      });
    });

    var edges = [];
    var edgeSeen = {};
    rawEdges.forEach(function(edge) {
      var rawId = Number(edge.id);
      var eid = (edge.id != null && Number.isFinite(rawId))
        ? rawId
        : (edge.source + ":" + edge.target + ":" + edge.memory_id);
      if (edgeSeen[eid]) return;
      edgeSeen[eid] = true;
      edges.push({
        id: eid, source: Number(edge.source), target: Number(edge.target),
        relation_type: edge.relation_type || "related",
        memory_id: Number(edge.memory_id || 0),
        weight: Number(edge.weight || 1),
        confidence: Number(edge.confidence || 0.8),
        __color: relationColor(edge.relation_type),
      });
    });

    /* Build memory→node index */
    var mem2node = {};
    edges.forEach(function(edge) {
      if (!mem2node[edge.memory_id]) mem2node[edge.memory_id] = new Set();
      mem2node[edge.memory_id].add(edge.source);
      mem2node[edge.memory_id].add(edge.target);
    });

    this.animator.setData(nodes, edges);
    this._mem2node = mem2node;
    this.animator._mem2node = mem2node;
    this._nodes = nodes;
    this._edges = edges;

    /* Determine center: if there's a selection, use it; else pick highest weight node */
    var centerId = null;
    if (this.selection && this.selection.type === "node") {
      centerId = this.selection.id;
    } else if (this.selection && this.selection.type === "memory" && mem2node[this.selection.id]) {
      var mids = Array.from(mem2node[this.selection.id]);
      if (mids.length > 0) centerId = mids[0];
    }

    /* Apply centered force layout with animation */
    this.animator.layoutGraph(centerId);

    this.animator.wake();
  };

  Graph2D.prototype.selectNode = function(nodeId) {
    this.selection = { type: "node", id: nodeId };
    this.renderer._selection = this.selection;
    /* Recenter on selected node with smooth animation */
    this.animator.recenter(nodeId);
    if (this.renderer) this.renderer.spawnRipple(nodeId);
  };

  Graph2D.prototype.selectMemory = function(memoryId) {
    this.selection = { type: "memory", id: memoryId };
    this.renderer._selection = this.selection;
    if (this._mem2node && this._mem2node[memoryId]) {
      var nodes = Array.from(this._mem2node[memoryId]);
      if (nodes.length) {
        this.animator.recenter(nodes[0]);
        if (this.renderer) this.renderer.spawnRipple(nodes[0]);
      }
    }
    this.animator.wake();
  };

  /* ── 记忆碎片层控制（graph-ui 调用） ── */
  Graph2D.prototype.setFragmentPool = function(pool) {
    if (this.renderer) this.renderer.setFragmentPool(pool);
  };

  Graph2D.prototype.setFragmentsEnabled = function(on) {
    if (this.renderer) this.renderer.setFragmentsEnabled(on);
  };

  Graph2D.prototype.clearSelection = function() {
    this.selection = null;
    this.renderer._selection = null;
    /* Re-layout with highest-scoring node as center */
    this.animator.layoutGraph(null);
  };

  Graph2D.prototype.resize = function() {
    if (this.renderer) this.renderer.resize();
    if (this.animator) {
      var centerId = this.selection && this.selection.type === "node" ? this.selection.id : null;
      this.animator.fitViewport({ centerId: centerId });
      this.animator.wake();
    }
  };

  Graph2D.prototype.destroy = function() {
    if (this.animator) this.animator.stop();
    if (this.canvas && this.canvas.parentElement) {
      this.canvas.parentElement.removeChild(this.canvas);
    }
    this._initialized = false;
  };

  function relationColor(type) {
    /* 关系线配色：蓝灰族低饱和（星夜氛围，不抢节点的类型微着色） */
    var palette = ["#7f8cb8", "#7385ab", "#8b87a8", "#6f93a8", "#8a7f9e", "#7c96b5"];
    var h = String(type || "related").split("").reduce(function(a, c) { return a * 31 + c.charCodeAt(0); }, 7);
    return palette[Math.abs(h) % palette.length];
  }

  /* ═══════════════════════════════════════════════════════════════
     Export
     ═══════════════════════════════════════════════════════════════ */
  window.Graph2D = new Graph2D();
})();
