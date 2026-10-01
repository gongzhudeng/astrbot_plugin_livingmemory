/**
 * Timeline Page - 时间轴页面
 * 按天分组的记忆浏览视图（Scheme B），点击卡片打开居中阅读层
 */

import { normalizeImportance, esc, statusPill, typeLabel } from "./utils.js";

const PAGE_SIZE = 100;

export class TimelinePage {
  constructor(state, apiClient, peekPanel) {
    this.state = state;
    this.api = apiClient;
    this.peek = peekPanel;

    this.items = [];       // 归一化后的记忆（新→旧）
    this.page = 0;         // 已加载页数（从 1 开始）
    this.hasMore = false;
    this.loading = false;
    this._stats = null;
  }

  hasData() {
    return this.items.length > 0;
  }

  /** 数据失效：编辑/删除/新增后调用，下次进入页面重新拉取 */
  invalidate() {
    this.items = [];
    this.page = 0;
    this.hasMore = false;
    this._stats = null;
  }

  async fetch() {
    if (this.loading) return;
    if (this.items.length === 0) {
      // 首次进入（或数据被 invalidate）：从头加载第一页
      this.page = 0;
      this.loading = true;
      try {
        await this._fetchStats();
        await this._fetchPage();
      } catch (e) {
        this._showToast(e.message || window.t("timeline.loadFail"), true);
      } finally {
        this.loading = false;
      }
    } else {
      // 已有数据：仅重绘（语言切换/主题变化后保持内容）
      this.render();
    }
  }

  async _fetchStats() {
    if (this._stats) return;
    try {
      this._stats = await this.api.get("stats");
    } catch (e) {
      this._stats = {};
    }
  }

  async _fetchPage() {
    const nextPage = this.page + 1;
    const data = await this.api.get("memories", {
      page: String(nextPage),
      page_size: String(PAGE_SIZE),
      sort: "created_desc",
    });

    this.page = nextPage;
    this.hasMore = Boolean(data.has_more);

    const incoming = (Array.isArray(data.items) ? data.items : []).map(item => this._normalize(item));
    // 合并去重（按 memory_id）
    const seen = new Set(this.items.map(i => i.memory_id));
    incoming.forEach(item => {
      if (!seen.has(item.memory_id)) {
        this.items.push(item);
        seen.add(item.memory_id);
      }
    });

    this.render();
  }

  _normalize(item) {
    const createTime = item.metadata && item.metadata.create_time
      ? new Date(item.metadata.create_time * 1000)
      : null;
    return {
      memory_id: item.id,
      summary: item.text || item.content || "",
      content: item.text || item.content,
      memory_type: (item.metadata && item.metadata.memory_type) || "GENERAL",
      importance: normalizeImportance(item.metadata && item.metadata.importance),
      status: (item.metadata && item.metadata.status) || "active",
      created_at: createTime
        ? createTime.toLocaleString()
        : item.created_at || "--",
      _date: createTime,
      consolidated_count: (item.metadata && Array.isArray(item.metadata.consolidated_from))
        ? item.metadata.consolidated_from.length
        : 0,
      raw: item,
    };
  }

  /** 按天分组渲染 */
  render() {
    const container = document.getElementById("timeline-container");
    if (!container) return;

    this._renderStats();

    if (!this.items.length) {
      container.innerHTML = '<div class="timeline-empty">' + esc(window.t("timeline.empty")) + '</div>';
      this._updateMoreBtn();
      return;
    }

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const DAY_MS = 86400000;

    const groups = []; // [{key, date, items}]
    let current = null;
    for (const item of this.items) {
      const d = item._date;
      const key = d
        ? d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0")
        : "unknown";
      if (!current || current.key !== key) {
        current = { key, date: d, items: [] };
        groups.push(current);
      }
      current.items.push(item);
    }

    let html = "";
    for (const group of groups) {
      html += '<div class="tl-day">';
      html += '<div class="tl-date">' + esc(this._dayLabel(group.date, startOfToday, DAY_MS)) + '</div>';
      html += '<div class="tl-items">';
      for (const item of group.items) {
        const time = item._date
          ? String(item._date.getHours()).padStart(2, "0") + ":" + String(item._date.getMinutes()).padStart(2, "0")
          : "";
        html += '<div class="tl-card" data-id="' + esc(String(item.memory_id)) + '">';
        html += '<span class="tl-card-time">' + esc(time) + '</span>';
        html += '<div class="tl-card-main">';
        html += '<div class="tl-card-head">';
        html += '<span class="type-tag">' + esc(typeLabel(item.memory_type)) + '</span>';
        if (item.consolidated_count > 0) {
          html += '<span class="type-tag cons-badge" title="' + esc(window.t("table.consolidatedTitle")) + '">' + window.t("table.consolidated", item.consolidated_count) + '</span>';
        }
        if (item.status && item.status !== "active") {
          html += statusPill(item.status);
        }
        html += '</div>';
        html += '<div class="tl-card-text">' + esc(item.summary || "") + '</div>';
        html += '</div></div>';
      }
      html += '</div></div>';
    }

    container.innerHTML = html;
    this._updateMoreBtn();
  }

  _renderStats() {
    const totalEl = document.getElementById("tls-total");
    const nodesEl = document.getElementById("tls-nodes");
    const weekEl = document.getElementById("tls-week");
    const loadedEl = document.getElementById("tls-loaded");
    if (!totalEl) return;

    const stats = this._stats || {};
    totalEl.textContent = stats.total_memories != null ? stats.total_memories : "--";
    nodesEl.textContent = stats.graph_nodes != null ? stats.graph_nodes : "--";

    const weekAgo = Date.now() - 7 * 86400000;
    const weekNew = this.items.filter(i => i._date && i._date.getTime() >= weekAgo).length;
    weekEl.textContent = String(weekNew);
    loadedEl.textContent = window.t("timeline.loaded", this.items.length);
  }

  _dayLabel(date, startOfToday, DAY_MS) {
    if (!date) return "--";
    const dayStart = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
    if (dayStart === startOfToday) return window.t("timeline.today");
    if (dayStart === startOfToday - DAY_MS) return window.t("timeline.yesterday");

    const lang = (typeof window.getLanguage === "function" ? window.getLanguage() : "zh");
    const locale = lang === "zh" ? "zh-CN" : lang === "ru" ? "ru-RU" : "en-US";
    const sameYear = date.getFullYear() === new Date().getFullYear();
    return date.toLocaleDateString(locale, sameYear
      ? { month: "long", day: "numeric" }
      : { year: "numeric", month: "long", day: "numeric" });
  }

  _updateMoreBtn() {
    const btn = document.getElementById("tl-more-btn");
    if (!btn) return;
    btn.disabled = !this.hasMore;
    btn.textContent = this.hasMore
      ? window.t("timeline.loadMore")
      : window.t("timeline.noMore");
  }

  initEventListeners() {
    const moreBtn = document.getElementById("tl-more-btn");
    if (moreBtn) {
      moreBtn.addEventListener("click", async () => {
        if (this.loading || !this.hasMore) return;
        moreBtn.disabled = true;
        try {
          await this._fetchPage();
        } catch (e) {
          this._showToast(e.message || window.t("timeline.loadFail"), true);
          this._updateMoreBtn();
        }
      });
    }

    const container = document.getElementById("timeline-container");
    if (container) {
      container.addEventListener("click", (e) => {
        const card = e.target.closest(".tl-card");
        if (!card || !card.dataset.id) return;
        const id = Number(card.dataset.id);
        const item = this.items.find(i => i.memory_id === id);
        if (item) this.peek.renderMemory(item);
      });
    }
  }

  _showToast(message, isError = false) {
    if (window.lmShowToast) {
      window.lmShowToast(message, isError);
    }
  }
}
