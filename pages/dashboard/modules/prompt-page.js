/**
 * Prompt Page - 提示词管理页面
 * 集中管理插件所有可自定义的提示词模板
 */

import { esc, confirmDiscardChanges } from "./utils.js";

export class PromptPage {
  constructor(state, apiClient) {
    this.state = state;
    this.api = apiClient;
    this.prompts = [];
    this.categories = [];
    this.editingId = null;
    this.editContent = null;
    this._resetMode = false;
    this._editorGeneration = 0;
    this._editorHome = null;    // 编辑器在 index.html 中的原始挂载点
    this._editorDetail = null;  // 当前编辑提示词的详情（用于重渲染后恢复）
  }

  /**
   * 按当前语言选择中/英文文案
   * 非中文语言优先英文字段，缺失时回退中文
   * @param {string} zhText - 中文字段
   * @param {string} enText - 英文字段
   * @returns {string}
   */
  _pickLang(zhText, enText) {
    const lang = window.getLanguage ? window.getLanguage() : "zh";
    if (lang === "zh") return zhText || enText || "";
    return enText || zhText || "";
  }

  /**
   * 获取提示词列表
   */
  async fetch() {
    try {
      const data = await this.api.get("prompts");
      this.prompts = data.prompts || [];
      this.categories = data.categories || [];
      this.render();
    } catch (e) {
      this.showToast(e.message || window.t("prompt.fetchFailed"), true);
    }
  }

  /**
   * 渲染主页面
   */
  render() {
    const container = document.getElementById("prompt-content");
    if (!container) return;

    // 若编辑器正打开：先记下草稿并把它挪回原始挂载点，避免被下面的 innerHTML 重建销毁
    const editorEl = document.getElementById("prompt-editor");
    const wasOpen = Boolean(this.editingId && editorEl && !editorEl.classList.contains("hidden"));
    let draft = null;
    if (wasOpen && editorEl) {
      const ta = document.getElementById("prompt-editor-textarea");
      draft = ta ? ta.value : null;
      if (this._editorHome && editorEl.parentNode !== this._editorHome) {
        this._editorHome.appendChild(editorEl);
      }
    }

    if (!this.prompts.length) {
      container.innerHTML =
        '<div class="table-empty">' + window.t("prompt.noPrompts") + "</div>";
      return;
    }

    // 按分类分组
    const grouped = {};
    this.prompts.forEach((p) => {
      const cat = p.category || "other";
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(p);
    });

    let html = "";

    // 顶部警告
    html +=
      '<div class="prompt-warning">' +
      '<div class="prompt-warning-title">' +
      esc(window.t("prompt.warningTitle")) +
      "</div>" +
      '<div class="prompt-warning-text">' +
      esc(window.t("prompt.warningText")) +
      "</div></div>";

    for (const [catId, prompts] of Object.entries(grouped)) {
      const catInfo = this.categories.find((c) => c.id === catId) || {};
      const catName = this._pickLang(catInfo.name, catInfo.name_en) || catId;
      // 中文界面附英文副标题，其他语言只显示当前语言名称
      const catNameAlt =
        (window.getLanguage ? window.getLanguage() : "zh") === "zh"
          ? catInfo.name_en || ""
          : "";
      const catDesc = this._pickLang(catInfo.description, catInfo.description_en);

      html += '<div class="prompt-category">';
      html +=
        '<div class="prompt-category-header">' +
        '<span class="prompt-category-name">' +
        esc(catName) +
        "</span>";
      if (catNameAlt)
        html +=
          ' <span class="prompt-category-name-en">' + esc(catNameAlt) + "</span>";
      if (catDesc)
        html +=
          '<p class="prompt-category-desc">' + esc(catDesc) + "</p>";
      html += "</div>";

      html += '<div class="prompt-list">';
      prompts.forEach((p) => {
        const customBadge = p.is_custom
          ? ' <span class="badge badge-custom">' +
            window.t("prompt.customized") +
            "</span>"
          : "";
        const jsonBadge =
          p.category === "memory_processing"
            ? ' <span class="badge badge-json-warn" title="' +
              esc(window.t("prompt.jsonRequired")) +
              '">JSON</span>'
            : "";
        const varList = (p.variables || [])
          .map((v) => '<code>' + esc(v) + "</code>")
          .join(" ");
        const descText = this._pickLang(p.description, p.description_en);

        html +=
          '<div class="prompt-item" data-id="' +
          esc(p.id) +
          '">' +
          '<div class="prompt-item-header">' +
          '<span class="prompt-item-name">' +
          esc(this._pickLang(p.name, p.name_en) || p.id) +
          jsonBadge +
          customBadge +
          "</span>";
        const nameAlt =
          (window.getLanguage ? window.getLanguage() : "zh") === "zh"
            ? p.name_en || ""
            : "";
        if (nameAlt)
          html +=
            '<span class="prompt-item-name-en">' + esc(nameAlt) + "</span>";
        html += "</div>";
        if (descText)
          html +=
            '<p class="prompt-item-desc">' + esc(descText) + "</p>";
        const usageNote = this._pickLang(p.usage_note, p.usage_note_en);
        if (usageNote)
          html +=
            '<p class="prompt-item-usage">' + esc(usageNote) + "</p>";
        if (varList)
          html +=
            '<div class="prompt-item-vars">' +
            window.t("prompt.variables") +
            ": " +
            varList +
            "</div>";
        html +=
          '<div class="prompt-item-actions">' +
          '<button class="btn btn-sm btn-secondary prompt-edit-btn" data-id="' +
          esc(p.id) +
          '"><i data-lucide="square-pen" aria-hidden="true"></i><span>' +
          window.t("prompt.edit") +
          "</span></button>" +
          "</div>";
        html += "</div>";
      });
      html += "</div></div>";
    }

    container.innerHTML = html;
    if (window.lmHydrateIcons) window.lmHydrateIcons();

    // 绑定编辑按钮事件
    container.querySelectorAll(".prompt-edit-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        this.openEditor(id);
      });
    });

    // 重渲染后恢复仍在打开的编辑器（内联回到原条目下方）
    if (wasOpen) this._remountEditor(draft);
  }

  /**
   * 把编辑器内联挂载到指定提示词条目的下方
   * @param {string} promptId - 提示词ID
   * @returns {boolean} 是否挂载成功
   */
  _mountEditorInline(promptId) {
    const container = document.getElementById("prompt-content");
    const editorEl = document.getElementById("prompt-editor");
    if (!container || !editorEl) return false;
    const target = Array.from(container.querySelectorAll(".prompt-item"))
      .find((el) => el.dataset.id === promptId);
    if (!target) return false;
    target.insertAdjacentElement("afterend", editorEl);
    return true;
  }

  /**
   * 列表重渲染后，把仍处于打开状态的编辑器挂回原条目下方并恢复内容
   * @param {string|null} draft - 重渲染前编辑器里的草稿
   */
  _remountEditor(draft) {
    const editorEl = document.getElementById("prompt-editor");
    if (!editorEl || !this.editingId) return;
    if (!this._mountEditorInline(this.editingId)) {
      editorEl.classList.add("hidden");
      return;
    }

    const prompt = this.prompts.find((p) => p.id === this.editingId);
    const detail = this._editorDetail || {};
    const titleEl = document.getElementById("prompt-editor-title");
    if (prompt && titleEl) {
      titleEl.textContent = this._pickLang(prompt.name, prompt.name_en) || prompt.id;
    }
    const varsEl = document.getElementById("prompt-editor-vars");
    if (varsEl) {
      varsEl.innerHTML = (detail.variables || [])
        .map((v) => '<code class="prompt-var-tag">' + esc(v) + "</code>")
        .join(" ");
    }
    const statusEl = document.getElementById("prompt-editor-status");
    if (statusEl) {
      statusEl.textContent = detail.is_custom
        ? " " + window.t("prompt.customizedStatus")
        : " " + window.t("prompt.defaultStatus");
    }
    const textarea = document.getElementById("prompt-editor-textarea");
    if (textarea) textarea.value = draft !== null ? draft : this.editContent;
    editorEl.classList.remove("hidden");
    editorEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /**
   * 打开编辑器
   * @param {string} promptId - 提示词ID
   */
  async openEditor(promptId) {
    const generation = ++this._editorGeneration;
    if (!await this.canLeave() || generation !== this._editorGeneration) return;
    const prompt = this.prompts.find((p) => p.id === promptId);
    if (!prompt) return;

    this._opening = true;
    this.updateEditorControls();
    try {
      const detail = await this.api.get("prompts/detail", { id: promptId });
      if (generation !== this._editorGeneration) return;
      this.editingId = promptId;
      this.editContent = detail.content || "";
      this._editorDetail = detail;

      const editorEl = document.getElementById("prompt-editor");
      if (!editorEl) return;
      if (!this._editorHome) this._editorHome = editorEl.parentNode;

      document.getElementById("prompt-editor-title").textContent =
        this._pickLang(prompt.name, prompt.name_en) || prompt.id;
      document.getElementById("prompt-editor-textarea").value =
        this.editContent;
      document.getElementById("prompt-editor-vars").innerHTML = (
        detail.variables || []
      )
        .map(
          (v) =>
            '<code class="prompt-var-tag">' + esc(v) + "</code>"
        )
        .join(" ");
      document.getElementById("prompt-editor-status").textContent = detail
        .is_custom
        ? " " + window.t("prompt.customizedStatus")
        : " " + window.t("prompt.defaultStatus");

      // 内联展开：挂载到被点击条目的正下方，而不是页面底部
      if (!this._mountEditorInline(promptId) && this._editorHome) {
        this._editorHome.appendChild(editorEl);
      }
      editorEl.classList.remove("hidden");

      // 绑定按钮事件
      const saveBtn = document.getElementById("prompt-save-btn");
      const resetBtn = document.getElementById("prompt-reset-btn");
      const cancelBtn = document.getElementById("prompt-cancel-btn");
      const textarea = document.getElementById("prompt-editor-textarea");

      const newSaveBtn = saveBtn.cloneNode(true);
      const newResetBtn = resetBtn.cloneNode(true);
      const newCancelBtn = cancelBtn.cloneNode(true);
      saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
      resetBtn.parentNode.replaceChild(newResetBtn, resetBtn);
      cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);

      newSaveBtn.addEventListener("click", () => this.savePrompt());
      newResetBtn.addEventListener("click", () => this.resetPrompt());
      newCancelBtn.addEventListener("click", () => this.closeEditor());

      // 检测修改（textarea 是常驻节点，不能像按钮一样 clone 替换；
      // 用可移除的具名 handler 防止每次打开编辑器都叠加监听）
      if (this._textareaInputHandler) {
        textarea.removeEventListener("input", this._textareaInputHandler);
      }
      this._textareaInputHandler = () => {
        // 手动编辑后退出恢复默认模式
        if (this._resetMode) {
          this._resetMode = false;
        }
        const modified = textarea.value !== this.editContent;
        const currentSaveBtn = document.getElementById("prompt-save-btn");
        if (currentSaveBtn) currentSaveBtn.disabled = this._busy || !modified;
      };
      textarea.addEventListener("input", this._textareaInputHandler);
      newSaveBtn.disabled = true;
      this._resetMode = false;

      this._opening = false;
      this.updateEditorControls();
      textarea.focus({ preventScroll: true });
      editorEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
      if (generation === this._editorGeneration) this.showToast(e.message || window.t("prompt.loadFailed"), true);
    } finally {
      if (generation === this._editorGeneration) {
        this._opening = false;
        this.updateEditorControls();
      }
    }
  }

  /**
   * 保存提示词
   */
  async savePrompt() {
    if (this._busy || !this.editingId || !this.hasUnsavedChanges()) return;
    const textarea = document.getElementById("prompt-editor-textarea");
    if (!textarea) return;
    const content = textarea.value;

    this._busy = true;
    this.updateEditorControls();
    const promptId = this.editingId;

    try {
      if (this._resetMode) {
        await this.api.post("prompts/reset", { id: promptId }, { retries: 0 });
        this._resetMode = false;
        this.showToast(window.t("prompt.resetDone"));
      } else {
        await this.api.post("prompts/update", {
          id: promptId,
          content: content,
        }, { retries: 0 });
        this.showToast(window.t("prompt.saved"));
      }
      this.editContent = content;
      await this.fetch();
      await this.closeEditor({ force: true });
    } catch (e) {
      console.error("[PromptPage] savePrompt failed:", e);
      this.showToast(e.message || window.t("prompt.saveFailed"), true);
    } finally {
      this._busy = false;
      this.updateEditorControls();
    }
  }

  /**
   * 填充默认内容到编辑器（不保存，需手动点保存）
   */
  async resetPrompt() {
    if (!await this.canLeave()) return;
    if (!this.editingId) {
      console.error("[PromptPage] resetPrompt: editingId is empty");
      return;
    }

    this._busy = true;
    this.updateEditorControls();
    const promptId = this.editingId;

    try {
      const result = await this.api.get("prompts/default", {
        id: promptId,
      });
      const newContent = result.content || "";
      const textarea = document.getElementById("prompt-editor-textarea");
      if (textarea) {
        textarea.value = newContent;
      }
      this._resetMode = true;
      const saveBtn = document.getElementById("prompt-save-btn");
      if (saveBtn) saveBtn.disabled = false;
      const statusEl = document.getElementById("prompt-editor-status");
      if (statusEl)
        statusEl.textContent = " " + window.t("prompt.defaultFilledStatus");
    } catch (e) {
      console.error("[PromptPage] resetPrompt failed:", e);
      this.showToast(e.message || window.t("prompt.resetFailed"), true);
    } finally {
      this._busy = false;
      this.updateEditorControls();
    }
  }

  /**
   * 关闭编辑器
   */
  async closeEditor({ force = false } = {}) {
    if (!force && !await this.canLeave()) return false;
    this._editorGeneration++;
    this._opening = false;
    const editorEl = document.getElementById("prompt-editor");
    if (editorEl) editorEl.classList.add("hidden");
    this.editingId = null;
    this.editContent = null;
    this._resetMode = false;
    return true;
  }

  hasUnsavedChanges() {
    if (!this.editingId) return false;
    return this._resetMode || document.getElementById("prompt-editor-textarea").value !== this.editContent;
  }

  async canLeave() {
    if (this._busy) {
      this.showToast(window.t("flow.waitForSave"));
      return false;
    }
    return !this.hasUnsavedChanges() || await confirmDiscardChanges();
  }

  updateEditorControls() {
    const feedback = document.getElementById("prompt-editor-feedback");
    if (feedback) {
      feedback.hidden = !this._busy && !this._opening;
      feedback.textContent = (this._busy || this._opening) ? window.t("flow.working") : "";
    }
    const editor = document.getElementById("prompt-editor");
    if (editor) editor.inert = Boolean(this._busy || this._opening);
    const save = document.getElementById("prompt-save-btn");
    if (save) save.disabled = this._busy || !this.hasUnsavedChanges();
    const reset = document.getElementById("prompt-reset-btn");
    if (reset) reset.disabled = Boolean(this._busy);
  }

  /**
   * 语言切换后刷新已打开编辑器的标题
   */
  refreshEditorTitle() {
    if (!this.editingId) return;
    const prompt = this.prompts.find((p) => p.id === this.editingId);
    const titleEl = document.getElementById("prompt-editor-title");
    if (prompt && titleEl) {
      titleEl.textContent = this._pickLang(prompt.name, prompt.name_en) || prompt.id;
    }
  }

  /**
   * 显示 Toast 提示
   * @param {string} message - 提示消息
   * @param {boolean} isError - 是否为错误
   */
  showToast(message, isError = false) {
    if (window.lmShowToast) {
      window.lmShowToast(message, isError);
    }
  }
}
