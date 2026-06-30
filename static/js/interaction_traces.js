(function () { 
  $.getJSON("static/showcase_data/interaction/edits.json", function (data) {
    const interactions = data;
    const editorText = document.getElementById("careEditorText");
    const playBtn = document.getElementById("carePlayBtn");
    const resetBtn = document.getElementById("careResetBtn");
    const queryPopup = document.getElementById("careQueryPopup");
    const queryText = document.getElementById("careQueryText");
    const responseBox = document.getElementById("careResponseBox");
    const sidebarPane = document.getElementById("careSidebarPane");
    const acceptBtn = document.getElementById("careAcceptButton");
    const rejectBtn = document.getElementById("careRejectButton");
    const timer = document.getElementById("careTimer");
    const wordCount = document.getElementById("careWordCount");

    let isPlaying = false;
    let docText = "";
    let timeouts = [];
    let pendingQueries = {};

    const deferredOpIndices = new Set();
    interactions.forEach((ev, i) => {
      if (ev.requestId !== undefined && ev.accepted === "t") {
        findDeferredOpsForResponse(i).forEach(({index}) => deferredOpIndices.add(index));
      }
    });

    function findDeferredOpsForResponse(responseIdx) {
      const response = interactions[responseIdx];
      const ops = [];

      for (let i = responseIdx + 1; i < interactions.length; i++) {
        const ev = interactions[i];
        if (!(ev.user === "Bot")) break;
        if (ev.createdAt < response.decidedAt) break;
        ops.push({ev, index: i});
      }
      return ops;
    }

    function compute_new_html(text, highlightRange) {
      let html = escapeHtml(text);
        if (highlightRange) {
        const { start, end, cls } = highlightRange;
        const before = escapeHtml(text.slice(0, start));
        const mid = escapeHtml(text.slice(start, end));
        const after = escapeHtml(text.slice(end));
        html = before + '<span class="' + cls + '">' + mid + '</span>' + after;
        }
      return html;
    }

    function compute_new_html_batch(text, highlightRanges) {
      let html = "";
      let cursor = 0;
      highlightRanges.forEach(({start, end}) => {
        html += escapeHtml(text.slice(cursor, start));
        const mid = escapeHtml(text.slice(start, end));
        html += '<span class="care-ins-bot">' + mid + '</span>';
        cursor = end;
      })
      html += escapeHtml(text.slice(cursor));
      return html;
    }

    function renderEditor(highlightRange) {
        editorText.innerHTML = compute_new_html(docText, highlightRange);
        updateWordCount();
    }

    function renderEditorBatch(highlightRanges) {
      editorText.innerHTML = compute_new_html_batch(docText, highlightRanges);
      updateWordCount();
    }

    function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function updateWordCount() {
      const words = docText.trim().split(/\s+/).filter(Boolean).length;
      wordCount.textContent = 'Word count: ' + words;
    }

    function formatTimer(seconds) {
      const m = Math.floor(seconds / 60).toString().padStart(2, "0");
      const s = Math.floor(seconds % 60).toString().padStart(2, "0");
      return m + ":" + s;
    }

    function typeIntoPopup(text, idx, onComplete) {
      queryText.textContent = text.slice(0, idx);
      if (idx >= text.length) {
        if (onComplete) onComplete();
        return;
      }
      const t = setTimeout(() => typeIntoPopup(text, idx + 1, onComplete), 20);
      timeouts.push(t);
    }

    function constructAnswerText(rsp, req) {
      let text = req.textContent;
      return text.slice(0, req.selectionIndex) + rsp.response + text.slice(req.selectionIndex + req.selectionLength);
    }

    function showGenerating() {
      responseBox.innerHTML =
        '<span class="care-generating">' +
          '<span class="care-spinner"></span>' +
          '<span>Generating…</span>' +
        '</span>';
      acceptBtn.classList.remove("is-active-accept");
      rejectBtn.classList.remove("is-active-reject");
    }

    function applyDeferredOpsBatch(ops) {
      ranges = [];
      ops.forEach(op => {
        if (op.operationType === "insert") {
          docText = docText.slice(0, op.offset) + op.text + docText.slice(op.offset);
          ranges.push({ start: op.offset, end: op.offset + op.text.length})
        } else if (op.operationType === "delete") {
          const end = op.offset + op.span;
          docText = docText.slice(0, op.offset) + docText.slice(end);
        }
      });
      if (ranges.length > 0) renderEditorBatch(ranges)
      else renderEditor();
    }

    function scheduleDeferredOps(ops, decidedAt, baseDelayMs) {
      const t = setTimeout(() => {
        applyDeferredOpsBatch(ops);
      }, baseDelayMs);
      timeouts.push(t);
      return baseDelayMs;
    }

    function applyEvent(ev, idx) {
      if (ev.operationType === "insert") {
          const text = ev.text;
          docText = docText.slice(0, ev.offset) + text + docText.slice(ev.offset);
          const cls = ev.user === 'Bot' ? 'care-ins-bot' : 'care-ins-user';
          renderEditor({ start: ev.offset, end: ev.offset + text.length, cls });

      } else if (ev.operationType === "delete") {
          const end = ev.offset + ev.span;
          renderEditor({ start: ev.offset, end, cls: 'care-fade-out' });
          docText = docText.slice(0, ev.offset) + docText.slice(end);
      } else if (ev.query !== undefined) {
        ev.textContent = editorText.textContent;
        pendingQueries[ev.id] = ev;

        queryPopup.style.display = "flex";
        queryText.textContent = "";
        if (ev.query !== null) {
          typeIntoPopup(ev.query, 0, showGenerating);
        } else {
          showGenerating();
        }

      } else if (ev.requestId !== undefined) {
        queryPopup.style.display = "none";
        const req = pendingQueries[ev.requestId];
        const accepted = ev.accepted === "t";
        const answerText = constructAnswerText(ev, req);

        const t = setTimeout(() => {
          responseBox.innerHTML = "";
          const respSpan = document.createElement("span");
          const html =  compute_new_html(answerText,{
            start: req.selectionIndex,
            end: req.selectionIndex + ev.response.length,
            cls: "care-answer-pending"});
          respSpan.innerHTML = html;
          respSpan.id = "careActiveAnswerSpan";
          responseBox.appendChild(respSpan);

          const decisionDelayMs = scaledDelay(ev.decidedAt - ev.createdAt);
          const tm = setTimeout(() => {
            const activeSpan = document.getElementById("careActiveAnswerSpan");
            if (activeSpan) {
              const cls = accepted ? "care-answer-accepted": "care-answer-rejected";
              const finalHtml = compute_new_html(answerText,{
                start: req.selectionIndex,
                end: req.selectionIndex + ev.response.length,
                cls
              })
              activeSpan.innerHTML = finalHtml;
              acceptBtn.classList.remove("is-active-accept");
              rejectBtn.classList.remove("is-active-reject");

              if (accepted) {
                acceptBtn.classList.add("is-active-accept");
              } else {
                rejectBtn.classList.add("is-active-reject");
              }
            }
            if (accepted) {
              const ops = findDeferredOpsForResponse(idx).map(o => o.ev);
              scheduleDeferredOps(ops, ev.decidedAt, 0);
            }
          }, decisionDelayMs);
          timeouts.push(tm);
        }, 1000);
        timeouts.push(t);
      }
    }

    function finishPlayback() {
        isPlaying = false;
    }

    function scaledDelay(delta) {
      return Math.sqrt(delta) * 200;
    }

    function scheduleAll() {
      let cumulativeMs = 0;
      let prevCreatedAt = interactions[0].createdAt;

      interactions.forEach((ev, i) => {
        if (!isPlaying) return;
        if (deferredOpIndices.has(i)) {
          prevCreatedAt = ev.createdAt;
          return;
        }
        const delta = ev.createdAt - prevCreatedAt;
        cumulativeMs += scaledDelay(delta);
        prevCreatedAt = ev.createdAt;

        const t = setTimeout(() => {
          applyEvent(ev, i);
          timer.textContent = formatTimer(ev.createdAt);
          timeouts.push(t);
          if (i === interactions.length - 1) {
              finishPlayback();
          }
        }, cumulativeMs);
        timeouts.push(t);

        if (ev.requestId !== undefined && ev.decidedAt !== undefined) {
          const decisionDelayMs = scaledDelay(ev.decidedAt - ev.createdAt);
          cumulativeMs += 1000 + decisionDelayMs;

          if (ev.accepted === "t") {
            const ops = findDeferredOpsForResponse(i).map(o => o.ev);
            let prevOpCreatedAt = ev.decidedAt;
            let opsSpanMs = 0;
            ops.forEach(opEv => {
              opsSpanMs += Math.max(0, scaledDelay(opEv.createdAt - prevOpCreatedAt));
              prevOpCreatedAt = opEv.createdAt;
            });
            cumulativeMs += opsSpanMs;
          }
        }
      })
    }

    function clearAllTimeouts() {
      timeouts.forEach(t => clearTimeout(t));
      timeouts = [];
    }

    function resetReplay() {
      docText = "";
      renderEditor();
      responseBox.innerHTML = "";
      timer.textContent = "00:00";
      
      responseBox.innerHTML = "";
      const placeholder = document.createElement("span");
      placeholder.className = "care-response-placeholder";
      placeholder.id = "careResponsePlaceholder";
      placeholder.textContent = "The model answer will be displayed here";
      responseBox.appendChild(placeholder);

      queryPopup.style.display = "none";

      acceptBtn.classList.remove("is-active-accept");
      rejectBtn.classList.remove("is-active-reject");
    }

  playBtn.addEventListener("click", () => {
    if (isPlaying) {
        return;
    }
    isPlaying = true;
    scheduleAll();
  });

  resetBtn.addEventListener("click", () => {
    clearAllTimeouts();
    isPlaying = false;
    resetReplay();
  });
});
})();