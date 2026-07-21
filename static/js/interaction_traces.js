(function () { 
  $.getJSON("static/showcase_data/interaction/edits.json", function (data) {
    const interactions = data;
    const editorText = document.getElementById("careEditorText");
    const playBtn = document.getElementById("carePlayBtn");
    const stopBtn = document.getElementById("careStopBtn");
    const resetBtn = document.getElementById("careResetBtn");
    const queryPopup = document.getElementById("careQueryPopup");
    const queryText = document.getElementById("careQueryText");
    const responseBox = document.getElementById("careResponseBox");
    const sidebarPane = document.getElementById("careSidebarPane");
    const acceptBtn = document.getElementById("careAcceptButton");
    const rejectBtn = document.getElementById("careRejectButton");
    const timer = document.getElementById("careTimer");
    const wordCount = document.getElementById("careWordCount");

    let hasStarted = false;
    let isPaused = false;
    let docText = "";
    let pendingQueries = {};

    // ---- Virtual clock: single source of truth for scheduling ----------
    // Everything schedules against a "virtual time" that only advances
    // while playing. Pausing simply stops the ticker; nothing that was
    // scheduled can fire until resume() restarts it. This avoids having
    // to individually pause/resume dozens of independent setTimeouts.
    const clock = (function () {
      let running = false;
      let virtualTime = 0;
      let lastTick = null;
      let pending = []; // { time, fn, id }
      let nextId = 1;
      let intervalHandle = null;

      function tick() {
        const now = performance.now();
        if (lastTick !== null) {
          virtualTime += now - lastTick;
        }
        lastTick = now;

        if (pending.length === 0) return;
        const due = pending.filter(p => p.time <= virtualTime);
        if (due.length) {
          pending = pending.filter(p => p.time > virtualTime);
          due.sort((a, b) => a.time - b.time);
          due.forEach(p => p.fn());
        }
      }

      return {
        start() {
          if (running) return;
          running = true;
          lastTick = performance.now();
          intervalHandle = setInterval(tick, 30);
        },
        pause() {
          if (!running) return;
          running = false;
          clearInterval(intervalHandle);
          intervalHandle = null;
          lastTick = null;
        },
        reset() {
          running = false;
          if (intervalHandle) clearInterval(intervalHandle);
          intervalHandle = null;
          virtualTime = 0;
          lastTick = null;
          pending = [];
        },
        schedule(delay, fn) {
          const id = nextId++;
          pending.push({ time: virtualTime + Math.max(0, delay), fn, id });
          return id;
        },
        cancel(id) {
          pending = pending.filter(p => p.id !== id);
        },
        isRunning() {
          return running;
        }
      };
    })();
    // ----------------------------------------------------------------------

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
      clock.schedule(20, () => typeIntoPopup(text, idx + 1, onComplete));
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
      let ranges = [];
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
      clock.schedule(baseDelayMs, () => {
        applyDeferredOpsBatch(ops);
      });
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

        clock.schedule(1000, () => {
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
          clock.schedule(decisionDelayMs, () => {
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
          });
        });
      }
    }

    function finishPlayback() {
        clock.pause();
        hasStarted = false;
        isPaused = false;
        setButtonsToPlay();
    }

    function scaledDelay(delta) {
      return Math.sqrt(delta) * 200;
    }

    function scheduleAll() {
      let cumulativeMs = 0;
      let prevCreatedAt = interactions[0].createdAt;

      interactions.forEach((ev, i) => {
        if (deferredOpIndices.has(i)) {
          prevCreatedAt = ev.createdAt;
          return;
        }
        const delta = ev.createdAt - prevCreatedAt;
        cumulativeMs += scaledDelay(delta);
        prevCreatedAt = ev.createdAt;

        clock.schedule(cumulativeMs, () => {
          applyEvent(ev, i);
          timer.textContent = formatTimer(ev.createdAt);
          if (i === interactions.length - 1) {
              finishPlayback();
          }
        });

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

    function setButtonsToPlay() {
      playBtn.style.display = "";
      stopBtn.style.display = "none";
    }

    function setButtonsToStop() {
      playBtn.style.display = "none";
      stopBtn.style.display = "";
    }

    playBtn.addEventListener("click", () => {
      if (hasStarted && !isPaused) {
        return; // already running
      }

      if (!hasStarted) {
        hasStarted = true;
        scheduleAll();
      }
      isPaused = false;
      clock.start();
      setButtonsToStop();
    });

    stopBtn.addEventListener("click", () => {
      if (!hasStarted || isPaused) {
        return;
      }
      isPaused = true;
      clock.pause();
      setButtonsToPlay();
    });

    resetBtn.addEventListener("click", () => {
      clock.reset();
      hasStarted = false;
      isPaused = false;
      resetReplay();
      setButtonsToPlay();
    });
  });
})();