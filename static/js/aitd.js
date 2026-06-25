(function() {
    const notionsPath = "notions/";
    const notions = [
        "spanLevel",
        "documentLevel",
        "sentenceLevel",
        "contentLevel",
        "intentLevel",
        "membershipLevel",
        // "boundaryLevel",
    ]
    const notionsData = [];
    const textBody = document.getElementById("aitdTextBody");

    notions.forEach(notion => {
        $.getJSON("notions/" + notion + ".json", function (data) {
            const el = document.getElementById(notion);
            el.addEventListener("click", () => renderSegments(data));
            notionsData.push({ el, data });
        });
    });

    function renderSegments(segments) {
        console.log("Segments: ", segments);
        const html = segments.map(seg => {
            const cls = seg.author === "User"? "aitd-seg-user" : "aitd-seg-bot";
            return '<span class="aitd-seg ' + cls + '">' + escapeHtml(seg.text)+'</span>';
        }).join("");
        textBody.innerHTML = html;
    }

    function escapeHtml(html) {
        return String(html)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }
})();