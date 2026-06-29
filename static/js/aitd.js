(function() {
    const notionsPath = "notions/";
    const notions = [
        "spanLevel",
        "documentLevel",
        "sentenceLevel",
        "contentLevel",
        "intentLevel",
        "membershipLevel",
        "boundaryLevel",
    ]
    const notionsData = [];
    const textBody = document.getElementById("aitdTextBody");

    notions.forEach(notion => {
        $.getJSON("notions/" + notion + ".json", function (data) {
            const el = document.getElementById(notion);
            el.addEventListener("click", () => renderSegments(data, notion));
            notionsData.push({ el, data });
        });
    });

    function renderSegments(segments, notion) {
        const html = segments.map((seg, i) => {
            const cls = seg.author === "User"? "aitd-seg-user" : "aitd-seg-bot";
            const nextText = i < segments.length - 1 ? segments[i + 1].text : null;
            return '<span class="aitd-seg ' + cls + '">' + constructText(seg.text, notion, nextText) + '</span>';
        }).join("");
        textBody.innerHTML = html;
    }

    function constructText(data, notion, nextText) {
        if (["documentLevel", "spanLevel", "boundaryLevel"].includes(notion)) return data;
        if (notion === "membershipLevel") { 
            if (data == "(" || [")", ",", "."].includes(nextText)) return data;
            if (nextText === null) return data + ".";
            return data + " ";
        }
        return data + " ";
    }

})();