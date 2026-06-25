(function() {
    const notions = {
        "spanLevel": document.getElementById("spanLevel"),
        "documentLevel": document.getElementById("documentLevel"),
        "sentenceLevel": document.getElementById("sentenceLevel"),
        "boundaryLevel": document.getElementById("boundaryLevel"),
        "contentLevel": document.getElementById("contentLevel"),
        "intentLevel": document.getElementById("intentLevel"),
        "authorshipLevel": document.getElementById("authorshipLevel"),
        "membershipLevel": document.getElementById("membershipLevel"),
    }

    const boundaryPath = "sessions/";

    for (const [id, el] of Object.entries(notions)) {
        el.addEventListener("click", () => selectNotion(id));
    };

    function selectNotion(notion)  {
        
    }
})();