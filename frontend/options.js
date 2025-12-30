const backendInput = document.getElementById("backendUrl");
const highlightToggle = document.getElementById("highlightToggle");
const mockToggle = document.getElementById("mockToggle");

backendInput.addEventListener("change", (e) => {
    chrome.storage.sync.set({ backendUrl: e.target.value });
});

highlightToggle.addEventListener("change", (e) => {
    chrome.storage.sync.set({ highlight: e.target.checked });
});

mockToggle.addEventListener("change", (e) => {
    chrome.storage.sync.set({ mockMode: e.target.checked });
});

window.onload = () => {
    chrome.storage.sync.get(["backendUrl", "highlight", "mockMode"], (data) => {
        if (data.backendUrl) backendInput.value = data.backendUrl;
        highlightToggle.checked = !!data.highlight;
        mockToggle.checked = !!data.mockMode;
    });
};
