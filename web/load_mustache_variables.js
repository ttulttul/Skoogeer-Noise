import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "LoadMustacheVariables";
const PATH_WIDGET = "file_path";

async function fetchListing(path, showHidden) {
    const params = new URLSearchParams({ path: path || "", show_hidden: showHidden ? "1" : "0" });
    const response = await api.fetchApi(`/skoogeer/mustache/browse?${params}`);
    const body = await response.json();
    if (!response.ok) {
        throw new Error(body.error || `Browse failed (${response.status})`);
    }
    return body;
}

function el(tag, style = {}, props = {}) {
    const node = document.createElement(tag);
    Object.assign(node.style, style);
    Object.assign(node, props);
    return node;
}

function openPathChooser(startPath) {
    return new Promise((resolve) => {
        const overlay = el("div", {
            position: "fixed", inset: "0", background: "rgba(0,0,0,0.5)", zIndex: "10000",
            display: "flex", alignItems: "center", justifyContent: "center",
        });
        const panel = el("div", {
            background: "var(--comfy-menu-bg, #222)", color: "var(--fg-color, #ddd)",
            border: "1px solid var(--border-color, #444)", borderRadius: "8px", padding: "12px",
            width: "min(640px, 92vw)", height: "min(560px, 85vh)", display: "flex", flexDirection: "column",
            gap: "8px", fontFamily: "sans-serif", fontSize: "13px",
        });
        const inputStyle = {
            background: "var(--comfy-input-bg, #111)", color: "inherit",
            border: "1px solid var(--border-color, #444)", borderRadius: "4px", padding: "4px 8px",
        };

        const title = el("div", { fontWeight: "bold", fontSize: "14px" }, { textContent: "Choose Mustache variables file" });
        const pathRow = el("div", { display: "flex", gap: "6px" });
        const upButton = el("button", inputStyle, { textContent: "↑ Up", title: "Parent directory" });
        const pathInput = el("input", { ...inputStyle, flex: "1", minWidth: "0" }, { type: "text", spellcheck: false });
        const goButton = el("button", inputStyle, { textContent: "Go" });
        pathRow.append(upButton, pathInput, goButton);

        const list = el("div", {
            flex: "1", overflowY: "auto", border: "1px solid var(--border-color, #444)", borderRadius: "4px",
            background: "var(--comfy-input-bg, #111)",
        });
        const status = el("div", { minHeight: "1.2em", color: "#e88" });

        const footer = el("div", { display: "flex", gap: "8px", alignItems: "center" });
        const hiddenLabel = el("label", { display: "flex", gap: "4px", alignItems: "center", marginRight: "auto" });
        const hiddenToggle = el("input", {}, { type: "checkbox" });
        hiddenLabel.append(hiddenToggle, document.createTextNode("Show hidden files"));
        const cancelButton = el("button", inputStyle, { textContent: "Cancel" });
        const selectButton = el("button", inputStyle, { textContent: "Use typed path", title: "Use the path in the text box as-is" });
        footer.append(hiddenLabel, cancelButton, selectButton);

        panel.append(title, pathRow, list, status, footer);
        overlay.append(panel);
        document.body.append(overlay);

        let current = null;

        const close = (value) => {
            document.removeEventListener("keydown", onKey, true);
            overlay.remove();
            resolve(value);
        };
        const onKey = (event) => {
            if (event.key === "Escape") {
                event.stopPropagation();
                close(null);
            }
        };
        document.addEventListener("keydown", onKey, true);

        const render = (listing) => {
            current = listing;
            pathInput.value = listing.path;
            upButton.disabled = !listing.parent;
            list.replaceChildren();
            if (listing.entries.length === 0) {
                list.append(el("div", { padding: "8px", opacity: "0.6" }, { textContent: "(empty directory)" }));
            }
            for (const entry of listing.entries) {
                const row = el("div", { padding: "3px 8px", cursor: "pointer", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, {
                    textContent: `${entry.is_dir ? "📁" : "📄"} ${entry.name}`,
                    title: entry.path,
                });
                row.addEventListener("mouseenter", () => { row.style.background = "rgba(255,255,255,0.08)"; });
                row.addEventListener("mouseleave", () => { row.style.background = ""; });
                row.addEventListener("click", () => {
                    if (entry.is_dir) {
                        navigate(entry.path);
                    } else {
                        close(entry.path);
                    }
                });
                list.append(row);
            }
        };

        const navigate = async (path) => {
            status.textContent = "";
            try {
                render(await fetchListing(path, hiddenToggle.checked));
            } catch (error) {
                status.textContent = error.message;
            }
        };

        upButton.addEventListener("click", () => current?.parent && navigate(current.parent));
        goButton.addEventListener("click", () => navigate(pathInput.value));
        pathInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") navigate(pathInput.value);
        });
        hiddenToggle.addEventListener("change", () => navigate(current?.path ?? startPath));
        cancelButton.addEventListener("click", () => close(null));
        selectButton.addEventListener("click", () => close(pathInput.value.trim() || null));
        overlay.addEventListener("mousedown", (event) => {
            if (event.target === overlay) close(null);
        });

        navigate(startPath).then(() => {
            // Fall back to the home directory if the previous location no longer exists.
            if (!current && startPath) navigate("");
        });
    });
}

app.registerExtension({
    name: "skoogeer.LoadMustacheVariables",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const pathWidget = this.widgets?.find((widget) => widget.name === PATH_WIDGET);
            const button = this.addWidget("button", "choose file", null, async () => {
                const chosen = await openPathChooser(pathWidget?.value || "");
                if (chosen && pathWidget) {
                    pathWidget.value = chosen;
                    pathWidget.callback?.(chosen);
                    this.setDirtyCanvas(true, true);
                }
            }, { serialize: false });
            // Keep the button out of widgets_values so saved workflows stay compatible.
            button.serialize = false;
            return result;
        };
    },
});
