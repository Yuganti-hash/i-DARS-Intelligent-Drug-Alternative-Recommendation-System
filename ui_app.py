"""
ui_app.py — Advanced Drug Alternative Recommendation System
Premium dark UI. Charcoal / Amber / Emerald palette. Zero blue.

Layout:
  Left rail  (340px fixed): mode tabs, inputs, run button, MRU splay panel
  Right panel (expanding) : scrollable recommendation cards
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import RecommendationEngine
from generate_dataset import generate_dataset

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "medicines_dataset.csv")

# ── Colour palette (no blue) ─────────────────────────────────────────
C = {
    "bg":          "#0C0C0E",
    "surface":     "#141418",
    "surface2":    "#1C1C22",
    "border":      "#2A2A35",
    "border_hi":   "#3D3D4E",
    "amber":       "#E8A020",
    "amber_dim":   "#7A5510",
    "amber_light": "#F5C84A",
    "emerald":     "#34C97A",
    "emerald_dim": "#1A6640",
    "rose":        "#E05555",
    "rose_dim":    "#6B2020",
    "lavender":    "#B89FE0",
    "lavender_dim":"#4A3B6B",
    "text":        "#EEEEF5",
    "text_sub":    "#9090A8",
    "text_muted":  "#50505E",
    "disc_bg":     "#2A0A0A",
    "disc_fg":     "#E07070",
}

F = {
    "h1":    ("Segoe UI Semibold", 20),
    "h2":    ("Segoe UI Semibold", 13),
    "h3":    ("Segoe UI", 12, "bold"),
    "body":  ("Segoe UI", 11),
    "small": ("Segoe UI", 9),
    "mono":  ("Consolas", 10),
    "score": ("Consolas", 13, "bold"),
    "badge": ("Segoe UI", 8, "bold"),
    "price": ("Segoe UI Semibold", 15),
}

AGE_GROUPS = ["Adult", "Child", "Senior"]


class DrugRecommenderApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Advanced Drug Alternative Recommendation System")
        self.configure(bg=C["bg"])
        self.geometry("1260x860")
        self.minsize(980, 700)

        self.engine: RecommendationEngine = None
        self.engine_ready = False

        self._setup_styles()
        self._build_ui()
        threading.Thread(target=self._load_engine, daemon=True).start()

    # ── ttk styles ────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("RX.TNotebook", background=C["bg"], borderwidth=0, tabmargins=0)
        s.configure("RX.TNotebook.Tab",
                    background=C["surface2"], foreground=C["text_sub"],
                    padding=[16, 7], font=F["body"], borderwidth=0)
        s.map("RX.TNotebook.Tab",
              background=[("selected", C["amber"]), ("active", C["border_hi"])],
              foreground=[("selected", C["bg"]), ("active", C["text"])])

        s.configure("RX.TCombobox",
                    fieldbackground=C["surface2"], background=C["surface"],
                    foreground=C["text"], selectbackground=C["amber"],
                    selectforeground=C["bg"], arrowcolor=C["amber"],
                    bordercolor=C["border"], lightcolor=C["border"],
                    darkcolor=C["border"])
        s.map("RX.TCombobox",
              fieldbackground=[("readonly", C["surface2"])],
              foreground=[("readonly", C["text"])])

        s.configure("RX.Vertical.TScrollbar",
                    background=C["surface2"], troughcolor=C["bg"],
                    bordercolor=C["bg"], arrowcolor=C["text_muted"], relief="flat")

    # ── Main layout ───────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_disclaimer()

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=18, pady=(6, 12))

        rail = tk.Frame(body, bg=C["bg"], width=340)
        rail.pack(side="left", fill="y", padx=(0, 12))
        rail.pack_propagate(False)
        self._build_rail(rail)

        results_col = tk.Frame(body, bg=C["bg"])
        results_col.pack(side="left", fill="both", expand=True)
        self._build_results_panel(results_col)

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["surface"])
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=C["amber"], width=4).pack(side="left", fill="y")
        inner = tk.Frame(hdr, bg=C["surface"], pady=14, padx=18)
        inner.pack(side="left", fill="both", expand=True)
        tk.Label(inner, text="Rx  Drug Alternative Recommendation System",
                 font=F["h1"], fg=C["text"], bg=C["surface"]).pack(side="left")
        self.status_lbl = tk.Label(inner, text="⏳  Initialising…",
                                    font=F["small"], fg=C["amber_dim"],
                                    bg=C["surface"])
        self.status_lbl.pack(side="right")

    def _build_disclaimer(self):
        bar = tk.Frame(self, bg=C["disc_bg"], pady=7)
        bar.pack(fill="x", padx=18, pady=(8, 0))
        tk.Label(bar,
                 text="⚠  MEDICAL DISCLAIMER: Algorithmic suggestions only — NOT a "
                      "substitute for professional medical advice. Always consult a "
                      "qualified physician before altering any prescribed medication.",
                 font=F["small"], fg=C["disc_fg"], bg=C["disc_bg"],
                 wraplength=1200, justify="left").pack(padx=12, anchor="w")

    # ── Left rail ─────────────────────────────────────────────────────

    def _build_rail(self, parent):
        self._cap(parent, "QUERY PARAMETERS", pady=(12, 6))

        self.nb = ttk.Notebook(parent, style="RX.TNotebook")
        self.nb.pack(fill="x")

        tab_a = tk.Frame(self.nb, bg=C["surface"], padx=14, pady=12)
        self.nb.add(tab_a, text="  Path A — Direct  ")
        self._build_tab_a(tab_a)

        tab_b = tk.Frame(self.nb, bg=C["surface"], padx=14, pady=12)
        self.nb.add(tab_b, text="  Path B — Symptoms  ")
        self._build_tab_b(tab_b)

        # Top-N slider
        ctrl = tk.Frame(parent, bg=C["bg"], pady=8)
        ctrl.pack(fill="x")
        self._cap(ctrl, "RESULTS TO SHOW")
        row = tk.Frame(ctrl, bg=C["bg"])
        row.pack(fill="x")
        self.top_n_var = tk.IntVar(value=5)
        self._n_lbl = tk.Label(row, text="5", font=F["score"],
                                fg=C["amber"], bg=C["bg"], width=3)
        self._n_lbl.pack(side="right")
        tk.Scale(row, from_=1, to=10, orient="horizontal",
                 variable=self.top_n_var, bg=C["bg"], fg=C["text_sub"],
                 highlightthickness=0, troughcolor=C["border"],
                 activebackground=C["amber"], sliderrelief="flat",
                 command=lambda v: self._n_lbl.config(text=str(int(float(v))))
                 ).pack(fill="x", expand=True, side="left")

        # Run button
        self.run_btn = tk.Button(
            parent, text="▶   Run Recommendation",
            font=F["h3"], fg=C["bg"], bg=C["amber"],
            activebackground=C["amber_light"], activeforeground=C["bg"],
            relief="flat", bd=0, pady=12, cursor="hand2",
            state="disabled", command=self._run_query)
        self.run_btn.pack(fill="x", pady=(10, 0))

        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=(14, 8))

        # MRU splay panel
        self._cap(parent, "SPLAY CACHE — RECENT QUERIES")
        self.mru_frame = tk.Frame(parent, bg=C["surface"], padx=10, pady=8)
        self.mru_frame.pack(fill="x", pady=(4, 0))
        self.mru_lbl = tk.Label(self.mru_frame, text="No queries yet.",
                                 font=F["small"], fg=C["text_muted"],
                                 bg=C["surface"], justify="left")
        self.mru_lbl.pack(anchor="w")

    def _build_tab_a(self, p):
        self._field_lbl(p, "DISEASE")
        self.disease_var = tk.StringVar()
        self.disease_cb = ttk.Combobox(p, textvariable=self.disease_var,
                                        font=F["body"], style="RX.TCombobox", values=[])
        self.disease_cb.pack(fill="x", pady=(2, 10))
        self.disease_cb.bind("<<ComboboxSelected>>", self._on_disease_selected)

        self._field_lbl(p, "BASELINE MEDICINE")
        self.medicine_var = tk.StringVar()
        self.medicine_cb = ttk.Combobox(p, textvariable=self.medicine_var,
                                         font=F["body"], style="RX.TCombobox", values=[])
        self.medicine_cb.pack(fill="x", pady=(2, 10))

        self._field_lbl(p, "PATIENT AGE GROUP")
        self.age_a = tk.StringVar(value="Adult")
        ttk.Combobox(p, textvariable=self.age_a, values=AGE_GROUPS,
                     state="readonly", font=F["body"],
                     style="RX.TCombobox").pack(fill="x", pady=(2, 4))

    def _build_tab_b(self, p):
        self._field_lbl(p, "DESCRIBE SYMPTOMS")
        self.sym_text = tk.Text(
            p, height=5, font=F["body"],
            bg=C["surface2"], fg=C["text_sub"],
            insertbackground=C["amber"],
            relief="flat", bd=6, wrap="word",
            highlightthickness=1,
            highlightcolor=C["border"],
            highlightbackground=C["border"])
        self.sym_text.pack(fill="x", pady=(2, 8))
        self._placeholder_text = "e.g. eye irritation, vomitting, urethra inflammation…"
        self.sym_text.insert("1.0", self._placeholder_text)
        self.sym_text.bind("<FocusIn>", self._clear_placeholder)
        self.sym_text.bind("<FocusOut>", self._restore_placeholder)

        self._field_lbl(p, "PATIENT AGE GROUP")
        self.age_b = tk.StringVar(value="Adult")
        ttk.Combobox(p, textvariable=self.age_b, values=AGE_GROUPS,
                     state="readonly", font=F["body"],
                     style="RX.TCombobox").pack(fill="x", pady=(2, 8))

        self.inferred_lbl = tk.Label(p, text="Inferred disease will appear here.",
                                      font=F["small"], fg=C["text_muted"],
                                      bg=C["surface"], wraplength=290, justify="left")
        self.inferred_lbl.pack(anchor="w")

    # ── Results panel ─────────────────────────────────────────────────

    def _build_results_panel(self, parent):
        hrow = tk.Frame(parent, bg=C["bg"])
        hrow.pack(fill="x", pady=(12, 6))
        self._cap(hrow, "RECOMMENDATIONS", side="left")
        self.result_meta = tk.Label(hrow, text="", font=F["small"],
                                     fg=C["text_muted"], bg=C["bg"])
        self.result_meta.pack(side="right")

        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=(0, 8))

        container = tk.Frame(parent, bg=C["bg"])
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, bg=C["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical",
                            command=self.canvas.yview, style="RX.Vertical.TScrollbar")
        self.scroll_frame = tk.Frame(self.canvas, bg=C["bg"])

        self.scroll_frame.bind("<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self._cwin = self.canvas.create_window((0, 0),
                                                window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.canvas.bind("<Configure>",
            lambda e: self.canvas.itemconfig(self._cwin, width=e.width))
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self.placeholder = tk.Label(
            self.scroll_frame,
            text="Results will appear here.\n\n"
                 "Path A — Find alternatives for a specific medicine.\n"
                 "Path B — Describe symptoms to get recommendations.",
            font=F["body"], fg=C["text_muted"], bg=C["bg"], justify="center")
        self.placeholder.pack(pady=80)

    # ── Engine loading ─────────────────────────────────────────────────

    def _load_engine(self):
        try:
            if not os.path.exists(DATASET_PATH):
                self._set_status("⚙  Generating dataset…", "amber")
                generate_dataset(DATASET_PATH, 2200)
            self._set_status("◌  Loading data structures…", "amber")
            eng = RecommendationEngine(t=3)
            eng.load_csv(DATASET_PATH)
            self.engine = eng
            self.engine_ready = True
            nd = len(eng.get_all_diseases())
            self._set_status(f"✓  Ready — {nd} diseases loaded", "emerald")
            self.after(0, self._on_engine_ready)
        except Exception as e:
            self._set_status(f"✗  Error: {e}", "rose")

    def _on_engine_ready(self):
        self.run_btn.config(state="normal")
        diseases = sorted(self.engine.get_all_diseases())
        self.disease_cb.config(values=diseases)

    def _set_status(self, text, tone="amber"):
        clr = {"amber": C["amber"], "emerald": C["emerald"],
                "rose": C["rose"]}.get(tone, C["text_sub"])
        self.after(0, lambda: self.status_lbl.config(text=text, fg=clr))

    def _on_disease_selected(self, _=None):
        if not self.engine_ready:
            return
        meds = self.engine.get_medicines_for_disease(self.disease_var.get().strip())
        self.medicine_cb.config(values=sorted(meds))
        self.medicine_var.set("")

    # ── Query dispatch ─────────────────────────────────────────────────

    def _run_query(self):
        if not self.engine_ready:
            messagebox.showwarning("Not Ready", "Engine is still loading.")
            return
        tab = self.nb.index(self.nb.select())
        top_n = self.top_n_var.get()
        if tab == 0:
            self._exec_path_a(top_n)
        else:
            self._exec_path_b(top_n)

    def _exec_path_a(self, top_n):
        disease = self.disease_var.get().strip()
        medicine = self.medicine_var.get().strip()
        age = self.age_a.get()
        if not disease or not medicine:
            messagebox.showwarning("Missing Input",
                                    "Please enter both Disease and Baseline Medicine.")
            return

        def _worker():
            res = self.engine.find_alternatives(disease, medicine, age, top_n)
            title = f"Alternatives for '{medicine}'  ·  {disease}  ·  {age}"
            self.after(0, lambda: self._display_results(res, title))
            self.after(0, self._update_mru)

        self._show_loading()
        threading.Thread(target=_worker, daemon=True).start()

    def _exec_path_b(self, top_n):
        symptoms = self.sym_text.get("1.0", "end-1c").strip()
        age = self.age_b.get()
        if not symptoms or symptoms == self._placeholder_text:
            messagebox.showwarning("Missing Input",
                                    "Please describe the patient's symptoms.")
            return

        def _worker():
            result = self.engine.predict_from_symptoms(symptoms, age, top_n)
            disease = result["inferred_disease"]
            self.after(0, lambda: self.inferred_lbl.config(
                text=f"Inferred Disease: {disease}", fg=C["emerald"]))
            title = f"Recommendations for '{disease}'  ·  {age}"
            self.after(0, lambda: self._display_results(result["recommendations"], title))
            self.after(0, self._update_mru)

        self._show_loading()
        threading.Thread(target=_worker, daemon=True).start()

    # ── Results rendering ─────────────────────────────────────────────

    def _show_loading(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        tk.Label(self.scroll_frame, text="Computing…",
                 font=F["h2"], fg=C["amber"], bg=C["bg"]).pack(pady=60)

    def _display_results(self, results: list, title: str):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.canvas.yview_moveto(0)

        if not results:
            tk.Label(self.scroll_frame,
                     text="No matching results.\n\nTry a different disease, "
                          "medicine, or age group.",
                     font=F["body"], fg=C["amber_dim"], bg=C["bg"],
                     justify="center").pack(pady=60)
            self.result_meta.config(text="0 results")
            return

        self.result_meta.config(
            text=f"{len(results)} result{'s' if len(results) != 1 else ''}")

        tk.Label(self.scroll_frame, text=title, font=F["h2"],
                 fg=C["text"], bg=C["bg"], anchor="w").pack(fill="x", pady=(0, 8))
        tk.Frame(self.scroll_frame, bg=C["border"], height=1).pack(
            fill="x", pady=(0, 8))

        for rank, item in enumerate(results, 1):
            self._build_card(self.scroll_frame, rank, item)
            tk.Frame(self.scroll_frame, bg=C["bg"], height=6).pack(fill="x")

    def _build_card(self, parent, rank: int, item: dict):
        med = item["medicine"]
        score = item["score"]

        card = tk.Frame(parent, bg=C["surface"], relief="flat")
        card.pack(fill="x", padx=2)

        accent = {1: C["amber"], 2: C["text_sub"], 3: C["amber_dim"]}.get(
            rank, C["border_hi"])
        tk.Frame(card, bg=accent, width=4).pack(side="left", fill="y")

        body = tk.Frame(card, bg=C["surface"], padx=16, pady=12)
        body.pack(side="left", fill="both", expand=True)

        # Top row: rank + name + availability
        top = tk.Frame(body, bg=C["surface"])
        top.pack(fill="x")

        sym = {1: "◆ #1", 2: "◇ #2", 3: "◇ #3"}.get(rank, f"  #{rank}")
        tk.Label(top, text=sym, font=F["small"],
                 fg=accent, bg=C["surface"]).pack(side="left", padx=(0, 10))
        tk.Label(top, text=med.name, font=F["h2"],
                 fg=C["text"], bg=C["surface"]).pack(side="left")

        avail_text = "● Available" if med.availability else "○ Unavailable"
        avail_fg = C["emerald"] if med.availability else C["rose"]
        tk.Label(top, text=avail_text, font=F["small"],
                 fg=avail_fg, bg=C["surface"]).pack(side="right")

        # Score bar
        srow = tk.Frame(body, bg=C["surface"])
        srow.pack(fill="x", pady=(6, 0))
        tk.Label(srow, text=f"Score  {score:.4f}",
                 font=F["mono"], fg=C["amber"], bg=C["surface"]).pack(side="left")

        bar_bg = tk.Frame(srow, bg=C["border"], height=6, width=220)
        bar_bg.pack(side="left", padx=(10, 0), pady=3)
        bar_bg.pack_propagate(False)
        fill_w = max(2, int(score * 220))
        bar_clr = (C["emerald"] if score >= 0.60
                   else C["amber"] if score >= 0.35
                   else C["rose"])
        tk.Frame(bar_bg, bg=bar_clr, width=fill_w, height=6).pack(side="left")

        # Divider
        tk.Frame(body, bg=C["border"], height=1).pack(fill="x", pady=(8, 8))

        # Details row
        details = tk.Frame(body, bg=C["surface"])
        details.pack(fill="x")

        # Left: composition
        comp_col = tk.Frame(details, bg=C["surface"])
        comp_col.pack(side="left", fill="both", expand=True)
        tk.Label(comp_col, text="COMPOSITION", font=F["small"],
                 fg=C["text_muted"], bg=C["surface"]).pack(anchor="w")

        for c in med.composition:
            mg_val = c["mg"]
            if isinstance(mg_val, float) and mg_val >= 1 and mg_val == int(mg_val):
                mg_str = f"{int(mg_val)}mg"
            else:
                mg_str = f"{mg_val:g}mg"
            tk.Label(comp_col,
                     text=f"  {c['ingredient']}  {mg_str}  ({c['percentage']}%)",
                     font=F["mono"], fg=C["text_sub"],
                     bg=C["surface"]).pack(anchor="w")

        # Right: price, effectiveness, badges
        right = tk.Frame(details, bg=C["surface"])
        right.pack(side="right", padx=(16, 0), anchor="ne")

        tk.Label(right, text=f"₹ {med.price:,.2f}",
                 font=F["price"], fg=C["amber_light"],
                 bg=C["surface"]).pack(anchor="e")

        eff_pct = int(med.effectiveness_score * 100)
        eff_row = tk.Frame(right, bg=C["surface"])
        eff_row.pack(anchor="e", pady=(2, 0))
        tk.Label(eff_row, text="Effectiveness", font=F["small"],
                 fg=C["text_muted"], bg=C["surface"]).pack(side="left")
        tk.Label(eff_row,
                 text=f"  {eff_pct}%",
                 font=F["small"],
                 fg=C["emerald"] if eff_pct >= 75 else C["amber"],
                 bg=C["surface"]).pack(side="left")

        badge_row = tk.Frame(right, bg=C["surface"])
        badge_row.pack(anchor="e", pady=(6, 0))
        badge_cfg = {
            "Child":  (C["lavender_dim"], C["lavender"]),
            "Adult":  (C["emerald_dim"],  C["emerald"]),
            "Senior": (C["amber_dim"],    C["amber"]),
        }
        for grp in ["Child", "Adult", "Senior"]:
            if grp in med.suitable_for:
                bg, fg = badge_cfg[grp]
                tk.Label(badge_row, text=f"  {grp}  ",
                         font=F["badge"], fg=fg, bg=bg,
                         pady=3).pack(side="left", padx=2)

        tk.Label(right, text=med.disease_target, font=F["small"],
                 fg=C["text_muted"], bg=C["surface"]).pack(anchor="e", pady=(4, 0))

    # ── MRU splay refresh ─────────────────────────────────────────────

    def _update_mru(self):
        if not self.engine:
            return
        top = self.engine.get_mru_diseases(6)
        if not top:
            return
        self.mru_lbl.config(
            text="\n".join(f"  {i+1}.  {d}  ({c}×)" for i, (d, c) in enumerate(top)),
            fg=C["text_sub"])

    # ── Widget helpers ────────────────────────────────────────────────

    def _cap(self, parent, text, side="left", pady=(0, 4)):
        tk.Label(parent, text=text, font=F["small"],
                 fg=C["text_muted"], bg=C["bg"]).pack(anchor="w", pady=pady)

    def _field_lbl(self, parent, text):
        tk.Label(parent, text=text, font=F["small"],
                 fg=C["text_muted"], bg=parent["bg"]).pack(anchor="w")

    def _clear_placeholder(self, _):
        current = self.sym_text.get("1.0", "end-1c")
        if current == self._placeholder_text:
            self.sym_text.delete("1.0", "end")
            self.sym_text.config(fg=C["text"])

    def _restore_placeholder(self, _):
        if not self.sym_text.get("1.0", "end-1c").strip():
            self.sym_text.config(fg=C["text_sub"])
            self.sym_text.insert("1.0", self._placeholder_text)


if __name__ == "__main__":
    app = DrugRecommenderApp()
    app.mainloop()
