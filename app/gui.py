import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master, padding=12)
        self.master = master
        self.controller = controller
        self.pack(fill="both", expand=True)

        self.task_var = tk.StringVar(value="Text Classification")
        self.input_type = tk.StringVar(value="Text")
        self.image_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self._build_menu()
        self._build_layout()
        self._update_model_info()

    # ---------------- Menu ----------------
    def _build_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About", "HIT137 Assignment 3 — Tkinter + Hugging Face"
            ),
        )
        menubar.add_cascade(label="Help", menu=help_menu)

    # ------------- Layout (matches brief) -------------
    def _build_layout(self):
        # Top row: Model Selection + Load + Clear
        row = ttk.Frame(self); row.pack(fill="x", pady=(0,8))
        ttk.Label(row, text="Model Selection:").pack(side="left")
        task_box = ttk.Combobox(
            row, textvariable=self.task_var, state="readonly",
            values=["Text Classification", "Image Classification"], width=22
        )
        task_box.pack(side="left", padx=8)
        task_box.bind("<<ComboboxSelected>>", lambda e: self._on_task_change())
        ttk.Button(row, text="Load Model", command=self._on_load_model).pack(side="left", padx=(6,0))
        ttk.Button(row, text="Clear", command=self._on_clear).pack(side="right")

        # Middle: two columns
        middle = ttk.Frame(self); middle.pack(fill="both", expand=True)

        # Left: User Input Section
        left = ttk.LabelFrame(middle, text="User Input Section", padding=8)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))

        radios = ttk.Frame(left); radios.pack(anchor="w", pady=(0,6))
        ttk.Radiobutton(radios, text="Text", value="Text", variable=self.input_type,
                        command=self._sync_input_controls).pack(side="left")
        ttk.Radiobutton(radios, text="Image", value="Image", variable=self.input_type,
                        command=self._sync_input_controls).pack(side="left", padx=(10,0))

        self.text_input = tk.Text(left, height=6)
        self.text_input.pack(fill="x", pady=(0,6))

        img_row = ttk.Frame(left); img_row.pack(fill="x")
        ttk.Label(img_row, text="Image Path:").pack(side="left")
        self.image_entry = ttk.Entry(img_row, textvariable=self.image_path_var, width=50)
        self.image_entry.pack(side="left", padx=6)
        ttk.Button(img_row, text="Browse", command=self._on_browse_image).pack(side="left")

        run_row = ttk.Frame(left); run_row.pack(anchor="w", pady=(8,0))
        ttk.Button(run_row, text="Run Model 1 (Text)", command=self._run_text).pack(side="left")
        ttk.Button(run_row, text="Run Model 2 (Image)", command=self._run_image).pack(side="left", padx=8)

        # Right: Model Output Section
        right = ttk.LabelFrame(middle, text="Model Output Section", padding=8)
        right.pack(side="left", fill="both", expand=True, padx=(6,0))
        self.output_box = tk.Text(right, height=14)
        self.output_box.pack(fill="both", expand=True)

        # Bottom: Model Information & Explanation
        info = ttk.LabelFrame(self, text="Model Information & Explanation", padding=8)
        info.pack(fill="x", pady=(8,0))

        left_info = ttk.Frame(info); left_info.pack(side="left", fill="x", expand=True, padx=(0,6))
        right_info = ttk.Frame(info); right_info.pack(side="left", fill="x", expand=True, padx=(6,0))

        ttk.Label(left_info, text="Selected Model Info", font=("", 10, "bold")).pack(anchor="w")
        self.info_name = ttk.Label(left_info, text="Model: -"); self.info_name.pack(anchor="w")
        self.info_cat  = ttk.Label(left_info, text="Category: -"); self.info_cat.pack(anchor="w")
        self.info_desc = ttk.Label(left_info, text="Description: -", wraplength=420, justify="left")
        self.info_desc.pack(anchor="w", pady=(0,4))

        ttk.Label(right_info, text="OOP Concepts Explanation", font=("", 10, "bold")).pack(anchor="w")
        oop_text = (
            "• Multiple Inheritance: ModelInfoMixin + ModelBase\n"
            "• Encapsulation: _pipe, _name, _task\n"
            "• Polymorphism: all models expose run(input)\n"
            "• Method Overriding: preprocess/postprocess hooks\n"
            "• Multiple Decorators: @timed and @log_call on run()"
        )
        self.oop_label = ttk.Label(right_info, text=oop_text, justify="left", wraplength=420)
        self.oop_label.pack(anchor="w")

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(6,0))
        self._sync_input_controls()

    # ---------------- Actions ----------------
    def _on_task_change(self):
        self.input_type.set("Text" if self.task_var.get() == "Text Classification" else "Image")
        self._sync_input_controls()
        self._update_model_info()
        self._hint(f"Selected: {self.task_var.get()}")

    def _on_load_model(self):
        self._update_model_info()
        self._hint("Model info loaded.")

    def _on_browse_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if path: self.image_path_var.set(path)

    def _run_text(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Input required", "Please type some text."); return
        self._hint("Running Text model…")
        try:
            result = self.controller.run_text(text)
            self._display_output(result); self._hint("Done.")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self._hint("Error occurred.")

    def _run_image(self):
        path = self.image_path_var.get().strip()
        if not path:
            messagebox.showwarning("Input required", "Please select an image file."); return
        self._hint("Running Image model…")
        try:
            result = self.controller.run_image(path)
            self._display_output(result); self._hint("Done.")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self._hint("Error occurred.")

   def _display_output(self, result_obj):
        self.output_box.delete("1.0", "end")
        result = result_obj.get("result")
        elapsed = result_obj.get("elapsed_ms", 0)

        lines = ["Result:"]
        if isinstance(result, list) and result and isinstance(result[0], dict):
            for r in result[:5]:
                lbl = r.get("label", "-")
                scr = r.get("score", 0.0)
                lines.append(f"{lbl}: {scr:.4f}")
        elif isinstance(result, dict) and "label" in result:
            lines.append(f"{result.get('label','-')}: {result.get('score',0.0):.4f}")
        else:
            lines.append(str(result))

        lines.append("")
        lines.append(f"Elapsed: {elapsed} ms")
        self.output_box.insert("1.0", "\n".join(lines))



    def _update_model_info(self):
        info = self.controller.model_info(self.task_var.get())
        self.info_name.config(text=f"Model: {info.get('name', '-')}")
        self.info_cat.config(text=f"Category: {info.get('category', '-')}")
        self.info_desc.config(text=f"Description: {info.get('description', '-')}")

    def _sync_input_controls(self):
        is_text = self.input_type.get() == "Text"
        self.text_input.config(state="normal" if is_text else "disabled")
        self.image_entry.config(state="disabled" if is_text else "normal")

    def _on_clear(self):
        self.text_input.config(state="normal")
        self.text_input.delete("1.0", "end")
        self.image_path_var.set("")
        self.output_box.delete("1.0", "end")
        self._hint("Cleared.")

    def _hint(self, msg): self.status_var.set(msg)
