import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master, padding=12)
        self.master = master
        self.controller = controller
        self.pack(fill="both", expand=True)
        self._build_menu()
        self._build_layout()

    def _build_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Model Info", command=self._show_model_info)
        help_menu.add_command(label="OOP Explanations", command=self._show_oop_info)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "HIT137 Assignment 3 — Tkinter + Hugging Face"))
        menubar.add_cascade(label="Help", menu=help_menu)

    def _build_layout(self):
        top = ttk.Frame(self); top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Select Task:").pack(side="left")
        self.task_var = tk.StringVar(value="Text Classification")
        task_box = ttk.Combobox(
            top, textvariable=self.task_var, state="readonly",
            values=["Text Classification", "Image Classification"]
        )
        task_box.pack(side="left", padx=8)
        ttk.Button(top, text="Clear", command=self._on_clear).pack(side="right")

        self.input_frame = ttk.LabelFrame(self, text="Input"); self.input_frame.pack(fill="x", pady=4)
        self.text_input = tk.Text(self.input_frame, height=5); self.text_input.pack(fill="x", padx=8, pady=8)

        img_row = ttk.Frame(self.input_frame); img_row.pack(fill="x", padx=8, pady=(0,8))
        ttk.Label(img_row, text="Image Path:").pack(side="left")
        self.image_path_var = tk.StringVar()
        self.image_entry = ttk.Entry(img_row, textvariable=self.image_path_var, width=60); self.image_entry.pack(side="left", padx=6)
        ttk.Button(img_row, text="Browse…", command=self._on_browse_image).pack(side="left")

        ttk.Button(self, text="Run Selected Model", command=self._on_run).pack(pady=6)

        self.output_frame = ttk.LabelFrame(self, text="Output"); self.output_frame.pack(fill="both", expand=True, pady=4)
        self.output_box = tk.Text(self.output_frame, height=14); self.output_box.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(6,0))
        self._hint("Tip: For Text Classification, type text above. For Image Classification, browse to an image file.")

    def _on_browse_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if path: self.image_path_var.set(path)

    def _on_run(self):
        task = self.task_var.get()
        self._hint(f"Running: {task}…")
        try:
            if task == "Text Classification":
                text = self.text_input.get("1.0", "end").strip()
                if not text:
                    messagebox.showwarning("Input required", "Please type some text."); return
                result = self.controller.run_text(text)
            else:
                path = self.image_path_var.get().strip()
                if not path:
                    messagebox.showwarning("Input required", "Please select an image file."); return
                result = self.controller.run_image(path)
            self._display_output(result); self._hint("Done.")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self._hint("Error occurred.")

    def _display_output(self, result_obj):
        self.output_box.delete("1.0", "end")
        if isinstance(result_obj, dict):
            self.output_box.insert("1.0", f"Result:\n{result_obj.get('result')}\n\nElapsed: {result_obj.get('elapsed_ms')} ms")
        else:
            self.output_box.insert("1.0", str(result_obj))

    def _on_clear(self):
        self.text_input.delete("1.0", "end"); self.image_path_var.set(""); self.output_box.delete("1.0", "end")
        self._hint("Cleared.")

    def _show_model_info(self):
        msg = ("Selected Models (to be filled by Members 2 & 3):\n\n"
               "• Text: DistilBERT SST-2 (Sentiment Analysis)\n"
               "• Image: ViT Base Patch16-224 (Image Classification)\n\n"
               "Each integrated via Hugging Face Transformers.")
        messagebox.showinfo("Model Info", msg)

    def _show_oop_info(self):
        msg = ("OOP Concepts (explained in code & docs):\n"
               "• Multiple Inheritance\n• Encapsulation (private attrs)\n"
               "• Polymorphism (uniform run(input))\n"
               "• Method Overriding (child run/pre/post)\n"
               "• Multiple Decorators (@timed, @log_call)\n")
        messagebox.showinfo("OOP Explanations", msg)

    def _hint(self, text): self.status_var.set(text)
