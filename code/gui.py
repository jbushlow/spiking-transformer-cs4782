# run with: python gui.py

import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from spikingjelly.activation_based import functional

from config import GPTConfig
from model import SpikingGPT
from generate import encode, decode, load_model, generate


BG       = "#1e1e2e"
SURFACE  = "#2a2a3e"
ACCENT   = "#7c3aed"
ACCENT_H = "#9f5bf5"
FG       = "#e2e8f0"
FG_DIM   = "#94a3b8"
SUCCESS  = "#22c55e"
ERROR    = "#ef4444"
MONO     = ("Consolas", 11)
SANS     = ("Segoe UI", 10)
SANS_B   = ("Segoe UI", 10, "bold")


class SpikeGPTGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SpikeGPT — Text Generator")
        self.root.configure(bg=BG)
        self.root.minsize(860, 620)

        self.model: SpikingGPT | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token_queue: queue.Queue = queue.Queue()
        self._generating = False

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG, pady=8)
        top.pack(fill=tk.X, padx=16)

        tk.Label(top, text="SpikeGPT", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(side=tk.LEFT)
        self.device_label = tk.Label(
            top, text=f"device: {self.device}", font=SANS,
            bg=BG, fg=FG_DIM)
        self.device_label.pack(side=tk.RIGHT)

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG,
                               sashwidth=6, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left  = tk.Frame(paned, bg=BG)
        right = tk.Frame(paned, bg=BG)
        paned.add(left,  minsize=280, width=320)
        paned.add(right, minsize=400)

        self._build_controls(left)
        self._build_output(right)

    def _section(self, parent, title):
        frame = tk.LabelFrame(parent, text=f"  {title}  ", bg=SURFACE, fg=FG_DIM,
                              font=SANS, bd=1, relief=tk.FLAT,
                              highlightbackground=ACCENT, highlightthickness=1)
        frame.pack(fill=tk.X, padx=8, pady=6)
        return frame

    def _row(self, parent, label, widget_factory, pady=4):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, padx=10, pady=pady)
        tk.Label(row, text=label, width=14, anchor=tk.W,
                 bg=SURFACE, fg=FG, font=SANS).pack(side=tk.LEFT)
        w = widget_factory(row)
        w.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return w

    def _build_controls(self, parent):
        ckpt_frame = self._section(parent, "Checkpoint")
        ckpt_row = tk.Frame(ckpt_frame, bg=SURFACE)
        ckpt_row.pack(fill=tk.X, padx=10, pady=6)

        self.ckpt_var = tk.StringVar()
        ckpt_entry = tk.Entry(ckpt_row, textvariable=self.ckpt_var,
                              bg="#12121f", fg=FG, insertbackground=FG,
                              relief=tk.FLAT, font=MONO)
        ckpt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        tk.Button(ckpt_row, text="Browse", font=SANS,
                  bg=ACCENT, fg="white", relief=tk.FLAT, cursor="hand2",
                  command=self._browse_checkpoint,
                  activebackground=ACCENT_H).pack(side=tk.LEFT, padx=(6, 0))

        cfg_frame = self._section(parent, "Model Config (match training)")

        def spin(parent, from_, to, default):
            v = tk.IntVar(value=default)
            w = ttk.Spinbox(parent, from_=from_, to=to, textvariable=v,
                            width=8, font=MONO)
            w.configure()
            return w, v

        cfg_row1 = tk.Frame(cfg_frame, bg=SURFACE)
        cfg_row1.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(cfg_row1, text="ctx_len", width=9, anchor=tk.W,
                 bg=SURFACE, fg=FG, font=SANS).pack(side=tk.LEFT)
        self._ctx_spin, self.ctx_var = spin(cfg_row1, 32, 4096, 128)
        self._ctx_spin.pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(cfg_row1, text="n_embd", anchor=tk.W,
                 bg=SURFACE, fg=FG, font=SANS).pack(side=tk.LEFT)
        self._embd_spin, self.embd_var = spin(cfg_row1, 64, 1024, 128)
        self._embd_spin.pack(side=tk.LEFT)

        cfg_row2 = tk.Frame(cfg_frame, bg=SURFACE)
        cfg_row2.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Label(cfg_row2, text="n_layer", width=9, anchor=tk.W,
                 bg=SURFACE, fg=FG, font=SANS).pack(side=tk.LEFT)
        self._layer_spin, self.layer_var = spin(cfg_row2, 1, 48, 2)
        self._layer_spin.pack(side=tk.LEFT)

        self.load_btn = tk.Button(cfg_frame, text="Load Model", font=SANS_B,
                                  bg=ACCENT, fg="white", relief=tk.FLAT,
                                  cursor="hand2", pady=5,
                                  command=self._load_model,
                                  activebackground=ACCENT_H)
        self.load_btn.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.model_status = tk.Label(cfg_frame, text="No model loaded",
                                     bg=SURFACE, fg=FG_DIM, font=SANS)
        self.model_status.pack(padx=10, pady=(0, 6))

        gen_frame = self._section(parent, "Generation")

        def slider_row(label, from_, to, default, res=0.05, fmt="{:.2f}"):
            row = tk.Frame(gen_frame, bg=SURFACE)
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text=label, width=16, anchor=tk.W,
                     bg=SURFACE, fg=FG, font=SANS).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            val_lbl = tk.Label(row, text=fmt.format(default), width=6,
                               bg=SURFACE, fg=ACCENT, font=MONO)
            val_lbl.pack(side=tk.RIGHT)
            def update(v, lbl=val_lbl, f=fmt, sv=var):
                lbl.config(text=f.format(float(v)))
            sl = tk.Scale(row, variable=var, from_=from_, to=to,
                          resolution=res, orient=tk.HORIZONTAL,
                          bg=SURFACE, fg=FG, troughcolor=BG,
                          highlightthickness=0, showvalue=False,
                          command=update)
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return var

        self.temp_var    = slider_row("Temperature",    0.1, 2.0, 1.0)
        self.topk_var    = slider_row("Top-k",          0,   256, 40, res=1, fmt="{:.0f}")
        self.max_tok_var = slider_row("Max new tokens", 10,  500, 200, res=10, fmt="{:.0f}")

        greedy_row = tk.Frame(gen_frame, bg=SURFACE)
        greedy_row.pack(fill=tk.X, padx=10, pady=(2, 8))
        self.greedy_var = tk.BooleanVar(value=False)
        tk.Checkbutton(greedy_row, text="Greedy decoding (ignore temperature / top-k)",
                       variable=self.greedy_var, bg=SURFACE, fg=FG,
                       selectcolor=BG, activebackground=SURFACE,
                       font=SANS).pack(side=tk.LEFT)

    def _build_output(self, parent):
        prompt_frame = tk.Frame(parent, bg=BG)
        prompt_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(prompt_frame, text="Prompt", bg=BG, fg=FG_DIM,
                 font=SANS).pack(anchor=tk.W)
        self.prompt_text = tk.Text(prompt_frame, height=4,
                                   bg=SURFACE, fg=FG, insertbackground=FG,
                                   font=MONO, relief=tk.FLAT, wrap=tk.WORD,
                                   padx=8, pady=6)
        self.prompt_text.pack(fill=tk.X, pady=(2, 0))
        self.prompt_text.insert("1.0", " ")

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill=tk.X, padx=8, pady=4)

        self.gen_btn = tk.Button(btn_row, text="Generate", font=SANS_B,
                                 bg=SUCCESS, fg="white", relief=tk.FLAT,
                                 cursor="hand2", pady=6,
                                 command=self._start_generation,
                                 activebackground="#16a34a")
        self.gen_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.stop_btn = tk.Button(btn_row, text="Stop", font=SANS_B,
                                  bg=ERROR, fg="white", relief=tk.FLAT,
                                  cursor="hand2", pady=6,
                                  command=self._stop_generation,
                                  activebackground="#dc2626",
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        tk.Button(btn_row, text="Clear", font=SANS,
                  bg=SURFACE, fg=FG_DIM, relief=tk.FLAT,
                  cursor="hand2", pady=6,
                  command=self._clear_output,
                  activebackground=BG).pack(side=tk.LEFT, padx=(6, 0))

        out_frame = tk.Frame(parent, bg=BG)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        tk.Label(out_frame, text="Output", bg=BG, fg=FG_DIM,
                 font=SANS).pack(anchor=tk.W)

        self.output_text = tk.Text(out_frame, bg=SURFACE, fg=FG,
                                   insertbackground=FG, font=MONO,
                                   relief=tk.FLAT, wrap=tk.WORD,
                                   padx=8, pady=6, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        scrollbar = tk.Scrollbar(out_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(parent, textvariable=self.status_var,
                 bg=BG, fg=FG_DIM, font=SANS, anchor=tk.W).pack(
            fill=tk.X, padx=8, pady=(0, 4))

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _load_model(self):
        path = self.ckpt_var.get().strip()
        if not path:
            messagebox.showerror("Error", "Please specify a checkpoint path.")
            return
        if not Path(path).exists():
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        self.model_status.config(text="Loading…", fg=FG_DIM)
        self.root.update_idletasks()

        try:
            config = GPTConfig(
                ctx_len=self.ctx_var.get(),
                n_embd=self.embd_var.get(),
                n_layer=self.layer_var.get(),
            )
            self.model = load_model(path, config, self.device)
            epoch_info = torch.load(path, map_location="cpu").get("epoch", "?")
            self.model_status.config(
                text=f"Loaded  epoch={epoch_info}  ({self.device})", fg=SUCCESS)
            self.status_var.set("Model ready.")
        except Exception as exc:
            self.model_status.config(text=f"Error: {exc}", fg=ERROR)
            self.model = None

    def _start_generation(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model checkpoint first.")
            return
        if self._generating:
            return

        prompt = self.prompt_text.get("1.0", tk.END).rstrip("\n")
        prompt_ids = encode(prompt)

        max_new = int(self.max_tok_var.get())
        temperature = float(self.temp_var.get())
        top_k = int(self.topk_var.get())
        greedy = self.greedy_var.get()

        self._generating = True
        self._stop_flag = False
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Generating…")

        self._set_output(prompt)

        def run():
            try:
                if greedy:
                    model = self.model
                    model.eval()
                    ctx_len = model.ctx_len
                    tokens = list(prompt_ids)
                    with torch.no_grad():
                        for _ in range(max_new):
                            if self._stop_flag:
                                break
                            context = tokens[-ctx_len:]
                            idx = torch.tensor([context], dtype=torch.long,
                                               device=self.device)
                            logits = model(idx)
                            functional.reset_net(model)
                            next_token = logits[0, -1, :].argmax().item()
                            tokens.append(next_token)
                            self.token_queue.put(decode([next_token]))
                else:
                    def cb(ch):
                        if self._stop_flag:
                            raise StopIteration
                        self.token_queue.put(ch)
                    try:
                        generate(
                            self.model, prompt_ids, max_new,
                            temperature=temperature,
                            top_k=top_k,
                            device=self.device,
                            stream_callback=cb,
                        )
                    except StopIteration:
                        pass
            except Exception as exc:
                self.token_queue.put(f"\n[Error: {exc}]")
            finally:
                self.token_queue.put(None)  # None signals we're done

        threading.Thread(target=run, daemon=True).start()

    def _stop_generation(self):
        self._stop_flag = True

    def _clear_output(self):
        self._set_output("")

    def _set_output(self, text: str):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.config(state=tk.DISABLED)

    def _on_generation_done(self):
        self._generating = False
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        status = "Stopped." if self._stop_flag else "Done."
        self.status_var.set(status)

    def _poll_queue(self):
        # called every 30ms to pull tokens off the queue and append them to the output box
        try:
            while True:
                item = self.token_queue.get_nowait()
                if item is None:
                    self._on_generation_done()
                else:
                    self.output_text.config(state=tk.NORMAL)
                    self.output_text.insert(tk.END, item)
                    self.output_text.see(tk.END)
                    self.output_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)


def main():
    root = tk.Tk()
    app = SpikeGPTGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
