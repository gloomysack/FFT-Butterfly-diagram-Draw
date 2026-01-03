import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class FFTButterflyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Draw（made by Hz）")

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Sequence In:").grid(row=0, column=0)
        self.entry_seq = tk.Entry(control_frame, width=30)
        self.entry_seq.insert(0, "1, 2, 3, 4")
        self.entry_seq.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="N:").grid(row=0, column=2)
        self.entry_n = tk.Entry(control_frame, width=5)
        self.entry_n.insert(0, "4")
        self.entry_n.grid(row=0, column=3, padx=5)

        self.mode_var = tk.StringVar(value="DIT")
        tk.Radiobutton(control_frame, text="DIT", variable=self.mode_var, value="DIT").grid(row=0, column=4)
        tk.Radiobutton(control_frame, text="DIF", variable=self.mode_var, value="DIF").grid(row=0, column=5)

        btn_draw = tk.Button(control_frame, text="draw", command=self.draw_fft, bg="lightblue")
        btn_draw.grid(row=0, column=6, padx=10)

        # 绘图区域
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def bit_reverse(self, n, bits):
        return int(format(n, '0' + str(bits) + 'b')[::-1], 2)

    def draw_fft(self):
        try:
            raw_seq = [complex(x.strip()) for x in self.entry_seq.get().split(",")]
            n_input = int(self.entry_n.get())
            mode = self.mode_var.get()

            # 补零处理
            if len(raw_seq) < n_input:
                raw_seq += [0] * (n_input - len(raw_seq))
            raw_seq = raw_seq[:n_input]

            m = int(np.log2(n_input))
            if 2 ** m != n_input:
                messagebox.showerror("错误", "点数N必须为2的幂次")
                return

            self.fig.clear()
            ax = self.fig.add_subplot(111)

            nodes_val = np.zeros((m + 1, n_input), dtype=complex)

            if mode == "DIT":
                # DIT 输入倒序
                input_indices = [self.bit_reverse(i, m) for i in range(n_input)]
                for i, idx in enumerate(input_indices):
                    nodes_val[0, i] = raw_seq[idx]
                input_labels = [f"x({idx})={raw_seq[idx].real:.0f}" for idx in input_indices]
            else:
                # DIF 输入正序
                for i in range(n_input):
                    nodes_val[0, i] = raw_seq[i]
                input_labels = [f"x({i})={raw_seq[i].real:.0f}" for i in range(n_input)]

            temp_data = nodes_val[0, :].copy()
            for s in range(1, m + 1):
                if mode == "DIT":
                    span = 2 ** (s - 1)
                    group_size = 2 ** s
                else:  # DIF
                    span = 2 ** (m - s)
                    group_size = 2 ** (m - s + 1)

                for g in range(0, n_input, group_size):
                    for j in range(span):
                        idx1 = g + j
                        idx2 = g + j + span

                        if mode == "DIT":
                            w_idx = j * (n_input // group_size)
                            w = np.exp(-2j * np.pi * w_idx / n_input)
                            val1 = temp_data[idx1]
                            val2 = temp_data[idx2] * w
                            temp_data[idx1] = val1 + val2
                            temp_data[idx2] = val1 - val2
                        # DIF
                        else:
                            w_idx = j * (n_input // group_size)
                            w = np.exp(-2j * np.pi * w_idx / n_input)
                            val1 = temp_data[idx1]
                            val2 = temp_data[idx2]
                            temp_data[idx1] = val1 + val2
                            temp_data[idx2] = (val1 - val2) * w
                nodes_val[s, :] = temp_data

            for s in range(m + 1):
                # 画节点列
                x = s * 2
                for i in range(n_input):
                    ax.plot(x, -i, 'ko', markersize=8, zorder=3)
                    val_str = f"{nodes_val[s, i].real:.1f}" + (
                        f"{nodes_val[s, i].imag:+.1f}j" if abs(nodes_val[s, i].imag) > 0.01 else "")
                    ax.text(x, -i + 0.2, val_str, ha='center', fontsize=8, color='blue')

                    # 第一列和最后一列写标签
                    if s == 0:
                        ax.text(x - 0.5, -i, input_labels[i], ha='right')
                    if s == m:
                        out_idx = i if mode == "DIT" else self.bit_reverse(i, m)
                        ax.text(x + 0.5, -i, f"X({out_idx})", ha='left', fontweight='bold')

                # 画级间连线
                if s < m:
                    if mode == "DIT":
                        span = 2 ** s
                        group_size = 2 ** (s + 1)
                    else:
                        span = 2 ** (m - s - 1)
                        group_size = 2 ** (m - s)

                    for g in range(0, n_input, group_size):
                        for j in range(span):
                            idx1 = g + j
                            idx2 = g + j + span

                            # 获取旋转因子用于标注
                            w_idx = j * (n_input // (span * 2))
                            w_label = f"$W_{{{n_input}}}^{{{w_idx}}}$"

                            # 上支路加法
                            ax.annotate("", xy=(x + 2, -idx1), xytext=(x, -idx1), arrowprops=dict(arrowstyle="->"))
                            ax.annotate("", xy=(x + 2, -idx1), xytext=(x, -idx2), arrowprops=dict(arrowstyle="->"))
                            # 下支路减法
                            ax.annotate("", xy=(x + 2, -idx2), xytext=(x, -idx1), arrowprops=dict(arrowstyle="->"))
                            ax.annotate("", xy=(x + 2, -idx2), xytext=(x, -idx2), arrowprops=dict(arrowstyle="->"))

                            # 标注 -1 和 W
                            ax.text(x + 1.7, -idx2 - 0.1, "-1", fontsize=9)
                            if mode == "DIT":
                                ax.text(x + 0.3, -idx2 + 0.1, w_label, color='red', fontsize=9)
                            else:
                                ax.text(x + 1.5, -idx2 + 0.1, w_label, color='red', fontsize=9)

            ax.axis('off')
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("执行错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FFTButterflyApp(root)
    root.mainloop()
