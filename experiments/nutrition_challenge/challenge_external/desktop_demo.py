"""Offline display for the OmniFood baselines or the NutritionVerse plate model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = {"CLIP + LoRA": "clip", "ResNet18": "resnet18", "训练集均值": "mean"}
FIELDS = {
    "total_weight_g": "总重量",
    "total_calories_kcal": "热量",
    "protein_g": "蛋白质",
    "fat_g": "脂肪",
    "carbohydrate_g": "碳水化合物",
}


class Predictor:
    def __init__(self, nutritionverse=False):
        self.session = None
        self.nutritionverse = nutritionverse

    def run(self, path):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        if self.session is None:
            if self.nutritionverse:
                from challenge_external.nutritionverse_model import NutritionVerseSession
                self.session = NutritionVerseSession()
            else:
                from challenge_external.display_session import IntervalSession
                self.session = IntervalSession()
        if self.nutritionverse:
            outputs = {"adapted": self.session.predict_rgb(rgb)}
        else:
            outputs = self.session.predict_all(path)
        rgb.thumbnail((270, 270), Image.Resampling.LANCZOS)
        return rgb, outputs


class NutritionWindow:
    def __init__(self, root, nutritionverse=False):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.predictor = Predictor(nutritionverse)
        self.outputs = {}
        self.image = None
        self.future = None
        self.path = None
        self.coverage_version = None
        if nutritionverse:
            metrics = RESULTS.parent / "nutritionverse_extension/metrics.json"
            release = json.loads((metrics.parent / "inference_release.json").read_text(encoding="utf8"))
            self.coverage_version = "nv90-" + release["evaluated_calibration_sha256"][:12]
            example = RESULTS.parent / "data_supplement_v1/nutritionverse/images/dish_36.jpg"
            self.models = {"餐盘适配模型": "adapted"}
            domain_note = "覆盖率来自 NutritionVerse 的 67 道留出菜品。"
        else:
            metrics = RESULTS / "metrics.json"
            example = RESULTS / "interface_sample.jpg"
            self.models = MODELS
            domain_note = "覆盖率来自 OmniFood8K 的 2,368 个留出场景。"
        rows = json.loads(metrics.read_text(encoding="utf8"))
        self.coverage = {(row["model"], row["field"]): row["coverage"] for row in rows}

        root.title("FoodLens · 单图营养估计")
        root.minsize(900, 440)
        root.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(root, padding=20)
        body.pack(fill="both", expand=True)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        self.open_button = ttk.Button(toolbar, text="选择食物图片", command=self.choose)
        self.open_button.pack(side="left")
        self.demo_button = ttk.Button(toolbar, text="打开示例", command=lambda: self.load(example))
        self.demo_button.pack(side="left", padx=6)
        self.model = tk.StringVar(value=next(iter(self.models)))
        picker = ttk.Combobox(toolbar, values=list(self.models), textvariable=self.model, state="readonly", width=17)
        picker.pack(side="left", padx=10)
        picker.bind("<<ComboboxSelected>>", lambda _: self.render())
        self.export_button = ttk.Button(toolbar, text="导出预测 JSON", command=self.export, state="disabled")
        self.export_button.pack(side="right")

        main = ttk.Frame(body)
        main.pack(fill="both", expand=True, pady=18)
        self.preview = ttk.Label(main, text="选择一张 RGB 食物图片", anchor="center", width=32)
        self.preview.pack(side="left", padx=(0, 14), fill="both")
        columns = ("field", "point", "bounds", "width", "coverage")
        self.table = ttk.Treeview(main, columns=columns, show="headings", height=6)
        for key, heading, width in zip(columns, ("字段", "点预测", "预测区间", "区间宽度", "评估集覆盖率"), (100, 90, 145, 90, 110)):
            self.table.heading(key, text=heading)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(side="left", expand=True, fill="both")
        self.status = tk.StringVar(value="请选择图片或打开示例。")
        self.details = tk.StringVar()
        for variable in (self.status, self.details):
            ttk.Label(body, textvariable=variable, wraplength=850).pack(anchor="w", pady=2)
        ttk.Label(body, text=domain_note + "单张图片的真实误差未知。").pack(anchor="w")

    def choose(self):
        name = filedialog.askopenfilename(parent=self.root, filetypes=[
            ("食物图片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("所有文件", "*.*")])
        if name:
            self.load(Path(name))

    def load(self, path):
        if self.future is not None:
            return
        self.path = Path(path)
        self.outputs = {}
        self.image = None
        self.preview.configure(image="", text=self.path.name)
        self.render()
        self.status.set("正在读取图片并计算预测，首次调用需要校验和加载权重。")
        for button in (self.open_button, self.demo_button, self.export_button):
            button.configure(state="disabled")
        self.future = self.pool.submit(self.predictor.run, self.path)
        self.root.after(60, self.poll)

    def poll(self):
        if not self.future.done():
            self.root.after(60, self.poll)
            return
        try:
            preview, self.outputs = self.future.result()
            self.image = ImageTk.PhotoImage(preview, master=self.root)
            self.preview.configure(image=self.image, text="", width=0)
            self.render()
            self.status.set(self.path.name)
            self.export_button.configure(state="normal")
        except Exception as exc:
            self.status.set(f"读取或预测失败：{exc}")
        finally:
            self.future = None
            for button in (self.open_button, self.demo_button):
                button.configure(state="normal")

    def render(self):
        self.table.delete(*self.table.get_children())
        self.details.set("")
        if not self.outputs:
            return
        mid = self.models[self.model.get()]
        out = self.outputs[mid]
        matched = self.coverage_version is None or self.coverage_version == out["calibration_version"]
        for key, title in FIELDS.items():
            field = out["fields"][key]
            lo, hi = field["lower"], field["upper"]
            self.table.insert("", "end", iid=key, values=(
                f"{title} ({field['unit']})", f"{field['point']:.1f}", f"{lo:.1f} – {hi:.1f}",
                f"{hi-lo:.1f}", f"{self.coverage[mid,key]:.2%}" if matched else "此版本待评估"))
        self.details.set(f"逐字段目标覆盖率 {out['target_coverage']:.0%}　｜　校准版本 {out['calibration_version']}　｜　模型版本 {out['model_version']}")

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.root, initialfile=self.path.stem+"_nutrition.json",
                                             defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.outputs, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
                self.status.set(f"已导出 {Path(path).name}")
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=self.root)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nutritionverse", action="store_true", help="Open the NutritionVerse plate model")
    args = parser.parse_args()
    window = tk.Tk()
    NutritionWindow(window, nutritionverse=args.nutritionverse)
    window.mainloop()
