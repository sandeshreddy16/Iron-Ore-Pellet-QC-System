

import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd
from pathlib import Path
import json, os, shutil, math
import os

 
CONFIG = {
   
    "min_size_mm"  : 8.0,    
    "max_size_mm"  : 18.0,     
    "pixel_to_mm"  : 0.5,  
    "reference_pellet_real_mm": None,   
    "min_peak_distance"   : 12,    
    "watershed_threshold" : 0.25,  
   "circularity_threshold": 0.60,
    "min_contour_area_px": 50,

    "output_image_path": os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pellet_output.jpg"),

    "output_csv_path"  : os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pellet_results.csv"),
}

DIAGNOSTIC_MODE = False



COLORS = {
    "OK"           : (0,   210,  50),   
    "SMALL"        : (0,   165, 255),   
    "LARGE"        : (50,   50, 255),   
    "SHAPE_REJECT" : (255,  0,  200),   
    "TEXT"         : (255, 255, 255),   
    "OVERLAY"      : (15,   15,  15),   
}



class PelletAnalyzer:

    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        self._pixel_to_mm = cfg["pixel_to_mm"]  

    
    def load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f"Cannot open image: '{path}'\n"
                "Check the file path and that it is a valid JPEG/PNG."
            )
        print(f"  Image loaded: {img.shape[1]}×{img.shape[0]} px")
        return img

    
    def preprocess(self, img: np.ndarray):
       
        shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

       
        lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab_eq = cv2.merge([clahe.apply(l), a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

       
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        
        thresh = ndi.binary_fill_holes(thresh).astype(np.uint8) * 255

       
        k = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)

        return enhanced, gray, thresh

   
    def segment(self, thresh: np.ndarray):
       
        D = ndi.distance_transform_edt(thresh)

      
        coords = peak_local_max(
            D,
            min_distance=self.cfg["min_peak_distance"],
            labels=thresh
        )
        peak_mask = np.zeros(D.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers = ndi.label(peak_mask)[0]

        labels = watershed(-D, markers, mask=thresh)
        return labels, D

   
    def _auto_calibrate(self, labels: np.ndarray, gray: np.ndarray):
       
        ref_mm = self.cfg.get("reference_pellet_real_mm")
        if ref_mm is None:
            return  
        best_r = 0
        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == lbl] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            _, r = cv2.minEnclosingCircle(cnt)
            if r > best_r:
                best_r = r

        if best_r > 0:
            self._pixel_to_mm = ref_mm / (best_r * 2)
            print(f"  [CALIBRATION] Largest object = {best_r*2:.1f} px → "
                  f"pixel_to_mm = {self._pixel_to_mm:.4f}")

    
    def _measure_pellet(self, mask: np.ndarray):
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)

        if area_px < self.cfg["min_contour_area_px"]:
            return None  
       
        (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
        center      = (int(cx), int(cy))
        radius_px   = max(int(radius_px), 1)
        diameter_mm = radius_px * 2 * self._pixel_to_mm

        
        perimeter   = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area_px / perimeter ** 2) if perimeter > 0 else 0

        if DIAGNOSTIC_MODE:
            print(f"    [DIAG] center={center}  r_px={radius_px}  "
                  f"d_mm={diameter_mm:.2f}  circ={circularity:.3f}")

       
        mn = self.cfg["min_size_mm"]
        mx = self.cfg["max_size_mm"]
        ct = self.cfg["circularity_threshold"]

        if circularity < ct:
            status = "SHAPE_REJECT"
        elif diameter_mm < mn:
            status = "SMALL"
        elif diameter_mm > mx:
            status = "LARGE"
        else:
            status = "OK"

        color = COLORS[status]  

        return dict(
            diameter_mm = round(diameter_mm, 2),
            radius_px   = radius_px,
            center      = center,
            circularity = round(circularity, 3),
            status      = status,
            color       = color,
        )

    
    def analyse(self, img: np.ndarray):
       
        enhanced, gray, thresh = self.preprocess(img)
        labels, D              = self.segment(thresh)

        self._auto_calibrate(labels, gray)   

        output  = img.copy()
        records = []

        for label in np.unique(labels):
            if label == 0:
                continue  # background

            pellet_mask = np.zeros(gray.shape, dtype="uint8")
            pellet_mask[labels == label] = 255

            m = self._measure_pellet(pellet_mask)
            if m is None:
                continue

            records.append({
                "label"      : int(label),
                "diameter_mm": m["diameter_mm"],
                "circularity": m["circularity"],
                "status"     : m["status"],
                "center_x"   : m["center"][0],
                "center_y"   : m["center"][1],
                "radius_px"  : m["radius_px"],
            })

           
            cv2.circle(output, m["center"], m["radius_px"], m["color"], 2)

            text_y = m["center"][1] - m["radius_px"] - 5
            if text_y < 10:
                text_y = m["center"][1] + m["radius_px"] + 15

            cv2.putText(
                output,
                f"{m['diameter_mm']:.1f}mm",
                (m["center"][0] - 14, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, m["color"], 1, cv2.LINE_AA
            )

        df = pd.DataFrame(records)

        
        if not df.empty:
            output = self._draw_dashboard(output, df)
        output = self._draw_legend(output)

        return output, df, dict(thresh=thresh, labels=labels, D=D, enhanced=enhanced)

   
    def _draw_dashboard(self, img: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        total  = len(df)
        ok     = (df["status"] == "OK").sum()
        rate   = ok / total * 100
        mean_d = df["diameter_mm"].mean()
        std_d  = df["diameter_mm"].std()
        d10, d50, d90 = np.percentile(df["diameter_mm"], [10, 50, 90])

        lines = [
            f"Total        : {total}",
            f"Pass (OK)    : {ok}  ({rate:.1f}%)",
            f"Mean ± SD    : {mean_d:.1f} ± {std_d:.1f} mm",
            f"D10/D50/D90  : {d10:.1f}/{d50:.1f}/{d90:.1f} mm",
            f"Spec window  : {self.cfg['min_size_mm']}–{self.cfg['max_size_mm']} mm",
            f"px→mm factor : {self._pixel_to_mm:.4f}",
        ]

        box_w, box_h = 280, 20 + len(lines) * 20
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (box_w, box_h), COLORS["OVERLAY"], -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        for i, line in enumerate(lines):
            cv2.putText(img, line, (12, 24 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLORS["TEXT"], 1, cv2.LINE_AA)
        return img

    
    def _draw_legend(self, img: np.ndarray) -> np.ndarray:
        legend = [
            ("OK",           COLORS["OK"]),
            ("SMALL",        COLORS["SMALL"]),
            ("LARGE",        COLORS["LARGE"]),
            ("SHAPE REJECT", COLORS["SHAPE_REJECT"]),
        ]
        x0 = img.shape[1] - 165
        y0 = 10
        box_h = 15 + len(legend) * 20

        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 5, y0), (img.shape[1] - 5, y0 + box_h),
                      COLORS["OVERLAY"], -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        for i, (label, color) in enumerate(legend):
            y = y0 + 16 + i * 20
            cv2.circle(img, (x0 + 6, y - 4), 6, color, -1)
            cv2.putText(img, label, (x0 + 18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["TEXT"], 1, cv2.LINE_AA)
        return img

    
    def save(self, output_img: np.ndarray, df: pd.DataFrame):
        p = self.cfg.get("output_image_path")
        if p:
            cv2.imwrite(p, output_img)
            print(f"  [✓] Annotated image → {p}")

        c = self.cfg.get("output_csv_path")
        if c and not df.empty:
            df.to_csv(c, index=False)
            print(f"  [✓] CSV results     → {c}")

    
    def plot_report(self, img_orig, output_img, df, intermediates):
        bg   = "#0f0f1a"
        tc   = "white"
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.patch.set_facecolor(bg)

        def style(ax, title):
            ax.set_facecolor(bg)
            ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=6)
            ax.tick_params(colors="#777")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")

       
        axes[0,0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        style(axes[0,0], "① Original Image")
        axes[0,0].axis("off")

        
        axes[0,1].imshow(intermediates["thresh"], cmap="gray")
        style(axes[0,1], "② Binary Mask (CLAHE + Adaptive Threshold)")
        axes[0,1].axis("off")

        
        axes[0,2].imshow(intermediates["labels"], cmap="nipy_spectral")
        style(axes[0,2], "③ Watershed Segmentation")
        axes[0,2].axis("off")

       
        axes[1,0].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        style(axes[1,0], f"④ Final Detection  (n={len(df)})")
        axes[1,0].axis("off")

        if not df.empty:
            mn = self.cfg["min_size_mm"]
            mx = self.cfg["max_size_mm"]

           
            axes[1,1].set_facecolor("#0d0d1f")
            axes[1,1].axvspan(mn, mx, alpha=0.18, color="lime", label="Spec window")
            axes[1,1].hist(df["diameter_mm"], bins=20,
                           color="#4fc3f7", edgecolor="#0d0d1f", linewidth=0.5)
            axes[1,1].axvline(df["diameter_mm"].mean(), color="yellow",
                              ls="--", lw=1.3, label=f"Mean {df['diameter_mm'].mean():.1f}mm")
            axes[1,1].set_xlabel("Diameter (mm)", color="#aaa")
            axes[1,1].set_ylabel("Count", color="#aaa")
            axes[1,1].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            style(axes[1,1], "⑤ Granulometry Distribution")

            
            counts = df["status"].value_counts()
            pie_clr_map = {
                "OK"           : "#00c853",
                "SMALL"        : "#ff6d00",
                "LARGE"        : "#dd2222",
                "SHAPE_REJECT" : "#e040fb",
            }
            pie_colors = [pie_clr_map.get(s, "#888") for s in counts.index]
            axes[1,2].set_facecolor("#0d0d1f")
            wedges, texts, autos = axes[1,2].pie(
                counts.values, labels=counts.index, colors=pie_colors,
                autopct="%1.1f%%",
                textprops=dict(color="white", fontsize=9),
                wedgeprops=dict(linewidth=1.4, edgecolor="#0f0f1a")
            )
            for a in autos:
                a.set_fontsize(8)
            style(axes[1,2], "⑥ Status Breakdown")

        else:
            for ax in axes[1,1:]:
                ax.text(0.5, 0.5, "No pellets detected",
                        ha="center", va="center", color="white", fontsize=12)
                ax.axis("off")

        plt.suptitle("Iron Ore Pellet Classification Report  —  v2",
                     color="white", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    
    def print_summary(self, df: pd.DataFrame):
        if df.empty:
            print("  No pellets detected.")
            return

        total  = len(df)
        counts = df["status"].value_counts()
        ok     = counts.get("OK", 0)
        mn, mx = self.cfg["min_size_mm"], self.cfg["max_size_mm"]
        d10, d50, d90 = np.percentile(df["diameter_mm"], [10, 50, 90])

        print("\n" + "═" * 52)
        print("   PELLET CLASSIFICATION SUMMARY")
        print("═" * 52)
        print(f"   Total detected   : {total}")
        for status, cnt in counts.items():
            bar  = "█" * int(cnt / total * 20)
            pct  = cnt / total * 100
            print(f"   {status:<16} : {cnt:>4}  ({pct:5.1f}%)  {bar}")
        print("─" * 52)
        print(f"   Pass rate        : {ok/total*100:.1f}%")
        print(f"   Mean diameter    : {df['diameter_mm'].mean():.2f} mm")
        print(f"   Std deviation    : {df['diameter_mm'].std():.2f} mm")
        print(f"   D10 / D50 / D90  : {d10:.1f} / {d50:.1f} / {d90:.1f} mm")
        print(f"   Spec window      : {mn}–{mx} mm")
        print(f"   pixel_to_mm used : {self._pixel_to_mm:.4f}")
        print("═" * 52)

        if ok / total < 0.85:
            print("\n  ⚠️  PASS RATE IS LOW — CALIBRATION CHECKLIST:")
            print(f"     Current pixel_to_mm = {self._pixel_to_mm:.4f}")
            print(f"     Mean detected size   = {df['diameter_mm'].mean():.2f} mm")
            print(f"     Expected mean        ≈ {(mn+mx)/2:.1f} mm  (centre of spec)")
            ratio = ((mn+mx)/2) / max(df["diameter_mm"].mean(), 0.001)
            print(f"     Suggested pixel_to_mm = {self._pixel_to_mm * ratio:.4f}  "
                  f"(multiply current by {ratio:.2f})")
            print("     → Update CONFIG['pixel_to_mm'] with the suggested value.\n")



def export_yolo_dataset(
    image_paths     : list,
    output_dir      : str = "yolo_dataset",
    cfg             : dict = CONFIG,
    train_split     : float = 0.8,
):
   
    CLASS_MAP = {"OK": 0, "SMALL": 1, "LARGE": 2, "SHAPE_REJECT": 3}
    CLASS_NAMES = ["pellet_ok", "pellet_small", "pellet_large", "pellet_shape_reject"]

    analyzer = PelletAnalyzer(cfg)

    
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            Path(f"{output_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

    n_train = int(len(image_paths) * train_split)
    splits  = (["train"] * n_train) + (["val"] * (len(image_paths) - n_train))

    total_labels = 0

    for path, split in zip(image_paths, splits):
        print(f"  Exporting {Path(path).name}  [{split}]")
        try:
            img = analyzer.load(path)
        except FileNotFoundError as e:
            print(f"    [!] {e}")
            continue

        _, df, _ = analyzer.analyse(img)
        if df.empty:
            continue

        h, w = img.shape[:2]
        stem  = Path(path).stem

        
        dst_img = f"{output_dir}/{split}/images/{stem}.jpg"
        shutil.copy(path, dst_img)

        
        label_lines = []
        for _, row in df.iterrows():
            cls = CLASS_MAP.get(row["status"])
            if cls is None:
                continue
            cx = row["center_x"] / w
            cy = row["center_y"] / h
            bw = (row["radius_px"] * 2) / w
            bh = (row["radius_px"] * 2) / h
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_lbl = f"{output_dir}/{split}/labels/{stem}.txt"
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))

        total_labels += len(label_lines)

    
    yaml_content = (
        f"path: {Path(output_dir).resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(f"{output_dir}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n  [✓] YOLO dataset saved → {output_dir}/")
    print(f"      {len(image_paths)} images  |  {total_labels} bounding boxes")
    print(f"\n  NEXT STEPS — Train YOLOv8:")
    print(f"    pip install ultralytics")
    print(f"    yolo train model=yolov8n.pt data={output_dir}/data.yaml epochs=100 imgsz=640")
    print(f"    yolo val   model=runs/detect/train/weights/best.pt data={output_dir}/data.yaml")



def run_batch(image_paths: list, cfg: dict = CONFIG, show_plots: bool = False):
    analyzer    = PelletAnalyzer(cfg)
    all_records = []

    for path in image_paths:
        print(f"\n[→] {path}")
        try:
            img = analyzer.load(path)
        except FileNotFoundError as e:
            print(f"  [!] {e}")
            continue

        output_img, df, intermediates = analyzer.analyse(img)
        df["source_image"] = Path(path).name
        all_records.append(df)
        analyzer.print_summary(df)

        stem = Path(path).stem
        cv2.imwrite(f"{stem}_annotated.jpg", output_img)

        if show_plots:
            analyzer.plot_report(img, output_img, df, intermediates)

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        combined.to_csv("batch_results.csv", index=False)
        print(f"\n[✓] Batch CSV → batch_results.csv  ({len(combined)} pellets)")
        return combined
    return pd.DataFrame()



if __name__ == "__main__":

    IMAGE_PATH = r"C:\Users\sande\Downloads\Pallet Project\Original_images\pellet_9.jpeg"

    

    print("=" * 52)
    print("  Iron Ore Pellet Classifier  v2")
    print("=" * 52)

    analyzer = PelletAnalyzer(CONFIG)

    print("\n[1/4] Loading …")
    img = analyzer.load(IMAGE_PATH)

    print("[2/4] Analysing …")
    output_img, df, intermediates = analyzer.analyse(img)

    print("[3/4] Saving …")
    analyzer.save(output_img, df)

    print("[4/4] Report …")
    analyzer.print_summary(df)
    analyzer.plot_report(img, output_img, df, intermediates)

   