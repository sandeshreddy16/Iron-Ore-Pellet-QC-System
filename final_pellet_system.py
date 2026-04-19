import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from pellet_analyzer_v2 import PelletAnalyzer, CONFIG
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH    = os.path.join(BASE_DIR, "runs", "detect","pellet_model_v2", "weights", "best.pt")
IMAGE_FOLDER  = os.path.join(BASE_DIR, "Original_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Final_output")
CONF_THRESHOLD = 0.25

CLASS_NAMES = {0: "OK", 1: "SMALL", 2: "LARGE", 3: "SHAPE_REJECT"}

COLORS = {
    "OK"          : (0,   210,  50),
    "SMALL"       : (0,   165, 255),
    "LARGE"       : (50,   50, 255),
    "SHAPE_REJECT": (255,   0,  200),
}

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def measure_region(region, analyzer):
    try:
        if region is None or region.size == 0:
            return 0.0
        enhanced, gray, thresh = analyzer.preprocess(region)
        labels, D = analyzer.segment(thresh)
        records = []
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            m = analyzer._measure_pellet(mask)
            if m:
                records.append({"diameter_mm": m["diameter_mm"]})
        df = pd.DataFrame(records)
        return round(df["diameter_mm"].mean(), 1) if not df.empty else 0.0
    except Exception:
        return 0.0


print("=" * 55)
print("  Iron Ore Pellet Classification System")
print("=" * 55)

yolo_model  = YOLO(MODEL_PATH)
cv_analyzer = PelletAnalyzer(CONFIG)

images = list(Path(IMAGE_FOLDER).glob("*.jpeg")) + \
         list(Path(IMAGE_FOLDER).glob("*.jpg"))

print(f"\nProcessing {len(images)} images...\n")

total_all = ok_all = small_all = large_all = shape_all = 0

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    output  = img.copy()
    results = yolo_model.predict(
        source  = str(img_path),
        conf    = CONF_THRESHOLD,
        iou     = 0.3,
        verbose = False
    )

    img_total = img_ok = img_small = img_large = img_shape = 0

    for r in results:
        for box, cls, conf_val in zip(
            r.boxes.xyxy.tolist(),
            r.boxes.cls.tolist(),
            r.boxes.conf.tolist()
        ):
            if int(cls) == 3 and conf_val < 0.65:
                cls = 0
            cls    = int(cls)
            status = CLASS_NAMES[cls]
            color  = COLORS[status]
            x1, y1, x2, y2 = map(int, box)
            cx  = (x1 + x2) // 2
            cy  = (y1 + y2) // 2
            rad = max((x2 - x1) // 2, 5)

            diameter = measure_region(img[y1:y2, x1:x2], cv_analyzer)
            if diameter == 0.0:
                diameter = round(((x2-x1+y2-y1)/2) * CONFIG["pixel_to_mm"], 1)

            cv2.circle(output, (cx, cy), rad, color, 2)
            text_y = cy - rad - 5 if cy - rad - 5 > 15 else cy + rad + 15
            cv2.putText(output, f"{diameter:.1f}mm",
                        (cx-15, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)

            img_total += 1
            if   cls == 0: img_ok    += 1
            elif cls == 1: img_small += 1
            elif cls == 2: img_large += 1
            elif cls == 3: img_shape += 1

    cv2.imwrite(str(Path(OUTPUT_FOLDER) / f"result_{img_path.stem}.jpg"), output)

    total_all += img_total
    ok_all    += img_ok
    small_all += img_small
    large_all += img_large
    shape_all += img_shape

    if img_total > 0:
        print(f"  {img_path.name}  →  "
              f"Total:{img_total}  OK:{img_ok}  "
              f"Pass:{img_ok/img_total*100:.1f}%  ✅")

print("\n" )
print("  FINAL SUMMARY")
print("\n")
print(f"  Images processed : {len(images)}")
print(f"  Total pellets    : {total_all}")
if total_all > 0:
    rate = ok_all / total_all * 100
    print(f"  OK               : {ok_all}  ({rate:.1f}%)")
    print(f"  Small            : {small_all}")
    print(f"  Large            : {large_all}")
    print(f"  Shape Reject     : {shape_all}")
    print(f"\n  PASS RATE        : {rate:.1f}%")
    


show = input("\nShow visual report? (y/n): ")
if show.lower() == 'y':

    sample_path   = images[0]
    img_sample    = cv2.imread(str(sample_path))
    output_sample = img_sample.copy()
    diams         = []

    r = yolo_model.predict(
        source  = str(sample_path),
        conf    = CONF_THRESHOLD,
        iou     = 0.3,
        verbose = False
    )[0]

    for box, cls, conf_val in zip(
        r.boxes.xyxy.tolist(),
        r.boxes.cls.tolist(),
        r.boxes.conf.tolist()
    ):
        if int(cls) == 3 and conf_val < 0.65:
            cls = 0
        cls    = int(cls)
        status = CLASS_NAMES[cls]
        color  = COLORS[status]
        x1, y1, x2, y2 = map(int, box)
        cx  = (x1+x2)//2
        cy  = (y1+y2)//2
        rad = max((x2-x1)//2, 5)

        diameter = measure_region(img_sample[y1:y2, x1:x2], cv_analyzer)
        if diameter == 0.0:
            diameter = round(((x2-x1+y2-y1)/2)*CONFIG["pixel_to_mm"], 1)

        diams.append(diameter)
        cv2.circle(output_sample, (cx, cy), rad, color, 2)
        text_y = cy - rad - 5 if cy - rad - 5 > 15 else cy + rad + 15
        cv2.putText(output_sample, f"{diameter:.1f}mm",
                    (cx-15, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    bg = "#0f0f1a"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(bg)

    def style(ax, title):
        ax.set_facecolor(bg)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#777")

    axes[0].imshow(cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB))
    style(axes[0], "① Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(output_sample, cv2.COLOR_BGR2RGB))
    style(axes[1], f"② YOLO Detection  (n={len(diams)})")
    axes[1].axis("off")

    if diams:
        mn = CONFIG["min_size_mm"]
        mx = CONFIG["max_size_mm"]
        axes[2].set_facecolor("#0d0d1f")
        axes[2].axvspan(mn, mx, alpha=0.18, color="lime", label="Spec window")
        axes[2].hist(diams, bins=20, color="#4fc3f7",
                     edgecolor="#0d0d1f", linewidth=0.5)
        mean_d = sum(diams) / len(diams)
        axes[2].axvline(mean_d, color="yellow", ls="--",
                        lw=1.3, label=f"Mean {mean_d:.1f}mm")
        axes[2].set_xlabel("Diameter (mm)", color="#aaa")
        axes[2].set_ylabel("Count", color="#aaa")
        axes[2].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        style(axes[2], "③ Granulometry Distribution")

    plt.suptitle(
        f"Iron Ore Pellet Report — {sample_path.name}",
        color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()