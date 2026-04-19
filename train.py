from ultralytics import YOLO

print("=" * 50)
print("  Iron Ore Pellet Classifier - Training v2")
print("=" * 50)


model = YOLO(r"C:\Users\sande\Downloads\Pallet Project\runs\detect\pellet_model\weights\best.pt")

results = model.train(
    data     = r"C:\Users\sande\Downloads\Pallet Project\yolo_dataset\data.yaml",
    epochs   = 200,      
    patience = 50,       
    
    imgsz    = 640, 
    batch    = 8,

    
    augment  = True,
    fliplr   = 0.5,       
    flipud   = 0.3,       
    degrees  = 45,        
    scale    = 0.5,       
    mosaic   = 1.0,       
    hsv_v    = 0.4,       
    hsv_s    = 0.7,       

   
    cls      = 1.5,       

   
    name     = "pellet_model_v2",
    verbose  = True,
)


print("  TRAINING COMPLETE!")


metrics = results
print(f"\n  Overall  mAP50 : {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
print(f"  Precision      : {results.results_dict.get('metrics/precision(B)', 0):.3f}")
print(f"  Recall         : {results.results_dict.get('metrics/recall(B)', 0):.3f}")


