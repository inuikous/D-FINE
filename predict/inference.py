"""D-FINE物体検出推論スクリプト"""
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor
from transformers.image_utils import load_image
from PIL import Image
import cv2
from pathlib import Path

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def visualize(img_path, results, out_path, th=0.5):
    img = cv2.imread(str(img_path))
    for r in results:
        for s, l, b in zip(r["scores"], r["labels"], r["boxes"]):
            if s.item() < th: continue
            x1, y1, x2, y2 = [int(c) for c in b.tolist()]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{COCO_CLASSES[l.item()]}: {s.item():.2f}"
            cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(str(out_path), img)
    print(f"結果保存: {out_path}")

def run(model="ustc-community/dfine_n_coco", img=None, out="./output", th=0.5, dev=None):
    print("="*60, "\nD-FINE物体検出\n", "="*60)
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {dev}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"\nモデルロード: {model}")
    proc = AutoImageProcessor.from_pretrained(model)
    m = DFineForObjectDetection.from_pretrained(model).to(dev)
    m.eval()
    
    if not img:
        print("サンプル画像ダウンロード...")
        image = load_image('http://images.cocodataset.org/val2017/000000039769.jpg')
        img = out / "input.jpg"
        image.save(img)
    else:
        img = Path(img)
        image = Image.open(img).convert("RGB")
    
    print(f"推論実行: {img}")
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = m(**inputs)
    
    results = proc.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[image.height, image.width]]).to(dev), threshold=th
    )
    
    print(f"\n検出結果 (閾値{th}):\n" + "-"*60)
    print(f"検出数: {len(results[0]['scores'])}")
    for s, l, b in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
        print(f"  {COCO_CLASSES[l.item()]}: {s.item():.3f} {[round(i,2) for i in b.tolist()]}")
    
    out_path = out / f"result_{img.stem}.jpg"
    visualize(img, results, out_path, th)
    print(f"\n{'='*60}\n推論完了!\n結果: {out_path}\n{'='*60}")
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="ustc-community/dfine_n_coco")
    p.add_argument("--image", "-i", default=None)
    p.add_argument("--output", "-o", default="./output")
    p.add_argument("--threshold", "-t", type=float, default=0.5)
    p.add_argument("--device", "-d", default=None)
    a = p.parse_args()
    run(a.model, a.image, a.output, a.threshold, a.device)
