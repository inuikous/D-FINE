"""OpenVINO変換後のD-FINEモデルで推論するスクリプト"""
import numpy as np
import cv2
from pathlib import Path
import json
import sys

try:
    import openvino as ov
except ImportError:
    print("✗ エラー: OpenVINOがインストールされていません")
    print("pip install openvino")
    sys.exit(1)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess_image(image_path, config):
    """画像を前処理"""
    image = cv2.imread(str(image_path))
    orig_h, orig_w = image.shape[:2]
    
    # リサイズ
    target_h = config["size"]["height"]
    target_w = config["size"]["width"]
    resized = cv2.resize(image, (target_w, target_h))
    
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # rescale (0-255 -> 0-1)
    rescale_factor = config.get("rescale_factor", 1.0 / 255.0)
    img_float = rgb.astype(np.float32) * rescale_factor
    
    # 正規化（do_normalizeがTrueの場合のみ）
    if config.get("do_normalize", False):
        mean = np.array(config["image_mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(config["image_std"], dtype=np.float32).reshape(1, 1, 3)
        img_float = (img_float - mean) / std
    
    # CHW形式に変換してバッチ次元を追加
    input_tensor = img_float.transpose(2, 0, 1)[np.newaxis, ...]
    
    return input_tensor, (orig_h, orig_w), image

def postprocess_output(logits, boxes, orig_size, threshold=0.5):
    """モデル出力を後処理"""
    orig_h, orig_w = orig_size
    
    # logitsをsoftmaxでスコアに変換
    exp_logits = np.exp(logits[0] - np.max(logits[0], axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # 最大スコアとクラスを取得
    scores = probs.max(axis=-1)
    labels = probs.argmax(axis=-1)
    
    # 閾値でフィルタ
    valid_idx = scores > threshold
    
    results = {
        "scores": scores[valid_idx],
        "labels": labels[valid_idx],
        "boxes": boxes[0][valid_idx].copy()
    }
    
    # D-FINEのボックスは[cx, cy, w, h]の正規化座標 (0-1)
    # これを[x1, y1, x2, y2]の絶対座標に変換
    cx = results["boxes"][:, 0] * orig_w
    cy = results["boxes"][:, 1] * orig_h
    w = results["boxes"][:, 2] * orig_w
    h = results["boxes"][:, 3] * orig_h
    
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    results["boxes"][:, 0] = cx - w / 2  # x1
    results["boxes"][:, 1] = cy - h / 2  # y1
    results["boxes"][:, 2] = cx + w / 2  # x2
    results["boxes"][:, 3] = cy + h / 2  # y2
    
    return results

def visualize_results(image, results, output_path, threshold=0.5):
    """検出結果を可視化"""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score < threshold:
            continue
        
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        x1, y1, x2, y2 = box.astype(int)
        
        # ボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ラベルを描画
        text = f"{class_name}: {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), image)
    print(f"結果保存: {output_path}")

def run_openvino_inference(
    model_path,
    image_path=None,
    output_dir="./output",
    threshold=0.5,
    device="CPU"
):
    """
    OpenVINOモデルで推論を実行
    
    Args:
        model_path: OpenVINO IRモデルのパス (.xml)
        image_path: 入力画像のパス
        output_dir: 出力ディレクトリ
        threshold: 検出閾値
        device: 実行デバイス (CPU/GPU)
    """
    print("=" * 60)
    print("D-FINE OpenVINO 推論")
    print("=" * 60)
    
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # プリプロセッサ設定の読み込み
    config_path = model_path.parent / "preprocessor_config.json"
    if not config_path.exists():
        print(f"✗ エラー: プリプロセッサ設定が見つかりません: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    # OpenVINOコアの初期化
    print(f"\nOpenVINOコアを初期化: {ov.__version__}")
    core = ov.Core()
    
    # 利用可能なデバイスを表示
    available_devices = core.available_devices
    print(f"利用可能なデバイス: {available_devices}")
    
    # モデルの読み込み
    print(f"\nモデルをロード: {model_path}")
    model = core.read_model(str(model_path))
    
    # デバイス上でコンパイル
    print(f"デバイスでコンパイル: {device}")
    compiled_model = core.compile_model(model, device)
    
    # 入出力情報
    input_layer = compiled_model.input(0)
    output_layers = compiled_model.outputs
    
    # 動的形状の場合は部分形状を表示
    try:
        print(f"\n入力形状: {input_layer.shape}")
    except RuntimeError:
        print(f"\n入力形状: {input_layer.partial_shape} (動的)")
    
    print(f"出力数: {len(output_layers)}")
    
    # 画像の読み込みと前処理
    if image_path is None:
        # サンプル画像をダウンロード
        print("\nサンプル画像をダウンロード...")
        import urllib.request
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image_path = output_dir / "input_image.jpg"
        urllib.request.urlretrieve(url, image_path)
        print(f"保存: {image_path}")
    else:
        image_path = Path(image_path)
    
    print(f"\n画像を前処理: {image_path}")
    input_tensor, orig_size, orig_image = preprocess_image(image_path, config)
    
    # 推論実行
    print("\n推論実行中...")
    result = compiled_model([input_tensor])
    
    # 出力を取得
    logits = result[output_layers[0]]  # クラスロジット
    boxes = result[output_layers[1]]   # バウンディングボックス
    
    print(f"出力形状:")
    print(f"  logits: {logits.shape}")
    print(f"  boxes: {boxes.shape}")
    
    # 後処理
    results = postprocess_output(logits, boxes, orig_size, threshold)
    
    # 結果表示
    print(f"\n検出結果 (閾値: {threshold}):")
    print("-" * 60)
    print(f"検出数: {len(results['scores'])}")
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
        print(f"  {class_name}: {score:.3f} {[round(float(x), 2) for x in box]}")
    
    # 可視化
    output_path = output_dir / f"result_openvino_{image_path.stem}.jpg"
    visualize_results(orig_image.copy(), results, output_path, threshold)
    
    print("\n" + "=" * 60)
    print("✓ 推論完了!")
    print(f"結果: {output_path}")
    print("=" * 60)
    
    return results

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenVINO D-FINE推論")
    parser.add_argument("--model", "-m", required=True,
                       help="OpenVINO IRモデルのパス (.xml)")
    parser.add_argument("--image", "-i", default=None,
                       help="入力画像のパス")
    parser.add_argument("--output", "-o", default="./output",
                       help="出力ディレクトリ")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="検出閾値")
    parser.add_argument("--device", "-d", default="CPU",
                       help="実行デバイス (CPU/GPU)")
    
    args = parser.parse_args()
    
    run_openvino_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device
    )

if __name__ == "__main__":
    main()
