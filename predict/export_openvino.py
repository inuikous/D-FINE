"""D-FINEモデルをOpenVINO形式に変換するスクリプト"""
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor
from pathlib import Path
import sys

def export_to_openvino(
    model_name="ustc-community/dfine_n_coco",
    output_dir="./openvino_models",
    input_size=(640, 640)
):
    """
    D-FINEモデルをOpenVINO IR形式に変換
    
    Args:
        model_name: Hugging Faceのモデル名
        output_dir: 出力ディレクトリ
        input_size: 入力画像サイズ (height, width)
    """
    print("=" * 60)
    print("D-FINE → OpenVINO 変換")
    print("=" * 60)
    
    # 出力ディレクトリの作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル名からフォルダ名を作成
    model_folder = model_name.replace("/", "_")
    model_output_dir = output_dir / model_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # OpenVINOのインポート確認
        try:
            import openvino as ov
            print(f"✓ OpenVINO version: {ov.__version__}")
        except ImportError:
            print("✗ エラー: OpenVINOがインストールされていません")
            print("pip install openvino openvino-dev")
            sys.exit(1)
        
        # モデルとプロセッサのロード
        print(f"\nモデルをロード中: {model_name}")
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = DFineForObjectDetection.from_pretrained(model_name)
        model.eval()
        print("✓ モデルのロード完了")
        
        # ダミー入力の作成
        height, width = input_size
        dummy_image = torch.randn(1, 3, height, width)
        print(f"\nダミー入力作成: {dummy_image.shape}")
        
        # ONNXへのエクスポート
        onnx_path = model_output_dir / "model.onnx"
        print(f"\nONNX形式にエクスポート中: {onnx_path}")
        
        torch.onnx.export(
            model,
            (dummy_image,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print("✓ ONNXエクスポート完了")
        
        # OpenVINO IRへの変換
        print("\nOpenVINO IR形式に変換中...")
        ov_model = ov.convert_model(str(onnx_path))
        
        # OpenVINO IRの保存
        ir_path = model_output_dir / "model.xml"
        ov.save_model(ov_model, str(ir_path))
        print(f"✓ OpenVINO IR保存完了: {ir_path}")
        
        # プロセッサの設定も保存（詳細情報を含む）
        processor_config = {
            "image_mean": image_processor.image_mean,
            "image_std": image_processor.image_std,
            "size": {"height": height, "width": width},
            "do_normalize": getattr(image_processor, 'do_normalize', False),  # デフォルトFalse
            "do_resize": getattr(image_processor, 'do_resize', True),
            "do_pad": getattr(image_processor, 'do_pad', False),
            "do_rescale": getattr(image_processor, 'do_rescale', True),
            "rescale_factor": getattr(image_processor, 'rescale_factor', 1.0 / 255.0),
            "resample": str(getattr(image_processor, 'resample', 'BILINEAR')),
        }
        
        import json
        config_path = model_output_dir / "preprocessor_config.json"
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        print(f"✓ プリプロセッサ設定保存: {config_path}")
        
        print("\n" + "=" * 60)
        print("✓ 変換完了!")
        print(f"\n出力ファイル:")
        print(f"  - {ir_path}")
        print(f"  - {ir_path.with_suffix('.bin')}")
        print(f"  - {config_path}")
        print(f"\nモデルサイズ: {ir_path.with_suffix('.bin').stat().st_size / 1024 / 1024:.1f} MB")
        print("=" * 60)
        
        return model_output_dir
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="D-FINEモデルをOpenVINOに変換")
    parser.add_argument("--model", "-m", default="ustc-community/dfine_n_coco",
                       help="モデル名")
    parser.add_argument("--output", "-o", default="./openvino_models",
                       help="出力ディレクトリ")
    parser.add_argument("--height", type=int, default=640,
                       help="入力画像の高さ")
    parser.add_argument("--width", type=int, default=640,
                       help="入力画像の幅")
    
    args = parser.parse_args()
    
    export_to_openvino(
        model_name=args.model,
        output_dir=args.output,
        input_size=(args.height, args.width)
    )

if __name__ == "__main__":
    main()
