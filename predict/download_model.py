"""
D-FINE事前学習済みモデルのダウンロードスクリプト (Transformers版)
Hugging Faceのtransformersライブラリを使用してモデルをダウンロードします。
"""
import sys

# transformersのバージョンチェック
try:
    import transformers
    from packaging import version
    
    required_version = "4.57.0"
    current_version = transformers.__version__
    
    if version.parse(current_version) < version.parse(required_version):
        print(f"✗ エラー: transformersのバージョンが古すぎます")
        print(f"  現在のバージョン: {current_version}")
        print(f"  必要なバージョン: {required_version}以上")
        print(f"\n以下のコマンドでアップデートしてください:")
        print(f"  pip install --upgrade transformers>=4.57.0")
        sys.exit(1)
        
except ImportError:
    print("✗ エラー: transformersがインストールされていません")
    print("\n以下のコマンドでインストールしてください:")
    print("  pip install transformers>=4.57.0")
    sys.exit(1)

from transformers import DFineForObjectDetection, AutoImageProcessor
import os
from pathlib import Path

def download_dfine_model(model_name="ustc-community/dfine_n_coco", save_dir="../weights"):
    """
    D-FINE事前学習済みモデルをダウンロード
    
    Args:
        model_name: Hugging Faceのモデル名
                   利用可能なモデル:
                   - ustc-community/dfine_n_coco (4M, 42.8% AP) - 最軽量
                   - ustc-community/dfine_s_coco (10M, 48.5% AP)
                   - ustc-community/dfine_m_coco (19M, 52.4% AP)
                   - ustc-community/dfine_l_coco (31M, 54.0% AP)
                   - ustc-community/dfine_x_coco (62M, 55.8% AP) - 最高精度
        save_dir: 保存先ディレクトリ（デフォルト: ../weights）
    """
    # 保存ディレクトリの作成
    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    # モデル名からディレクトリ名を生成
    model_dir = save_path / model_name.replace("/", "_")
    
    print(f"モデルをダウンロード中: {model_name}")
    print(f"保存先: {model_dir}")
    
    try:
        # モデルとプロセッサをダウンロード
        print("\n1. Image Processorをダウンロード中...")
        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=str(save_path)
        )
        # ローカルに保存
        processor_dir = model_dir / "processor"
        processor_dir.mkdir(parents=True, exist_ok=True)
        image_processor.save_pretrained(str(processor_dir))
        print(f"✓ Image Processorを保存: {processor_dir}")
        
        print("\n2. モデルをダウンロード中...")
        model = DFineForObjectDetection.from_pretrained(
            model_name,
            cache_dir=str(save_path)
        )
        # ローカルに保存
        model_save_dir = model_dir / "model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_save_dir))
        print(f"✓ モデルを保存: {model_save_dir}")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_save_dir))
        print(f"✓ モデルを保存: {model_save_dir}")
        
        # モデル情報を表示
        print(f"\n✓ ダウンロード完了!")
        print(f"モデル名: {model_name}")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"保存場所: {model_dir}")
        
        # 情報ファイルを作成
        info_file = model_dir / "info.txt"
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
            f.write(f"Processor Path: {processor_dir}\n")
            f.write(f"Model Path: {model_save_dir}\n")
        print(f"✓ 情報ファイル: {info_file}")
        
        return model, image_processor, model_dir
        
    except Exception as e:
        print(f"\n✗ ダウンロードエラー: {e}")
        raise

def main():
    """メイン関数"""
    print("=" * 60)
    print("D-FINE モデルダウンロードスクリプト (Transformers版)")
    print("=" * 60)
    
    # デフォルトで最軽量モデルをダウンロード
    model_name = "ustc-community/dfine_n_coco"
    save_dir = "../weights"
    
    try:
        model, image_processor, model_dir = download_dfine_model(
            model_name=model_name,
            save_dir=save_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ セットアップ完了!")
        print("\n次のステップ:")
        print(f"  python inference.py --model {model_dir}")
        print("  で推論を実行してください。")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ エラーが発生しました: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
