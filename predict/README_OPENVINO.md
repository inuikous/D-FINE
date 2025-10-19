# D-FINE OpenVINO 変換・推論ガイド

## 🚀 OpenVINO版の使い方

Intel CPUやGPUで最適化された高速推論が可能です。

> ⚠️ **注意事項**
> - OpenVINOモデル（`.xml`）には **`inference_openvino.py`** を使用してください
> - Transformersモデル（`ustc-community/dfine_*_coco`）には **`inference.py`** を使用してください

### クイックリファレンス

| モデル形式 | 推論スクリプト | コマンド例 |
|-----------|---------------|------------|
| **OpenVINO** (.xml) | `inference_openvino.py` | `python predict\inference_openvino.py --model openvino_models/.../model.xml` |
| **Transformers** | `inference.py` | `python predict\inference.py --model ustc-community/dfine_n_coco` |

---

## 📦 インストール

```bash
# requirements.txtから一括インストール
pip install -r requirements.txt

# または個別にインストール
pip install torch transformers openvino openvino-dev pillow opencv-python numpy onnxscript onnx
```

---

## ⚙️ ステップ1: モデルをOpenVINOに変換

### 基本的な変換

```bash
# 軽量モデルを変換（推奨）
python export_openvino.py --model ustc-community/dfine_n_coco

# 中型モデルを変換
python export_openvino.py --model ustc-community/dfine_l_coco

# 最高精度モデルを変換
python export_openvino.py --model ustc-community/dfine_x_coco
```

### オプション指定

```bash
# カスタム入力サイズで変換
python export_openvino.py \
    --model ustc-community/dfine_n_coco \
    --height 800 \
    --width 800

# 出力先を指定
python export_openvino.py \
    --model ustc-community/dfine_n_coco \
    --output ./my_models
```

### 変換後の出力

```
openvino_models/
└── ustc-community_dfine_n_coco/
    ├── model.xml              # OpenVINO IRモデル（メタデータ）
    ├── model.bin              # モデルの重み
    ├── model.onnx             # 中間ONNX形式
    └── preprocessor_config.json  # 前処理設定
```

---

## 🔍 ステップ2: OpenVINOモデルで推論

⚠️ **重要**: OpenVINOモデル(.xml)は`inference_openvino.py`を使用してください！

### 基本的な推論

```bash
# サンプル画像で推論
python predict\inference_openvino.py \
    --model openvino_models/ustc-community_dfine_n_coco/model.xml

# 自分の画像で推論
python predict\inference_openvino.py \
    --model openvino_models/ustc-community_dfine_n_coco/model.xml \
    --image your_image.jpg
```

### 高度な使い方

```bash
# 検出閾値を変更
python predict\inference_openvino.py \
    --model openvino_models/ustc-community_dfine_n_coco/model.xml \
    --image test.jpg \
    --threshold 0.3

# GPUで実行（利用可能な場合）
python predict\inference_openvino.py \
    --model openvino_models/ustc-community_dfine_x_coco/model.xml \
    --image test.jpg \
    --device GPU

# 出力先を指定
python inference_openvino.py \
    --model openvino_models/ustc-community_dfine_n_coco/model.xml \
    --image test.jpg \
    --output ./results
```

---

## 📊 パフォーマンス比較

| 実装方式 | 推論速度 (CPU) | メモリ使用量 | 精度 | 特徴 |
|---------|---------------|------------|------|------|
| **Transformers版** | 遅い | 大 | 100% | 最も簡単 |
| **OpenVINO版** | **高速** | **小** | 99.9% | Intel最適化 |

---

## 💡 コマンドオプション

### export_openvino.py

| オプション | 短縮形 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--model` | `-m` | `ustc-community/dfine_n_coco` | モデル名 |
| `--output` | `-o` | `./openvino_models` | 出力ディレクトリ |
| `--height` | - | `640` | 入力画像の高さ |
| `--width` | - | `640` | 入力画像の幅 |

### inference_openvino.py

| オプション | 短縮形 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--model` | `-m` | **必須** | OpenVINO IRモデルのパス (.xml) |
| `--image` | `-i` | `None` | 入力画像のパス |
| `--output` | `-o` | `./output` | 出力ディレクトリ |
| `--threshold` | `-t` | `0.5` | 検出閾値 |
| `--device` | `-d` | `CPU` | 実行デバイス (CPU/GPU) |

---

## 🎯 利用可能なモデル

| モデル名 | パラメータ数 | AP (COCO) | 推奨用途 |
|---------|------------|-----------|----------|
| `ustc-community/dfine_n_coco` | 4M | 42.8% | リアルタイム推論 |
| `ustc-community/dfine_s_coco` | 10M | 48.5% | バランス型 |
| `ustc-community/dfine_m_coco` | 19M | 52.4% | 中精度 |
| `ustc-community/dfine_l_coco` | 31M | 54.0% | 高精度 |
| `ustc-community/dfine_x_coco` | 62M | 55.8% | 最高精度 |

---

## ✨ OpenVINO版の利点

✅ **CPU最適化**: Intel CPUで最大3-5倍高速化  
✅ **省メモリ**: メモリ使用量が50%以上削減  
✅ **GPU対応**: Intel GPU/NPUでさらに高速化  
✅ **エッジデバイス**: Raspberry Pi等でも動作  
✅ **量子化サポート**: INT8量子化でさらなる高速化が可能  
✅ **クロスプラットフォーム**: Windows/Linux/macOSで動作  

---

## 🐛 トラブルシューティング

### OpenVINOのインストールに失敗する

```bash
# 最新版をインストール
pip install --upgrade openvino openvino-dev
```

### GPUデバイスが見つからない

```bash
# 利用可能なデバイスを確認
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

### メモリ不足エラー

- より小さいモデル（dfine_n_coco）を使用
- 入力サイズを小さく設定（--height 416 --width 416）

### 変換エラー

- PyTorchとtransformersのバージョンを確認
- ONNXエクスポートがサポートされているか確認

---

## 📚 参考資料

- **OpenVINO公式**: https://docs.openvino.ai/
- **D-FINE公式**: https://github.com/Peterande/D-FINE
- **Hugging Face**: https://huggingface.co/ustc-community

---

## 📋 ファイル構成

```
predict/
├── export_openvino.py         # OpenVINO変換スクリプト
├── inference_openvino.py      # OpenVINO推論スクリプト
├── inference.py               # Transformers推論スクリプト
├── download_model.py          # モデルダウンロード
├── requirements.txt           # 依存パッケージ一覧
├── requirements_transformers.txt
├── README.md                  # メインREADME
└── README_OPENVINO.md         # このファイル（OpenVINO版ガイド）
```
