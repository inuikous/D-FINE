# D-FINE 推論スクリプト# D-FINE 推論スクリプト



事前学習済みD-FINEモデルを使用した物体検出の推論・結果描画プログラムです。事前学習済みD-FINEモデルを使用した物体検出の推論・結果描画プログラムです。



## 🎯 2つの実装方法## 構成



### 1. **Hugging Face Transformers版（推奨）** ⭐```

公式のtransformersライブラリを使用した最も簡単な実装です。predict/

├── download_model.py  # モデルダウンロードスクリプト（オンライン環境用）

### 2. オリジナル版├── inference.py       # 推論スクリプト（オフライン環境用）

GitHub Releasesから直接モデルをダウンロードする実装です。└── README.md         # このファイル

```

---

## 使用方法

## 📦 方法1: Transformers版（推奨）

### ステップ1: モデルのダウンロード（オンライン環境）

### インストール

```bash

```bash# 必要なパッケージをインストール

pip install -r requirements_transformers.txtpip install torch torchvision huggingface_hub opencv-python pillow

```

# モデルをダウンロード

または:python download_model.py

```

```bash

pip install torch>=2.0.0 transformers>=4.57.0 pillow opencv-python numpyこれにより `weights/` ディレクトリにモデルファイルがダウンロードされます。

```

### ステップ2: モデルをオフライン環境に転送

### 使用方法

```bash

#### 基本的な使用（サンプル画像で試す）# weightsディレクトリをオフライン環境にコピー

# 例: USBメモリ、ネットワーク共有、など

```bash```

python inference_transformers.py

```### ステップ3: 推論の実行（オフライン環境）



#### 自分の画像で推論```bash

# 推論を実行（サンプル画像が自動生成されます）

```bashpython inference.py

python inference_transformers.py --image your_image.jpg```

```

独自の画像で推論する場合:

#### 利用可能なモデル```bash

# sample_input.jpg を用意してから実行

```bashpython inference.py

# 軽量モデル（デフォルト）```

python inference_transformers.py --model ustc-community/dfine_n_coco

## 出力

# 中型モデル

python inference_transformers.py --model ustc-community/dfine_l_coco- `output_result.jpg`: 検出結果が描画された画像



# 最大モデル（最高精度）## 対応モデル

python inference_transformers.py --model ustc-community/dfine_x_coco

```- `dfine-n`: 最軽量モデル（デフォルト）

- `dfine-s`: 小型モデル

#### オプション- `dfine-m`: 中型モデル

- `dfine-l`: 大型モデル

```bash- `dfine-x`: 最大モデル

python inference_transformers.py \

    --model ustc-community/dfine_x_coco \異なるモデルを使用する場合は、`download_model.py` の `model_name` を変更してください。

    --image your_image.jpg \

    --output ./results \## 必要な依存パッケージ

    --threshold 0.5 \

    --device cuda```

```torch>=2.0.0

torchvision>=0.15.0

**オプション一覧:**opencv-python>=4.8.0

- `--model, -m`: モデル名（デフォルト: `ustc-community/dfine_n_coco`）pillow>=10.0.0

- `--image, -i`: 入力画像のパス（指定しない場合はサンプル画像を使用）huggingface_hub>=0.16.0  # ダウンロード時のみ

- `--output, -o`: 出力ディレクトリ（デフォルト: `./output`）numpy>=1.24.0

- `--threshold, -t`: 検出の信頼度閾値（デフォルト: 0.5）```

- `--device, -d`: 使用デバイス（`cuda`/`cpu`、デフォルト: 自動選択）

## 注意事項

### 利用可能なモデル一覧

- このスクリプトは最小構成の実装です

| モデル名 | パラメータ数 | AP (COCO) | 特徴 |- 実際のD-FINEモデルの完全な推論には、公式リポジトリ（https://github.com/Peterande/D-FINE）の実装が必要です

|---------|------------|-----------|------|- GPU利用可能な場合は自動的にGPUを使用します

| `ustc-community/dfine_n_coco` | 4M | 42.8% | 最軽量・高速 |- CPU環境でも動作しますが、推論速度が遅くなります

| `ustc-community/dfine_s_coco` | 10M | 48.5% | 軽量 |

| `ustc-community/dfine_m_coco` | 19M | 52.4% | 中型 |## ライセンス

| `ustc-community/dfine_l_coco` | 31M | 54.0% | 大型 |

| `ustc-community/dfine_x_coco` | 62M | 55.8% | 最高精度 |D-FINEモデルのライセンスに従います。


### Transformers版の利点

✅ **簡単**: `pip install transformers` だけで使える  
✅ **公式サポート**: Hugging Faceの公式実装  
✅ **自動ダウンロード**: モデルは初回実行時に自動的にダウンロード  
✅ **標準API**: 他のTransformersモデルと同じAPIで使える  
✅ **最新**: 常に最新版が利用可能  

---

## 📦 方法2: オリジナル版

### ステップ1: モデルのダウンロード（オンライン環境）

```bash
# 必要なパッケージをインストール
pip install -r requirements.txt

# モデルをダウンロード
python download_model.py
```

これにより `weights/` ディレクトリにモデルファイルがダウンロードされます。

### ステップ2: モデルをオフライン環境に転送

```bash
# weightsディレクトリをオフライン環境にコピー
# 例: USBメモリ、ネットワーク共有、など
```

### ステップ3: 推論の実行（オフライン環境）

```bash
# 推論を実行（サンプル画像が自動生成されます）
python inference.py
```

独自の画像で推論する場合:
```bash
# sample_input.jpg を用意してから実行
python inference.py
```

---

## 🖼️ 出力

実行すると以下のファイルが生成されます：

- `output/result_*.jpg`: 検出結果が描画された画像
- コンソールに検出されたオブジェクトの一覧が表示されます

### 出力例

```
検出結果 (閾値: 0.5):
------------------------------------------------------------
検出数: 5
  cat: 0.958 [344.49, 23.4, 639.84, 374.27]
  cat: 0.956 [11.71, 53.52, 316.64, 472.33]
  remote: 0.947 [40.46, 73.7, 175.62, 117.57]
  sofa: 0.918 [0.59, 1.88, 640.25, 474.74]
  remote: 0.895 [333.48, 77.04, 370.77, 187.3]
```

---

## 📋 構成ファイル

```
predict/
├── inference_transformers.py  # Transformers版推論スクリプト（推奨）
├── download_model.py         # モデルダウンロードスクリプト（オリジナル版用）
├── inference.py              # 推論スクリプト（オリジナル版）
├── requirements_transformers.txt  # Transformers版の依存パッケージ
├── requirements.txt          # オリジナル版の依存パッケージ
└── README.md                # このファイル
```

---

## 🔧 動作環境

- Python 3.8以上
- PyTorch 2.0以上
- CUDA対応GPU（オプション、推奨）

**GPU使用時:**
- CUDA 11.7以上推奨
- VRAM: 最低2GB（dfine-nの場合）

**CPU使用時:**
- 動作しますが推論速度が遅くなります

---

## 💡 ヒント

### 検出精度を上げたい場合
- より大きなモデル（dfine_x_coco）を使用
- `--threshold` の値を下げる（例: 0.3）

### 実行速度を上げたい場合
- より小さなモデル（dfine_n_coco）を使用
- GPUを使用
- `--threshold` の値を上げる（例: 0.7）

### カスタムデータセットで学習したい場合
- 公式リポジトリを参照: https://github.com/Peterande/D-FINE

---

## 📚 参考資料

- **Hugging Face モデルページ**: https://huggingface.co/ustc-community
- **公式リポジトリ**: https://github.com/Peterande/D-FINE
- **論文**: [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)
- **Transformers ドキュメント**: https://huggingface.co/docs/transformers/model_doc/d_fine

---

## 🐛 トラブルシューティング

### モデルのダウンロードが失敗する

```bash
# プロキシを設定している場合
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### CUDAエラーが発生する

```bash
# CPUモードで実行
python inference_transformers.py --device cpu
```

### メモリ不足エラー

- より小さなモデルを使用してください（dfine_n_coco）
- CPUモードを試してください

---

## 📄 ライセンス

D-FINEモデルのライセンスに従います。

## 🙏 クレジット

- **D-FINE**: [Peterande/D-FINE](https://github.com/Peterande/D-FINE)
- **Hugging Face Transformers**: [huggingface/transformers](https://github.com/huggingface/transformers)
