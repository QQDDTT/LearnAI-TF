# LearnAI ディープラーニングトレーニングフレームワーク

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**設定駆動 · モジュール設計 · マルチモード学習 · 本番環境対応**

[クイックスタート](#クイックスタート) • [機能](#機能) • [ドキュメント](#ドキュメント) • [サンプル](#サンプル) • [貢献](#貢献)

</div>

---

## 📖 目次

- [紹介](#紹介)
- [コア機能](#コア機能)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [クイックスタート](#クイックスタート)
- [インストール](#インストール)
- [使用ガイド](#使用ガイド)
- [設定ファイル](#設定ファイル)
- [サポートされているトレーニングモード](#サポートされているトレーニングモード)
- [モデルのエクスポートとデプロイ](#モデルのエクスポートとデプロイ)
- [プロジェクト構造](#プロジェクト構造)
- [開発ガイド](#開発ガイド)
- [よくある質問](#よくある質問)
- [更新履歴](#更新履歴)
- [貢献](#貢献)
- [ライセンス](#ライセンス)

---

## 紹介

**LearnAI**は、TensorFlow 2.xベースのエンタープライズグレードのディープラーニングトレーニングフレームワークです。設定駆動型のアーキテクチャを採用し、教師あり学習、強化学習、自己教師あり学習など、複数のトレーニングパラダイムをサポートしています。フレームワークはYAML設定ファイルを通じて全てのトレーニングプロセスを定義し、コード修正なしで複雑なディープラーニングタスクを完了できます。

### なぜLearnAIを選ぶのか？

- ✅ **ゼロコードトレーニング**: YAML設定のみでモデルトレーニングを完了
- ✅ **マルチモードサポート**: 7つの主流トレーニングパラダイムをサポート
- ✅ **本番環境対応**: モデルエクスポート、デプロイ、監視機能を内蔵
- ✅ **高い柔軟性**: リフレクション機構により任意のPythonコンポーネントを動的にロード
- ✅ **エンタープライズグレード品質**: 完全なテストカバレッジ、ログシステム、エラー処理

---

## コア機能

### 🎯 設定駆動型アーキテクチャ

```yaml
# 1つの設定ファイルで完全なトレーニングプロセスを定義
global:
  name: "image_classifier"
  version: "v1.0.0"

training_mode:
  type: "supervised"

models:
  classifier:
    # リフレクション機構を使用してモデルを動的にロード
    reflection: "tensorflow.keras.Sequential"
    layers:
      - name: "conv1"
        reflection: "tensorflow.keras.layers.Conv2D"
        args: {filters: 32, kernel_size: [3,3]}
```

### 🚀 サポートされている機能

| 機能カテゴリ | サポート内容 |
|------------|------------|
| **トレーニングモード** | 教師あり、強化学習、教師なし、自己教師あり、半教師あり、マルチタスク、カスタム |
| **データソース** | CSV、NumPy、画像ディレクトリ、TFRecord、ネットワークAPI、カスタムローダー |
| **モデルアーキテクチャ** | Keras Sequential、Functional API、Model Subclassing、事前学習済みモデル、カスタムモデル |
| **オプティマイザ** | Adam、SGD、RMSprop、AdaGrad等すべてのTensorFlowオプティマイザ + 学習率スケジューリング |
| **損失関数** | TensorFlow組み込み損失 + カスタム損失（対照損失、Focal Loss、マルチタスク損失等） |
| **エクスポート形式** | SavedModel、ONNX、TensorFlow Lite、H5、ウェイトのみ |
| **デプロイ方法** | REST API、gRPC、TensorFlow Serving、Docker、カスタムデプロイ |

### 🛠️ リフレクション機構

`reflection`フィールドを通じて任意のPythonクラスまたは関数を動的に呼び出し:

```yaml
# TensorFlowコンポーネントの呼び出し
reflection: "tensorflow.keras.optimizers.Adam"

# カスタム関数の呼び出し
reflection: "modules.custom:my_training_function"

# サードパーティライブラリの呼び出し
reflection: "sklearn.preprocessing.StandardScaler"
```

### 📊 高度なトレーニングフロー制御

条件制御のためのBridge式:

```yaml
step_sequence:
  - name: "validation"
    reflection: "modules.evaluation:validate"
    bridge: "@skip:validation?${epoch}%10!=0"  # 10エポックごとに検証

  - name: "early_stop"
    reflection: "common.utils:check_convergence"
    bridge: "@jump:save_model?${accuracy}>0.95"  # 目標達成時にジャンプ
```

---

## システムアーキテクチャ

### 設計理念

```
┌─────────────────────────────────────────────────────────┐
│                   YAML設定ファイル                        │
│        (唯一の制御センター、全ての動作を定義)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ (ロードと検証)
┌─────────────────────────────────────────────────────────┐
│                    main.py                               │
│           (メインエントリーポイント、モジュール調整)        │
└────┬────────┬──────────┬──────────┬──────────┬──────────┘
     │        │          │          │          │
     ↓        ↓          ↓          ↓          ↓
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Data    │ Models  │Optimizer│ Losses  │Training │
│ Manager │ Builder │ Manager │ Manager │Pipeline │
└─────────┴─────────┴─────────┴─────────┴─────────┘
     ↓        ↓          ↓          ↓          ↓
┌──────────────────────────────────────────────────┐
│      common/utils.py (ユーティリティライブラリ)    │
│   • forward()  • compute_loss()  • backward()    │
└──────────────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────────────┐
│     TensorFlow 2.x / サードパーティライブラリ      │
└──────────────────────────────────────────────────┘
```

### モジュールの責任

| モジュール | ファイル | 責任 |
|----------|---------|-----|
| **メインコントローラー** | `main.py` | 設定読み込み、実行調整、ライフサイクル管理 |
| **データ管理** | `modules/data_manager.py` | データロード、前処理、拡張 |
| **モデル構築** | `modules/models.py` | モデル作成、レイヤー定義、アーキテクチャ管理 |
| **オプティマイザ管理** | `modules/optimizers.py` | オプティマイザ設定、学習率スケジューリング |
| **損失関数** | `modules/losses.py` | 損失計算、カスタム損失 |
| **トレーニングパイプライン** | `modules/training_pipeline.py` | トレーニングループ、Bridge制御、チェックポイント |
| **モデル評価** | `modules/evaluation.py` | メトリクス計算、モデル検証 |
| **モデルエクスポート** | `modules/export.py` | マルチフォーマットエクスポート、モデル最適化 |
| **モデルデプロイ** | `modules/deployment.py` | サービスデプロイ、API作成 |
| **共通ユーティリティ** | `common/common.py` | ログ、リフレクション、設定ロード |
| **トレーニングコンテキスト** | `common/train_context.py` | 状態管理、変数ストレージ |

---

## クイックスタート

### 前提条件

- Python 3.9+
- TensorFlow 2.16.1
- 4GB以上のRAM（8GB以上推奨）
- （オプション）CUDA 12.3対応のNVIDIA GPU

### 30秒クイック体験

```bash
# 1. リポジトリのクローン
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# 2. 依存関係のインストール
pip install -r requirements.txt

# 3. サンプルの実行
python main.py config/config_example.yaml
```

### 完全な例: 画像分類

```bash
# 1. データの準備
mkdir -p data/train data/val
# 画像を各ディレクトリに配置

# 2. 設定ファイルの作成
cat > config/my_classifier.yaml << 'EOF'
global:
  name: "my_image_classifier"
  version: "v1.0.0"
  seed: 42

training_mode:
  type: "supervised"

models:
  classifier:
    reflection: "tensorflow.keras.Sequential"
    layers:
      - name: "conv1"
        reflection: "tensorflow.keras.layers.Conv2D"
        args: {filters: 32, kernel_size: [3,3], activation: "relu"}
      - name: "pool1"
        reflection: "tensorflow.keras.layers.MaxPooling2D"
        args: {pool_size: [2,2]}
      - name: "flatten"
        reflection: "tensorflow.keras.layers.Flatten"
      - name: "dense1"
        reflection: "tensorflow.keras.layers.Dense"
        args: {units: 128, activation: "relu"}
      - name: "output"
        reflection: "tensorflow.keras.layers.Dense"
        args: {units: 10, activation: "softmax"}

data_manager:
  supervised_source:
    train:
      reflection: "tensorflow.keras.preprocessing.image_dataset_from_directory"
      args:
        directory: "data/train"
        image_size: [128, 128]
        batch_size: 32
        label_mode: "categorical"

optimizers:
  main_optimizer:
    reflection: "tensorflow.keras.optimizers.Adam"
    args:
      learning_rate: 0.001

losses:
  classification_loss:
    reflection: "tensorflow.keras.losses.CategoricalCrossentropy"

training_pipeline:
  supervised:
    loop_config:
      type: "epoch_batch"
      parameters:
        epochs: 50
        batch_size: 32
    step_sequence:
      - name: "forward"
        reflection: "common.utils:forward"
        args: {model: "${classifier}", inputs: "${batch_data}"}
      - name: "loss"
        reflection: "common.utils:compute_loss"
        args: {loss_fn: "${classification_loss}", predictions: "${forward}", targets: "${batch_labels}"}
      - name: "backward"
        reflection: "common.utils:compute_gradients"
        args: {loss: "${loss}", model: "${classifier}"}
      - name: "update"
        reflection: "common.utils:apply_gradients"
        args: {optimizer: "${main_optimizer}", gradients: "${backward}", model: "${classifier}"}

evaluation:
  supervised_eval:
    reflection: "modules.evaluation:evaluate_supervised"
    args:
      model: "${classifier}"
      dataset: "${val_data}"
      metrics: ["accuracy", "precision", "recall"]

export:
  export_onnx:
    model: "${classifier}"
    format: "onnx"
    output_path: "outputs/onnx/classifier.onnx"
EOF

# 3. トレーニング開始
python main.py config/my_classifier.yaml --export

# 4. 結果の確認
ls outputs/onnx/
```

---

## インストール

### クイックインストール（最小依存関係）

```bash
pip install tensorflow numpy pandas pyyaml colorama requests
```

### 完全インストール（全機能）

```bash
pip install -r requirements.txt
```

### GPUサポート

```bash
# CUDAサポート付きTensorFlow
pip install tensorflow[and-cuda]==2.16.1

# CUDA 12.3とcuDNN 8.9が必要
```

### 中国ミラー加速

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 開発環境

```bash
# リポジトリのクローン
git clone https://github.com/qqddtt/LearnAI.git
cd LearnAI

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 開発依存関係のインストール
pip install -r requirements.txt
pip install -r requirements-dev.txt  # テストとコード品質ツールを含む

# テストの実行
pytest test/
```

---

## 使用ガイド

### コマンドライン引数

```bash
python main.py <config_file> [options]

必須:
  config_file          設定ファイルパス（YAML）

オプション:
  --export             トレーニング後にモデルをエクスポート
  --deploy             エクスポート後にモデルをデプロイ
  --deploy-only        デプロイのみ、トレーニングをスキップ
  --skip-eval          評価フェーズをスキップ
  --checkpoint-dir     チェックポイントディレクトリ
  --verbose            詳細ログ出力
  --dry-run            設定の検証のみ、トレーニングを実行しない
```

### 典型的なワークフロー

#### 1. 設定の検証

```bash
# 設定ファイルの検証
python main.py config/my_config.yaml --dry-run
```

#### 2. モデルのトレーニング

```bash
# 標準トレーニング
python main.py config/my_config.yaml

# トレーニング + エクスポート
python main.py config/my_config.yaml --export

# トレーニング + エクスポート + デプロイ
python main.py config/my_config.yaml --export --deploy
```

#### 3. 既存モデルのデプロイのみ

```bash
python main.py config/my_config.yaml --deploy-only
```

#### 4. デバッグモード

```bash
python main.py config/my_config.yaml --verbose
```

---

## 設定ファイル

### 設定ファイル構造

詳細な設定ドキュメントについては、[設定ファイル構造ドキュメント](docs/配置文件結構説明文档.md)を参照してください。

```yaml
global:           # グローバル設定（プロジェクト名、バージョン、ランダムシード）
training_mode:    # トレーニングモード（supervised/reinforcement/unsupervised等）
models:           # モデル定義
data_manager:     # データ管理
optimizers:       # オプティマイザ設定
losses:           # 損失関数設定
training_pipeline:# トレーニングプロセス
evaluation:       # 評価設定（オプション）
export:           # モデルエクスポート設定（オプション）
deployment:       # モデルデプロイ設定（オプション）
```

### 設定テンプレート生成

```python
from common.common import generate_config_template

# 教師あり学習テンプレートの生成
generate_config_template("supervised", "my_supervised_config.yaml")

# 強化学習テンプレートの生成
generate_config_template("reinforcement", "my_rl_config.yaml")
```

---

## サポートされているトレーニングモード

### 1. 教師あり学習

画像分類、物体検出、テキスト分類等に使用。

**サンプル設定:** `config/supervised_config.yaml`

**特徴:**
- エポック + バッチトレーニングループ
- ラベル付きデータ
- クロスエントロピー損失

### 2. 強化学習

ゲームAI、ロボット制御、自動運転等に使用。

**サンプル設定:** `config/reinforcement_config.yaml`

**特徴:**
- エピソード + ステップトレーニングループ
- 報酬シグナル
- 方策勾配/Q学習

### 3. 教師なし学習

クラスタリング、次元削減、異常検出等に使用。

**特徴:**
- 反復的トレーニング
- ラベルなしデータ
- 再構成/クラスタリング損失

### 4. 自己教師あり学習

対照学習、マスク予測、画像修復等に使用。

**サンプル設定:** `config/self_supervised_config.yaml`

**特徴:**
- 対照損失
- データ拡張
- 事前学習 + ファインチューニング

### 5. 半教師あり学習

限られたラベル付きデータのシナリオに使用。

**特徴:**
- ラベル付き + ラベルなしデータ
- 疑似ラベル
- 一貫性正則化

### 6. マルチタスク学習

複数の関連タスクを同時にトレーニング。

**特徴:**
- 共有エンコーダー
- 複数のタスクヘッド
- 重み付き損失

### 7. カスタムトレーニング

完全にカスタマイズされたトレーニングロジック。

**特徴:**
- 自由なループ定義
- カスタムステップシーケンス
- Bridge制御フロー

---

## モデルのエクスポートとデプロイ

### サポートされているエクスポート形式

| 形式 | 用途 | ファイル拡張子 |
|------|------|--------------|
| **SavedModel** | TensorFlow Serving本番環境 | ディレクトリ構造 |
| **ONNX** | クロスプラットフォームデプロイ（マルチフレームワーク） | `.onnx` |
| **TensorFlow Lite** | モバイルおよび組み込みデバイス | `.tflite` |
| **H5** | Keras標準形式 | `.h5` |
| **Weights Only** | ウェイトのみ保存 | `.weights` |

### エクスポート例

```yaml
export:
  # SavedModel形式（TensorFlow Serving）
  export_savedmodel:
    model: "${classifier}"
    format: "savedmodel"
    output_path: "outputs/savedmodel/classifier"

  # ONNX形式（クロスプラットフォーム）
  export_onnx:
    model: "${classifier}"
    format: "onnx"
    output_path: "outputs/onnx/classifier.onnx"
    args:
      opset_version: 13

  # TFLite形式（モバイル）
  export_tflite:
    model: "${classifier}"
    format: "tflite"
    output_path: "outputs/tflite/classifier.tflite"
    args:
      optimizations: ["DEFAULT"]
```

### サポートされているデプロイ方法

| 方法 | 説明 | 使用ケース |
|-----|------|----------|
| **REST API** | Flaskサーバー | Webアプリケーション統合 |
| **gRPC** | 高性能RPC | マイクロサービスアーキテクチャ |
| **TensorFlow Serving** | 公式モデルサービング | 本番環境 |
| **Docker** | コンテナ化デプロイ | クラウドプラットフォーム |
| **カスタム** | リフレクションベースのカスタム関数 | 特殊要件 |

### デプロイ例

```yaml
deployment:
  # REST APIデプロイ
  rest_api:
    type: "rest_api"
    model_path: "${export_paths.classifier}"
    host: "0.0.0.0"
    port: 9000
    endpoints:
      predict: "/api/predict"
      health: "/health"
    performance:
      batch_size: 32
      timeout: 30
      workers: 4
```

### サービスのクイック起動

```bash
# トレーニングとデプロイ
python main.py config/my_config.yaml --export --deploy

# 既存モデルのデプロイのみ
python main.py config/my_config.yaml --deploy-only
```

---

## プロジェクト構造

```
LearnAI/
├── main.py                          # メインエントリーポイント
├── requirements.txt                 # 依存関係リスト
├── README.md                        # このファイル
│
├── config/                          # 設定ディレクトリ
│   ├── config_example.yaml         # サンプル設定
│   ├── supervised_config.yaml      # 教師あり学習
│   ├── reinforcement_config.yaml   # 強化学習
│   └── self_supervised_config.yaml # 自己教師あり学習
│
├── common/                          # 共通モジュール
│   ├── __init__.py
│   ├── common.py                   # 基本機能（ログ、リフレクション、設定）
│   ├── utils.py                    # ユーティリティ関数
│   ├── train_context.py            # トレーニングコンテキスト
│   ├── interfaces.py               # インターフェース定義
│   ├── config_validator.py         # 設定検証
│   └── validators/                 # 検証サブモジュール
│
├── modules/                         # コアモジュール
│   ├── __init__.py
│   ├── data_manager.py             # データ管理
│   ├── models.py                   # モデル構築
│   ├── optimizers.py               # オプティマイザ管理
│   ├── losses.py                   # 損失関数
│   ├── training_pipeline.py        # トレーニングパイプライン
│   ├── evaluation.py               # モデル評価
│   ├── export.py                   # モデルエクスポート
│   └── deployment.py               # モデルデプロイ
│
├── lib/                             # サードパーティライブラリラッパー
│   ├── __init__.py
│   └── deployment.py               # デプロイツール（Flask/gRPC）
│
├── data/                            # データディレクトリ
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                     # チェックポイントディレクトリ
├── logs/                           # ログディレクトリ
├── outputs/                        # 出力ディレクトリ
│   ├── onnx/
│   ├── savedmodel/
│   ├── tflite/
│   └── h5/
│
├── test/                           # テストコード
│   ├── test_config.py
│   ├── test_modules.py
│   └── config_test.yaml
│
└── docs/                           # ドキュメント
    ├── Configuration_Structure_JA.md
    └── AI_Platform_Roadmap_JA.md
```

---

## 開発ガイド

### カスタムコンポーネントの追加

#### 1. カスタムデータローダー

`common/utils.py`または`modules/data_manager.py`に追加:

```python
def load_my_custom_data(file_path: str, batch_size: int = 32):
    """カスタムデータローディング関数"""
    # データローディングロジックの実装
    dataset = ...
    return dataset
```

設定で使用:

```yaml
data_manager:
  custom_source:
    train:
      reflection: "common.utils:load_my_custom_data"
      args:
        file_path: "data/my_data.txt"
        batch_size: 32
```

#### 2. カスタム損失関数

`modules/losses.py`に追加:

```python
import tensorflow as tf

class MyCustomLoss(tf.keras.losses.Loss):
    """カスタム損失関数"""

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # 損失計算の実装
        loss = ...
        return loss
```

設定で使用:

```yaml
losses:
  custom_loss:
    reflection: "modules.losses:MyCustomLoss"
    args:
      alpha: 1.5
```

### コード規約

このプロジェクトはPEP 8スタイルガイドに従います。

```bash
# コードフォーマット
black .

# コードリント
flake8 .

# 型チェック
mypy .
```

### テスト

```bash
# 全テストの実行
pytest test/

# 特定のテスト実行
pytest test/test_modules.py::TestConfigLoading

# テストカバレッジの確認
pytest --cov=. --cov-report=html
```

---

## よくある質問

### Q1: 設定ファイルをデバッグするには？

**A:** `--dry-run`オプションで設定を検証:

```bash
python main.py config.yaml --dry-run
```

### Q2: マルチGPUトレーニングを使用するには？

**A:** 設定で分散戦略を設定:

```yaml
global:
  distributed_strategy:
    type: "MirroredStrategy"
    devices: ["GPU:0", "GPU:1"]
```

### Q3: モデルのホットリロードを実装するには？

**A:** デプロイ設定で自動リロードを有効化:

```yaml
deployment:
  rest_api:
    auto_reload: true
    reload_interval: 60  # 60秒ごとにチェック
```

### Q4: トレーニング中にチェックポイントを保存するには？

**A:** トレーニングパイプラインに保存ステップを追加:

```yaml
training_pipeline:
  supervised:
    step_sequence:
      # ... トレーニングステップ

      - name: "save_checkpoint"
        reflection: "common.utils:save_checkpoint"
        args:
          model: "${main_model}"
          epoch: "${current_epoch}"
        bridge: "@skip:save_checkpoint?${epoch}%5!=0"  # 5エポックごとに保存
```

### Q5: 独自のディープラーニングフレームワークを統合するには？

**A:** リフレクション機構で任意のPythonライブラリを呼び出し:

```yaml
models:
  my_pytorch_model:
    reflection: "my_pytorch_module:MyModel"
    args:
      input_dim: 784
      output_dim: 10
```

### Q6: 不均衡なデータセットを処理するには？

**A:** クラス重みまたはリサンプリングを使用:

```yaml
losses:
  weighted_loss:
    reflection: "tensorflow.keras.losses.CategoricalCrossentropy"

training_pipeline:
  supervised:
    parameters:
      class_weight: {0: 1.0, 1: 2.0, 2: 3.0}  # 少数クラスの重みを増加
```

### Q7: 早期停止を実装するには？

**A:** Bridge条件制御を使用:

```yaml
step_sequence:
  - name: "check_early_stop"
    reflection: "common.utils:check_convergence"
    args:
      metric: "${val_loss}"
      patience: 10
    bridge: "@jump:save_and_exit?${converged}==true"
```

---

## 更新履歴

### v2.0.0 (2025-11-03)

#### 新機能
- ✨ 完全な設定検証システム（マルチレイヤーバリデーター）
- ✨ Bridge式のサポート（条件制御、ループ、分岐）
- ✨ モデルエクスポート5形式サポート（SavedModel/ONNX/TFLite/H5/Weights）
- ✨ モデルデプロイ5方法サポート（REST API/gRPC/TF Serving/Docker/Custom）
- ✨ 統一されたトレーニングコンテキスト管理（TrainContext）
- ✨ 完全なテストスイート（37テストケース）

#### 改善
- 🔧 明確な責任を持つモジュールアーキテクチャのリファクタリング
- 🔧 カラー出力とファイルログ付きの改善されたログシステム
- 🔧 柔軟なパラメータ渡しのための最適化されたリフレクション機構
- 🔧 強化されたエラー処理と例外情報

#### ドキュメント
- 📚 設定ファイル構造ドキュメントの追加
- 📚 AI駆動プラットフォーム進化ロードマップの追加
- 📚 READMEとAPIドキュメントの改善

### v1.0.0 (2025-10-15)

#### 初回リリース
- 🎉 基本的な設定駆動型アーキテクチャ
- 🎉 教師あり、強化、自己教師あり学習のサポート
- 🎉 リフレクション機構の実装
- 🎉 基本的なモデルエクスポート機能

---

## 貢献

あらゆる形式の貢献を歓迎します！

### 貢献方法

1. **プロジェクトをフォーク**

```bash
git clone https://github.com/your-username/LearnAI.git
```

2. **機能ブランチを作成**

```bash
git checkout -b feature/your-feature-name
```

3. **変更をコミット**

```bash
git commit -m "Add: 機能追加"
```

4. **ブランチにプッシュ**

```bash
git push origin feature/your-feature-name
```

5. **プルリクエストを作成**

GitHubでPRを作成し、変更内容を説明。

### コミットメッセージ規約

```
Add: 新機能
Fix: バグ修正
Docs: ドキュメント更新
Style: コードフォーマット
Refactor: コードリファクタリング
Test: テスト関連
Chore: ビルドまたは補助ツールの変更
```

### コードレビュー基準

- ✅ PEP 8コードスタイルに従う
- ✅ 必要なテストを追加
- ✅ 関連ドキュメントを更新
- ✅ すべてのテストが合格することを確認
- ✅ 明確なコメントを追加

---

## ロードマップ

### 短期計画（3-6ヶ月）

- [ ] マイクロサービスアーキテクチャのリファクタリング
- [ ] 分散トレーニングのサポート（Horovod/Ray）
- [ ] Web UIコンソール
- [ ] 実験追跡システム（MLflow統合）
- [ ] コンテナ化デプロイ（Docker + Kubernetes）

### 中期計画（6-12ヶ月）

- [ ] AutoML機能（NAS + ハイパーパラメータ最適化）
- [ ] インテリジェントデータ生成（GAN/Diffusion）
- [ ] モデル圧縮と量子化
- [ ] パフォーマンス予測器
- [ ] A/Bテストサポート

### 長期ビジョン（12-24ヶ月）

- [ ] LLM駆動の設定生成
- [ ] 強化学習自動調整
- [ ] 自律的タスク発見
- [ ] 完全自律的トレーニングシステム

詳細: [AI駆動自動化ML プラットフォーム進化ロードマップ](docs/AI駆動的自動化機器学習平台演進路線図.md)

---

## コミュニティとサポート

### ヘルプを得る

- 📖 [ドキュメント](docs/)
- 💬 [GitHub Discussions](https://github.com/qqddtt/LearnAI/discussions)
- 🐛 [イシュートラッカー](https://github.com/qqddtt/LearnAI/issues)
- 📧 Email: support@learnai.org

### コミュニティに参加

- ⭐ プロジェクトにスターを付ける
- 🐛 バグを報告
- 💡 新機能を提案
- 📝 ドキュメントを改善
- 🤝 プルリクエストを送信

---

## ライセンス

このプロジェクトは**MITライセンス**の下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

```
MIT License

Copyright (c) 2025 LearnAI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 謝辞

以下のオープンソースプロジェクトと貢献者に感謝します：

- [TensorFlow](https://www.tensorflow.org/) - ディープラーニングフレームワーク
- [Keras](https://keras.io/) - 高レベルAPI
- [ONNX](https://onnx.ai/) - モデル交換フォーマット
- [Flask](https://flask.palletsprojects.com/) - Webフレームワーク
- [Ray](https://www.ray.io/) - 分散コンピューティングフレームワーク
- [MLflow](https://mlflow.org/) - 実験追跡システム

このプロジェクトに貢献したすべての開発者に感謝します！

---

## 引用

研究でLearnAIを使用する場合は、以下を引用してください：

```bibtex
@software{learnai2025,
  title = {LearnAI: A Configuration-Driven Deep Learning Training Framework},
  author = {LearnAI Team},
  year = {2025},
  url = {https://github.com/qqddtt/LearnAI}
}
```

---

<div align="center">

**⭐ このプロジェクトが役立つ場合は、スターを付けてください！ ⭐**

Made with ❤️ by [LearnAI Team](https://github.com/qqddtt)

[トップに戻る](#learnai-ディープラーニングトレーニングフレームワーク)

</div>
