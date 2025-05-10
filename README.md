# 中古車価格予測AIアプリケーション

## 概要

このアプリケーションは、中古車の特徴（年式、走行距離、燃料タイプなど）を入力すると、機械学習モデル（ランダムフォレスト）を用いてその販売価格を予測するWebアプリケーションです。

データ分析からモデル構築、API開発、フロントエンド連携までの一連の流れを実装しています。

## 技術スタック

* **フロントエンド:**
* Next.js (v15+) / React (v19+)
  * TypeScript
  * CSS Modules
  * Zod (フォームバリデーション)
  * React `useActionState` / Next.js Server Actions (フォーム処理・API連携)
* **バックエンド (API):**
  * Python (v3.x)
  * FastAPI (Webフレームワーク)
  * Uvicorn (ASGIサーバー)
  * Pandas (データ操作)
  * Scikit-learn (機械学習モデル)
  * Joblib (モデル/特徴量リストの保存・読み込み)
  * Pydantic (データバリデーション)
* **データセット:**
  * Kaggle: [Vehicle dataset from Cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`car data.csv`)

## セットアップとインストール

### 1. 前提条件

* Node.js (v18 またはそれ以降)
* npm または yarn
* Python (v3.8 またはそれ以降)
* pip (Python パッケージインストーラー)

### 2. リポジトリのクローン

```bash
git clone <リポジトリのURL>
cd <リポジトリ名> # 例: cd used-car-price-ai-forecaster
```

### 3. フロントエンドのセットアップ

プロジェクトルートディレクトリ (`used-car-price-ai-forecaster`) で以下を実行します。

```bash
npm install
# または
# yarn install
```

### 4. バックエンド (Python API) のセットアップ

バックエンドのコードは `api` ディレクトリにあります。

```bash
# 1. API ディレクトリに移動
cd api

# 2. Python 仮想環境の作成
python3 -m venv .venv

# 3. 仮想環境の有効化
#    - macOS / Linux (bash/zsh):
source .venv/bin/activate
#    - Windows (Command Prompt):
#      .\.venv\Scripts\activate.bat
#    - Windows (PowerShell):
#      .\.venv\Scripts\Activate.ps1
#    (成功するとプロンプトの先頭に (.venv) が表示されます)

# 4. 必要な Python ライブラリのインストール
pip3 install -r requirements.txt
# (もし pip3 で command not found になる場合は pip install -r requirements.txt を試してください)
```

### 5. データセットの準備

1. Kaggle から [Vehicle dataset from Cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) データセット (`car data.csv`) をダウンロードします。
2. ダウンロードした `car data.csv` ファイルを `api/data/` ディレクトリ内に配置します (`data` ディレクトリがない場合は作成してください)。

### 6. 機械学習モデルの学習

バックエンド API を起動する前に、モデルファイルを生成する必要があります。

1. `api` ディレクトリにいることを確認します。
2. **仮想環境が有効になっていることを確認**します (`(.venv)` が表示されている)。
3. 以下のコマンドを実行してモデルを学習させ、`api/model/` ディレクトリにファイルを保存します。

    ```bash
    python3 train_model.py
    ```

## アプリケーションの起動

### 1. バックエンド (FastAPI) サーバーの起動

1. `api` ディレクトリに移動します (`cd api`)。
2. **仮想環境を有効化**します (`source .venv/bin/activate`)。
3. 以下のコマンドで FastAPI サーバーを起動します。

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

    サーバーは `http://localhost:8000` で起動します。API ドキュメントは `http://localhost:8000/docs` で確認できます。

### 2. フロントエンド (Next.js) 開発サーバーの起動

1. プロジェクトの**ルートディレクトリ** (`used-car-price-ai-forecaster`) に移動します (`cd ..` など)。
2. 以下のコマンドで Next.js 開発サーバーを起動します。

    ```bash
    npm run dev
    # または
    # yarn dev
    ```

    アプリケーションは `http://localhost:3000` でアクセス可能になります。

ブラウザで `http://localhost:3000` を開き、フォームに情報を入力して価格予測をお試しください。

## ディレクトリ構造 (主要部分)

```
.
├── api/                  # Python バックエンド (FastAPI)
│   ├── data/             # データセット格納用 (car data.csv を配置)
│   ├── model/            # 学習済みモデル格納用 (train_model.py により生成)
│   ├── .venv/            # Python 仮想環境
│   ├── main.py           # FastAPI アプリケーション本体
│   ├── train_model.py    # モデル学習用スクリプト
│   └── requirements.txt  # Python 依存ライブラリ
├── src/                  # Next.js フロントエンド
│   ├── app/              # App Router 関連
│   │   ├── components/   # React コンポーネント
│   │   ├── actions.ts    # Next.js Server Actions
│   │   └── page.tsx      # トップページ
│   └── ...
├── public/               # 静的ファイル
├── node_modules/         # Node.js 依存ライブラリ
├── package.json          # Node.js プロジェクト設定
├── next.config.mjs       # Next.js 設定
├── tsconfig.json         # TypeScript 設定
└── README.md             # このファイル
```
