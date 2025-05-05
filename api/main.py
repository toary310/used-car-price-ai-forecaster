import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import os

# --- 定数 ---
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'car_price_model.joblib')
FEATURES_FILE = os.path.join(MODEL_DIR, 'model_features.joblib')

# --- モデルと特徴量リストの読み込み ---
try:
    model = joblib.load(MODEL_FILE)
    model_features = joblib.load(FEATURES_FILE)
    print("Model and features loaded successfully.")
    print(f"Expected features: {model_features}")
except FileNotFoundError:
    print(f"Error: Model or features file not found in {MODEL_DIR}. Run train_model.py first.")
    model = None
    model_features = None
except Exception as e:
    print(f"Error loading model or features: {e}")
    model = None
    model_features = None


# --- FastAPI アプリケーションの初期化 ---
app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to predict the selling price of used cars.",
    version="0.1.0"
)

# --- CORS 設定 ---
# フロントエンド (Next.js) からのリクエストを許可する
# 注意: 本番環境では、より厳密なオリジン設定が必要です
origins = [
    "http://localhost:3000", # Next.js の開発サーバー
    # 必要に応じて他のオリジンを追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # すべての HTTP メソッドを許可 (GET, POST など)
    allow_headers=["*"], # すべての HTTP ヘッダーを許可
)

# --- リクエストボディの Pydantic モデル ---
# フロントエンドから送信されるデータ構造を定義
# train_model.py で使用した特徴量 + 年 (Year)
class CarFeaturesInput(BaseModel):
    year: int = Field(..., gt=1979, lt=datetime.now().year + 2, description="Year of the car")
    present_price: float = Field(..., gt=0, description="Current showroom price (in Lakhs)") # データセットにあるが必要性が低いかも？ 学習には使用
    kms_driven: int = Field(..., ge=0, description="Kilometers driven")
    fuel_type: str = Field(..., description="Fuel type (Petrol, Diesel, CNG)") # データセットに合わせる
    seller_type: str = Field(..., description="Seller type (Dealer, Individual)") # データセットに合わせる
    transmission: str = Field(..., description="Transmission type (Manual, Automatic)") # データセットに合わせる
    owner: int = Field(..., ge=0, lt=5, description="Number of previous owners (0, 1, 3)") # データセットに合わせる (フロントエンドからの変換が必要)

    # pydantic v2 から推奨: model_config でサンプルを提供
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "year": 2018,
                    "present_price": 5.59,
                    "kms_driven": 27000,
                    "fuel_type": "Petrol",
                    "seller_type": "Dealer",
                    "transmission": "Manual",
                    "owner": 0
                }
            ]
        }
    }

# --- レスポンスボディの Pydantic モデル ---
class PredictionOutput(BaseModel):
    predicted_price_lakhs: float = Field(..., description="Predicted selling price in Lakhs")

# --- ルートエンドポイント (動作確認用) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

# --- 予測エンドポイント ---
@app.post("/predict", response_model=PredictionOutput)
async def predict_price(features: CarFeaturesInput):
    """
    Predicts the selling price of a used car based on its features.
    Receives car features, preprocesses them, and returns the predicted price.
    """
    if model is None or model_features is None:
        raise HTTPException(status_code=500, detail="Model not loaded. API cannot predict.")

    print(f"Received features for prediction: {features.model_dump()}") # pydantic v2

    # 1. 受け取ったデータを DataFrame に変換
    input_data = pd.DataFrame([features.model_dump()])

    # 2. train_model.py と同様の前処理を実行
    try:
        # 2a. 車齢 (Car_Age) の計算
        current_year = datetime.now().year
        input_data['Car_Age'] = current_year - input_data['year']
        # input_data = input_data.drop('year', axis=1) # year は model_features にないので削除不要

        # 2b. カテゴリ特徴量の One-Hot Encoding (学習時と同じ列を生成)
        # - Pydantic モデルのフィールド名 (snake_case) に合わせる
        categorical_cols_to_encode = ['fuel_type', 'seller_type', 'transmission']
        # - DataFrame 内のカテゴリカル列を明示的に category 型に変換
        for col in categorical_cols_to_encode:
             if col in input_data.columns:
                input_data[col] = input_data[col].astype('category')

        input_processed = pd.get_dummies(input_data, columns=categorical_cols_to_encode, drop_first=True, dtype=int) # drop_first=True を合わせる

        # 2c. 学習済みモデルが期待する特徴量の完全なセットを作成
        #    - 不足している One-Hot 列を 0 で埋める
        #    - 学習時に存在しなかった列を削除 (もしあれば)
        #    - 列の順序を学習時と完全に一致させる
        final_input = pd.DataFrame(columns=model_features) # 学習時の特徴量リストで空のDataFrame作成
        final_input = pd.concat([final_input, input_processed]) # 処理済みデータを結合 (NaN が入る)
        final_input = final_input.fillna(0) # 不足列 (NaN) を 0 で埋める
        final_input = final_input[model_features] # 列の順序を合わせ、余分な列を削除

        print(f"Processed features for prediction (shape: {final_input.shape}):\n{final_input.head()}")

    except Exception as e:
        print(f"Error during preprocessing input data: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing input features: {e}")

    # 3. モデルで予測を実行
    try:
        prediction = model.predict(final_input)
        predicted_price = prediction[0] # 予測結果は配列なので最初の要素を取得
        print(f"Prediction successful: {predicted_price}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    # 注意: データセットの Selling_Price は Lakhs 単位なので、そのまま返す
    return PredictionOutput(predicted_price_lakhs=predicted_price)

# --- Uvicorn でサーバーを起動するためのコマンド (ターミナルで実行) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000