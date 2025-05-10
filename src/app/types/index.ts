// API レスポンスの型定義（新しい形式）
export interface PredictionResult {
  predicted_price_jpy: number;
  lower_bound_jpy: number;
  upper_bound_jpy: number;
  confidence_level: number;
}

// 旧API レスポンスの型定義（互換性のため）
export interface OldPredictionResult {
  predicted_price_lakhs: number;
}

// 車の特徴データの型定義
export interface CarFeatures {
  year: number;
  present_price: number;
  kms_driven: number;
  fuel_type: string;
  seller_type: string;
  transmission: string;
  owner: number;
}
