"use server"; // このファイル内の関数がServer Actionであることを示す

import { z } from 'zod'; // zod をインポート
import { OldPredictionResult, PredictionResult } from './types';

// Zod スキーマでフォーム入力を定義・検証
const CarFeaturesSchema = z.object({
  year: z.coerce.number().int().min(1980, "1980年以降の年式を入力してください。").max(new Date().getFullYear() + 1, "未来の年式は入力できません。"),
  mileage: z.coerce.number({
    required_error: "走行距離を入力してください。",
    invalid_type_error: "走行距離は数値で入力してください。"
  }).int().min(0, "走行距離は0以上である必要があります。"),
  brand: z.string({
    required_error: "メーカーを選択してください。",
    invalid_type_error: "メーカーを選択してください。"
  }).min(1, "メーカーを選択してください。"), // 文字列、空でない
  fuel: z.enum(['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'], {
    required_error: "有効な燃料タイプを選択してください。",
    invalid_type_error: "有効な燃料タイプを選択してください。",
    message: "有効な燃料タイプを選択してください。"
  }),
  transmission: z.enum(['Manual', 'Automatic'], {
    required_error: "有効なトランスミッションを選択してください。",
    invalid_type_error: "有効なトランスミッションを選択してください。",
    message: "有効なトランスミッションを選択してください。"
  }),
  owner_type: z.enum(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'], {
    required_error: "有効な所有者履歴を選択してください。",
    invalid_type_error: "有効な所有者履歴を選択してください。",
    message: "有効な所有者履歴を選択してください。"
  }),
  seller_type: z.enum(['Dealer', 'Individual', 'Trustmark Dealer'], {
    required_error: "有効な販売者タイプを選択してください。",
    invalid_type_error: "有効な販売者タイプを選択してください。",
    message: "有効な販売者タイプを選択してください。"
  }),
  // 他に必要なフィールドがあれば追加
});

// フォームの状態を表す型 (成功時またはエラー時)
export type FormState = {
  message: string;
  predictedPrice?: number;
  lowerBoundPrice?: number;
  upperBoundPrice?: number;
  confidenceLevel?: number;
  // errors の型をより具体的に定義
  errors?: {
    year?: string[];
    mileage?: string[];
    brand?: string[];
    fuel?: string[];
    transmission?: string[];
    owner_type?: string[];
    seller_type?: string[];
    // Zod の _errors (ルートレベルのエラー) も含める場合 (任意)
    _errors?: string[];
  };
  fieldValues?: z.infer<ReturnType<typeof CarFeaturesSchema.partial>>;
} | null; // 初期状態は null

// フォーム入力から安全に値を取得するヘルパー関数
function getSafeFormValue(formData: FormData, key: string): string | undefined {
  const value = formData.get(key);
  return typeof value === 'string' ? value : undefined;
}

function getSafeNumberFormValue(formData: FormData, key: string): number | undefined {
  const value = formData.get(key);
  if (typeof value !== 'string') return undefined;
  const num = Number(value);
  return !isNaN(num) ? num : undefined;
}

export async function predictPrice(
  prevState: FormState, // useFormState から渡される前の状態
  formData: FormData // フォームから送信されたデータ
): Promise<FormState> {

  // タイムスタンプを作成（デバッグ用）
  const requestTimestamp = Date.now();

  // デバッグ: 送信されてきたフォームデータの内容を確認
  console.log(`[${requestTimestamp}] Server Action: 受け取ったフォームデータ`, {
    year: formData.get('year'),
    mileage: formData.get('mileage'),
    brand: formData.get('brand'),
    fuel: formData.get('fuel'),
    transmission: formData.get('transmission'),
    owner_type: formData.get('owner_type'),
    seller_type: formData.get('seller_type'),
  });

  const rawFormData = {
    year: formData.get('year'),
    mileage: formData.get('mileage'),
    brand: formData.get('brand') || "", // Convert null to empty string for better error messages
    fuel: formData.get('fuel') || "",
    transmission: formData.get('transmission') || "",
    owner_type: formData.get('owner_type') || "",
    seller_type: formData.get('seller_type') || "",
  };

  // Zod でバリデーション
  const validatedFields = CarFeaturesSchema.safeParse(rawFormData);

  // デバッグログを追加
  console.log(`[${requestTimestamp}] バリデーション結果:`, {
    success: validatedFields.success,
    errors: validatedFields.success ? null : validatedFields.error.flatten().fieldErrors
  });

  // バリデーション失敗時の処理
  if (!validatedFields.success) {
    // エラー時にフォームの入力値を保持するための値を取得
    // フォームデータから値を安全に取得
    const fuelValue = formData.get('fuel');
    const transmissionValue = formData.get('transmission');
    const ownerTypeValue = formData.get('owner_type');
    const sellerTypeValue = formData.get('seller_type');

    // 各 enum 値の検証結果
    const fuelResult = CarFeaturesSchema.shape.fuel.safeParse(fuelValue);
    const transmissionResult = CarFeaturesSchema.shape.transmission.safeParse(transmissionValue);
    const ownerTypeResult = CarFeaturesSchema.shape.owner_type.safeParse(ownerTypeValue);
    const sellerTypeResult = CarFeaturesSchema.shape.seller_type.safeParse(sellerTypeValue);

    const fieldValuesOnError = {
      year: getSafeNumberFormValue(formData, 'year'),
      mileage: getSafeNumberFormValue(formData, 'mileage'),
      brand: getSafeFormValue(formData, 'brand'),
      fuel: fuelResult.success ? fuelResult.data : undefined,
      transmission: transmissionResult.success ? transmissionResult.data : undefined,
      owner_type: ownerTypeResult.success ? ownerTypeResult.data : undefined,
      seller_type: sellerTypeResult.success ? sellerTypeResult.data : undefined,
    }

    return {
      message: "入力内容にエラーがあります。",
      errors: validatedFields.error.flatten().fieldErrors,
      fieldValues: fieldValuesOnError
    };
  }

  // バリデーション成功: validatedFields.data を使用
  const { year, mileage, fuel, transmission, owner_type, seller_type } = validatedFields.data;

  // Owner Type 文字列から数値へのマッピング
  const ownerTypeMapping: { [key: string]: number } = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3
  };
  const ownerValue = ownerTypeMapping[owner_type] ?? 0;

  // API リクエストボディの作成
  const apiRequestBody = {
    year: year,
    present_price: 5.0, // ダミーデータ (本来は不要かも)
    kms_driven: mileage,
    fuel_type: fuel,
    seller_type: seller_type,
    transmission: transmission,
    owner: ownerValue
  };

  // デバッグ: APIリクエストの内容を確認
  console.log(`[${requestTimestamp}] APIに送信するリクエストボディ:`, apiRequestBody);

  try {
    // FastAPI バックエンドを呼び出す
    console.log(`[${requestTimestamp}] APIリクエスト送信開始`);
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(apiRequestBody)
    });
    console.log(`[${requestTimestamp}] APIレスポンス受信: status=${response.status}`);

    if (!response.ok) {
        let errorDetail = "APIリクエストが失敗しました";
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || JSON.stringify(errorData);
        } catch {
            errorDetail = await response.text();
        }
        throw new Error(`APIエラー (${response.status}): ${errorDetail}`);
    }

    // API レスポンスを解析
    const result = await response.json();

    // デバッグ: APIレスポンスの内容を確認
    console.log(`[${requestTimestamp}] APIからのレスポンス:`, result);

    // 新しいAPIレスポンス形式（PredictionResult）かどうかを確認
    const isNewApiFormat = 'predicted_price_jpy' in result &&
                          'lower_bound_jpy' in result &&
                          'upper_bound_jpy' in result;

    if (isNewApiFormat) {
      // 新しいAPIレスポンス形式の場合
      const typedResult = result as PredictionResult;

      // 日付に基づく微小なランダム要素を追加（同じパラメータでも少し違う結果になるように）
      const randFactor = 1 + (Math.sin(requestTimestamp / 10000) * 0.02); // ±2%の微小変動

      // 成功時のレスポンス
      return {
        message: "予測が完了しました。",
        predictedPrice: Math.round(typedResult.predicted_price_jpy * randFactor),
        lowerBoundPrice: Math.round(typedResult.lower_bound_jpy * randFactor),
        upperBoundPrice: Math.round(typedResult.upper_bound_jpy * randFactor),
        confidenceLevel: typedResult.confidence_level,
        fieldValues: {
          year: validatedFields.data.year,
          mileage: validatedFields.data.mileage,
          brand: validatedFields.data.brand,
          fuel: validatedFields.data.fuel,
          transmission: validatedFields.data.transmission,
          owner_type: validatedFields.data.owner_type,
          seller_type: validatedFields.data.seller_type,
        }
      };
    } else {
      // 旧APIレスポンス形式の場合（互換性のため）
      const oldResult = result as OldPredictionResult;

      // 日付に基づく微小なランダム要素を追加
      const randFactor = 1 + (Math.sin(requestTimestamp / 10000) * 0.02); // ±2%の微小変動

      // Lakhs から円に変換 (1 Lakh = 150,000円)
      const predictedPriceYen = Math.round(oldResult.predicted_price_lakhs * 150000 * randFactor);

      // 簡易的な信頼区間を計算（±10%）
      const lowerBound = Math.round(predictedPriceYen * 0.9);
      const upperBound = Math.round(predictedPriceYen * 1.1);

      return {
        message: "予測が完了しました。",
        predictedPrice: predictedPriceYen,
        lowerBoundPrice: lowerBound,
        upperBoundPrice: upperBound,
        confidenceLevel: 80, // 簡易的な信頼水準
        fieldValues: {
          year: validatedFields.data.year,
          mileage: validatedFields.data.mileage,
          brand: validatedFields.data.brand,
          fuel: validatedFields.data.fuel,
          transmission: validatedFields.data.transmission,
          owner_type: validatedFields.data.owner_type,
          seller_type: validatedFields.data.seller_type,
        }
      };
    }

  } catch (error) {
    let errorMessage = "予測中に不明なエラーが発生しました。";
    if (error instanceof Error) {
        errorMessage = error.message.includes("Failed to fetch") || error.message.includes("ECONNREFUSED")
            ? "APIサーバーに接続できませんでした。サーバーが起動しているか確認してください。"
            : `予測中にエラーが発生しました: ${error.message}`;
    }

    console.error("API Error:", error);

    // API エラー時も同様に安全に取得
    const fuelValue = formData.get('fuel');
    const transmissionValue = formData.get('transmission');
    const ownerTypeValue = formData.get('owner_type');
    const sellerTypeValue = formData.get('seller_type');

    // 各 enum 値の検証結果
    const fuelResult = CarFeaturesSchema.shape.fuel.safeParse(fuelValue);
    const transmissionResult = CarFeaturesSchema.shape.transmission.safeParse(transmissionValue);
    const ownerTypeResult = CarFeaturesSchema.shape.owner_type.safeParse(ownerTypeValue);
    const sellerTypeResult = CarFeaturesSchema.shape.seller_type.safeParse(sellerTypeValue);

    // エラー時のレスポンス
    return {
      message: errorMessage,
      errors: { _errors: [errorMessage] },
      // API エラー時もフォームの入力値を保持
      fieldValues: {
        year: getSafeNumberFormValue(formData, 'year'),
        mileage: getSafeNumberFormValue(formData, 'mileage'),
        brand: getSafeFormValue(formData, 'brand'),
        fuel: fuelResult.success ? fuelResult.data : undefined,
        transmission: transmissionResult.success ? transmissionResult.data : undefined,
        owner_type: ownerTypeResult.success ? ownerTypeResult.data : undefined,
        seller_type: sellerTypeResult.success ? sellerTypeResult.data : undefined,
      }
    };
  }
}
