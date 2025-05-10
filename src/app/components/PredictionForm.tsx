"use client";

import { type FormState, predictPrice } from '@/app/actions';
import { useActionState, useEffect, useRef } from 'react';
import { useFormStatus } from 'react-dom';
import styles from './PredictionForm.module.css';

// 各Select要素の選択肢
const carBrands = ["Toyota", "Honda", "Nissan", "Suzuki", "Daihatsu", "Mazda", "Subaru", "Mitsubishi", "Lexus", "Other"];
const fuelTypes = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'];
const transmissionTypes = ['Manual', 'Automatic'];
const ownerTypes = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'];
const sellerTypes = ['Dealer', 'Individual', 'Trustmark Dealer'];


export default function PredictionForm() {
  const initialState: FormState = null;
  const [state, formAction] = useActionState(predictPrice, initialState);
  const formRef = useRef<HTMLFormElement>(null);

  // フォームのリセット
  useEffect(() => {
    if (state?.message === "予測が完了しました。" && state?.predictedPrice !== undefined) {
      formRef.current?.reset();
    }
  }, [state]);

  const fieldValues = state?.fieldValues; // エラー時に値を保持するため

  // 送信ボタン (useFormStatus を使う)
  function SubmitButton() {
    const { pending } = useFormStatus();
    return (
      <button
        type="submit"
        className={`${styles.submitButton} ${pending ? styles.submitButtonPending : ''}`}
        disabled={pending}
        aria-disabled={pending}
      >
        {pending ? '予測中...' : '価格を予測する'}
      </button>
    );
  }

  // 信頼区間のプログレスバーコンポーネント
  function ConfidenceIntervalBar({ lower, upper, predicted }: { lower: number, upper: number, predicted: number }) {
    // 予測値が範囲内に収まるように保証
    const safePredict = Math.min(Math.max(predicted, lower), upper);
    // 予測値の位置（%）を計算
    const position = ((safePredict - lower) / (upper - lower)) * 100;

    return (
      <div className={styles.confidenceBarContainer}>
        <div className={styles.confidenceBar}>
          <div
            className={styles.confidenceBarMarker}
            style={{ left: `${position}%` }}
          />
        </div>
        <div className={styles.confidenceBarLabels}>
          <span>{lower.toLocaleString()} 円</span>
          <span>{upper.toLocaleString()} 円</span>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.formContainer}>
      <h2>
        中古車価格予測AI
      </h2>

      <form ref={formRef} action={formAction} >

        {/* --- 年式 --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="year" className={styles.label}>年式</label>
          <input
            id="year"
            name="year"
            type="number"
            placeholder="例: 2018"
            className={`${styles.input} ${state?.errors?.year ? styles.inputError : ''}`}
            defaultValue={fieldValues?.year?.toString()} // defaultValue は string
            required
            aria-invalid={!!state?.errors?.year}
            aria-describedby="year-error"
          />
          {state?.errors?.year && (
            <p id="year-error" className={styles.errorMessage}>{state.errors.year[0]}</p>
          )}
        </div>

        {/* --- 走行距離 --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="mileage" className={styles.label}>走行距離 (km)</label>
          <input
            id="mileage"
            name="mileage"
            type="number"
            placeholder="例: 50000"
            className={`${styles.input} ${state?.errors?.mileage ? styles.inputError : ''}`}
            defaultValue={fieldValues?.mileage?.toString()} // defaultValue は string
            required
            aria-invalid={!!state?.errors?.mileage}
            aria-describedby="mileage-error"
          />
          {state?.errors?.mileage && (
            <p id="mileage-error" className={styles.errorMessage}>{state.errors.mileage[0]}</p>
          )}
        </div>

        {/* --- メーカー --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="brand" className={styles.label}>メーカー</label>
          <select
            id="brand"
            name="brand"
            className={`${styles.select} ${state?.errors?.brand ? styles.selectError : ''}`}
            defaultValue={fieldValues?.brand ?? ""}
            required
            aria-invalid={!!state?.errors?.brand}
            aria-describedby="brand-error"
          >
            <option value="" disabled>メーカーを選択...</option>
            {carBrands.map(brand => (
              <option key={brand} value={brand}>{brand}</option>
            ))}
          </select>
          {state?.errors?.brand && (
            <p id="brand-error" className={styles.errorMessage}>{state.errors.brand[0]}</p>
          )}
        </div>

        {/* --- 燃料タイプ --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="fuel" className={styles.label}>燃料タイプ</label>
          <select
            id="fuel"
            name="fuel"
            className={`${styles.select} ${state?.errors?.fuel ? styles.selectError : ''}`}
            defaultValue={fieldValues?.fuel ?? ""}
            required
            aria-invalid={!!state?.errors?.fuel}
            aria-describedby="fuel-error"
          >
            <option value="" disabled>燃料タイプを選択...</option>
            {fuelTypes.map(fuel => (
              <option key={fuel} value={fuel}>{fuel}</option>
            ))}
          </select>
          {state?.errors?.fuel && (
            <p id="fuel-error" className={styles.errorMessage}>{state.errors.fuel[0]}</p>
          )}
        </div>

        {/* --- トランスミッション --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="transmission" className={styles.label}>トランスミッション</label>
          <select
            id="transmission"
            name="transmission"
            className={`${styles.select} ${state?.errors?.transmission ? styles.selectError : ''}`}
            defaultValue={fieldValues?.transmission ?? ""}
            required
            aria-invalid={!!state?.errors?.transmission}
            aria-describedby="transmission-error"
          >
            <option value="" disabled>トランスミッションを選択...</option>
            {transmissionTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          {state?.errors?.transmission && (
            <p id="transmission-error" className={styles.errorMessage}>{state.errors.transmission[0]}</p>
          )}
        </div>

        {/* --- 所有者履歴 --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="owner_type" className={styles.label}>所有者履歴</label>
          <select
            id="owner_type"
            name="owner_type"
            className={`${styles.select} ${state?.errors?.owner_type ? styles.selectError : ''}`}
            defaultValue={fieldValues?.owner_type ?? ""}
            required
            aria-invalid={!!state?.errors?.owner_type}
            aria-describedby="owner_type-error"
          >
            <option value="" disabled>所有者履歴を選択...</option>
            {ownerTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          {state?.errors?.owner_type && (
            <p id="owner_type-error" className={styles.errorMessage}>{state.errors.owner_type[0]}</p>
          )}
        </div>

        {/* --- 販売者タイプ --- */}
        <div className={styles.fieldGroup}>
          <label htmlFor="seller_type" className={styles.label}>販売者タイプ</label>
          <select
            id="seller_type"
            name="seller_type"
            className={`${styles.select} ${state?.errors?.seller_type ? styles.selectError : ''}`}
            defaultValue={fieldValues?.seller_type ?? ""}
            required
            aria-invalid={!!state?.errors?.seller_type}
            aria-describedby="seller_type-error"
          >
            <option value="" disabled>販売者タイプを選択...</option>
            {sellerTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          {state?.errors?.seller_type && (
            <p id="seller_type-error" className={styles.errorMessage}>{state.errors.seller_type[0]}</p>
          )}
        </div>

        {/* --- 送信ボタン --- */}
        <div>
          <SubmitButton />
        </div>

      </form>

      {/* --- 結果表示エリア --- */}
      <div>
        {/* 予測成功時 */}
        {state?.message === "予測が完了しました。" && state?.predictedPrice !== undefined && (
          <div className={styles.predictionResult}>
            <h3>予測価格</h3>
            <p className={styles.predictedPrice}>{state.predictedPrice.toLocaleString()} 円</p>

            {/* 信頼区間の表示（新しい機能） */}
            {state.lowerBoundPrice !== undefined && state.upperBoundPrice !== undefined && (
              <div className={styles.confidenceInterval}>
                <h4>
                  予測範囲
                  {state.confidenceLevel && (
                    <span className={styles.confidenceLevel}>
                      （{state.confidenceLevel}% 信頼区間）
                    </span>
                  )}
                </h4>

                <ConfidenceIntervalBar
                  lower={state.lowerBoundPrice}
                  upper={state.upperBoundPrice}
                  predicted={state.predictedPrice}
                />

                <p className={styles.confidenceNote}>
                  ※ この予測は過去のデータに基づく参考値です。<br />
                  実際の市場価格は様々な要因により変動します。
                </p>
              </div>
            )}
          </div>
        )}

        {/* エラーメッセージ (APIエラーなど) */}
        {state?.message && state.errors?._errors && (
          <div className={styles.predictionResultError}>
            <p>{state.errors._errors[0] || state.message}</p>
          </div>
        )}
      </div>

    </div>
  );
}
