"use client";

import { type FormState, predictPrice } from '@/app/actions';
import { useActionState, useEffect, useRef, useState, useTransition } from 'react';
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

  // アニメーション用の状態
  const [showAnimation, setShowAnimation] = useState(false);

  // トランジション状態（React 18のトランジションAPI）
  const [isPending, startTransition] = useTransition();

  // アプリケーションの状態管理
  useEffect(() => {
    // 予測が完了した場合
    if (state?.message === "予測が完了しました。" && state?.predictedPrice !== undefined) {
      // 予測成功時にアニメーションをトリガー
      setShowAnimation(true);
    }
  }, [state]);

  // 信頼区間のプログレスバーコンポーネント
  function ConfidenceIntervalBar({ lower, upper, predicted }: { lower: number, upper: number, predicted: number }) {
    // 予測値が範囲内に収まるように保証
    const safePredict = Math.min(Math.max(predicted, lower), upper);
    // 予測値の位置（%）を計算
    const position = ((safePredict - lower) / (upper - lower)) * 100;

    return (
      <div className={styles.confidenceBarContainer}>
        {/* 予測価格の表示 */}
        <div className={styles.predictedPriceText}>
          予測価格: <strong>{predicted.toLocaleString()} 円</strong>
        </div>

        {/* 価格表示 */}
        <div className={styles.priceRangeContainer}>
          {/* 最低価格 (左) */}
          <div className={styles.priceRangeLeft}>
            <div className={styles.priceRangeLabel}>
              最低価格
            </div>
            <div className={styles.priceRangeValue}>
              {lower.toLocaleString()} 円
            </div>
          </div>

          {/* 信頼区間バー (中央) */}
          <div className={styles.priceRangeCenter}>
            <div className={styles.confidenceBar}>
              <div
                className={styles.confidenceBarMarker}
                style={{ left: `${position}%` }}
              />
            </div>
          </div>

          {/* 最高価格 (右) */}
          <div className={styles.priceRangeRight}>
            <div className={styles.priceRangeLabel}>
              最高価格
            </div>
            <div className={styles.priceRangeValue}>
              {upper.toLocaleString()} 円
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 価格ゲージコンポーネント - アニメーション付きの視覚的な価格表現
  function PriceGauge({ price, upperBound, color }: { price: number, upperBound: number, color: string }) {
    // 最大値を少し上回る範囲に設定
    const maxGaugeValue = upperBound * 1.1;
    // パーセンテージ計算（0-100%）
    const percentage = (price / maxGaugeValue) * 100;

    return (
      <div className={styles.priceGauge}>
        <svg width="200" height="200" viewBox="0 0 200 200">
          {/* 背景の円 */}
          <circle cx="100" cy="100" r="90" fill="none" stroke="#e5e7eb" strokeWidth="20" />

          {/* 進捗を示す円弧 */}
          <circle
            cx="100"
            cy="100"
            r="90"
            fill="none"
            stroke={color}
            strokeWidth="20"
            strokeDasharray={`${percentage * 5.65} 565`} // 565 = 2πr
            strokeLinecap="round"
            transform="rotate(-90 100 100)"
            className={showAnimation ? styles.priceIndicator : ''}
          />

          {/* 中央のテキスト */}
          <text x="100" y="100" textAnchor="middle" dominantBaseline="middle" fontSize="18" fontWeight="bold">
            {Math.round(percentage)}%
          </text>
          <text x="100" y="125" textAnchor="middle" dominantBaseline="middle" fontSize="12">
            予測価格帯
          </text>
        </svg>
      </div>
    );
  }

  // 予測結果の視覚化コンポーネント
  function PredictionVisualization({ price, lowerBound, upperBound }: { price: number, lowerBound: number, upperBound: number }) {
    // 価格帯に基づいて色を決定
    let gaugeColorDescription = '';
    let gaugeColor = '#4ade80'; // デフォルト緑

    if (price > upperBound) {
      gaugeColor = '#ef4444'; // 上限超え - 赤
      gaugeColorDescription = '相場より高い';
    } else if (price < lowerBound) {
      gaugeColor = '#facc15'; // 下限未満 - 黄
      gaugeColorDescription = '相場より安い';
    } else if (price > (lowerBound + upperBound) / 2) {
      gaugeColor = '#fb923c'; // 中間より上 - オレンジ
      gaugeColorDescription = '相場の上半分';
    } else {
      gaugeColorDescription = '相場の下半分';
    }

    return (
      <div className={`${styles.visualizationContainer} ${showAnimation ? styles.fadeInAnimation : ''}`}>
        <PriceGauge
          price={price}
          upperBound={upperBound}
          color={gaugeColor}
        />

        {/* 色の意味を説明する凡例 */}
        <div className={styles.colorLegend}>
          <div className={styles.colorIndicatorWrapper}>
            <span
              className={styles.colorIndicator}
              style={{ backgroundColor: gaugeColor }}
            ></span>
            <span className={styles.colorDescription}>{gaugeColorDescription}</span>
          </div>
          <div className={styles.colorLegendText}>
            グラフの色: 緑=相場下半分、オレンジ=相場上半分、黄=相場より安い、赤=相場より高い
          </div>
        </div>

        <div className={styles.predictionComparison}>
          <p className={styles.predictionComparisonText}>
            予測価格: <strong>{price.toLocaleString()} 円</strong>
          </p>
          <div className={styles.predictionComparisonRange}>
            <span>最低価格: {lowerBound.toLocaleString()} 円</span>
            <span>最高価格: {upperBound.toLocaleString()} 円</span>
          </div>
        </div>
      </div>
    );
  }

  // フォーム送信ハンドラー（preventDefaultを使用）
  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    // デフォルトの送信をキャンセル
    event.preventDefault();

    // アニメーションをリセット
    setShowAnimation(false);

    // フォームデータを取得
    if (formRef.current) {
      const formData = new FormData(formRef.current);

      // Server Actionを呼び出し
      void formAction(formData);
    }
  };

  return (
    <div className={styles.formContainer}>
      {/* 左側のカラム - フォーム */}
      <div className={styles.formColumn}>
        <h2>
          中古車価格予測AI
        </h2>

        <p className={styles.requiredNote}>
          <span className={styles.requiredMark}>*</span> 必須項目
        </p>

        <form
          ref={formRef}
          onSubmit={handleSubmit}
        >
          {/* 全体のエラーメッセージがある場合 */}
          {state?.errors?._errors && (
            <div className={styles.formError}>
              <p>{state.errors._errors[0]}</p>
            </div>
          )}

          {/* --- 年式 --- */}
          <div className={styles.fieldGroup}>
            <label htmlFor="year" className={styles.label}>年式 <span className={styles.requiredMark}>*</span></label>
            <input
              id="year"
              name="year"
              type="number"
              placeholder="例: 2018（必須）"
              className={`${styles.input} ${state?.errors?.year ? styles.inputError : ''}`}
              defaultValue={state?.fieldValues?.year?.toString() ?? ""}
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
            <label htmlFor="mileage" className={styles.label}>走行距離 (km) <span className={styles.requiredMark}>*</span></label>
            <input
              id="mileage"
              name="mileage"
              type="number"
              placeholder="例: 50000（必須）"
              className={`${styles.input} ${state?.errors?.mileage ? styles.inputError : ''}`}
              defaultValue={state?.fieldValues?.mileage?.toString() ?? ""}
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
            <label htmlFor="brand" className={styles.label}>メーカー <span className={styles.requiredMark}>*</span></label>
            <select
              id="brand"
              name="brand"
              className={`${styles.select} ${state?.errors?.brand ? styles.selectError : ''}`}
              defaultValue={state?.fieldValues?.brand ?? ""}
              aria-invalid={!!state?.errors?.brand}
              aria-describedby="brand-error"
              required
            >
              <option value="" disabled>メーカーを選択してください（必須）</option>
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
            <label htmlFor="fuel" className={styles.label}>燃料タイプ <span className={styles.requiredMark}>*</span></label>
            <select
              id="fuel"
              name="fuel"
              className={`${styles.select} ${state?.errors?.fuel ? styles.selectError : ''}`}
              defaultValue={state?.fieldValues?.fuel ?? ""}
              required
              aria-invalid={!!state?.errors?.fuel}
              aria-describedby="fuel-error"
            >
              <option value="" disabled>燃料タイプを選択してください（必須）</option>
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
            <label htmlFor="transmission" className={styles.label}>トランスミッション <span className={styles.requiredMark}>*</span></label>
            <select
              id="transmission"
              name="transmission"
              className={`${styles.select} ${state?.errors?.transmission ? styles.selectError : ''}`}
              defaultValue={state?.fieldValues?.transmission ?? ""}
              required
              aria-invalid={!!state?.errors?.transmission}
              aria-describedby="transmission-error"
            >
              <option value="" disabled>トランスミッションを選択してください（必須）</option>
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
            <label htmlFor="owner_type" className={styles.label}>所有者履歴 <span className={styles.requiredMark}>*</span></label>
            <select
              id="owner_type"
              name="owner_type"
              className={`${styles.select} ${state?.errors?.owner_type ? styles.selectError : ''}`}
              defaultValue={state?.fieldValues?.owner_type ?? ""}
              required
              aria-invalid={!!state?.errors?.owner_type}
              aria-describedby="owner_type-error"
            >
              <option value="" disabled>所有者履歴を選択してください（必須）</option>
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
            <label htmlFor="seller_type" className={styles.label}>販売者タイプ <span className={styles.requiredMark}>*</span></label>
            <select
              id="seller_type"
              name="seller_type"
              className={`${styles.select} ${state?.errors?.seller_type ? styles.selectError : ''}`}
              defaultValue={state?.fieldValues?.seller_type ?? ""}
              required
              aria-invalid={!!state?.errors?.seller_type}
              aria-describedby="seller_type-error"
            >
              <option value="" disabled>販売者タイプを選択してください（必須）</option>
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
            <button
              type="button"
              className={`${styles.submitButton} ${isPending ? styles.submitButtonPending : ''}`}
              disabled={isPending}
              onClick={() => {
                // アニメーションをリセット
                setShowAnimation(false);

                // フォームデータを直接取得して送信
                if (formRef.current) {
                  const formData = new FormData(formRef.current);

                  // startTransition内でServer Actionを呼び出し
                  startTransition(() => {
                    formAction(formData);
                  });
                }
              }}
            >
              {isPending ? '予測中...' : '価格を予測する'}
            </button>
          </div>

        </form>
      </div>

      {/* 右側のカラム - 結果とビジュアライゼーション */}
      <div className={styles.resultsColumn}>
        <h2>予測結果</h2>

        {/* 予測結果がない場合 */}
        {!state?.predictedPrice && !state?.errors?._errors && (
          <div className={styles.emptyStateMessage}>
            <p>左のフォームに車両情報を入力して、</p>
            <p>「価格を予測する」ボタンを押してください。</p>
          </div>
        )}

        {/* 予測成功時 */}
        {state?.message === "予測が完了しました。" && state?.predictedPrice !== undefined && (
          <div className={styles.predictionResult}>
            <h3>予測価格</h3>
            <p className={styles.predictedPrice}>{state.predictedPrice.toLocaleString()} 円</p>

            {/* ビジュアライゼーション */}
            {state.lowerBoundPrice !== undefined && state.upperBoundPrice !== undefined && (
              <>
                <PredictionVisualization
                  price={state.predictedPrice}
                  lowerBound={state.lowerBoundPrice}
                  upperBound={state.upperBoundPrice}
                />

                {/* 信頼区間の表示 */}
                <div className={styles.confidenceInterval}>
                  <h4 className={styles.confidenceIntervalTitle}>
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

                  <div className={styles.confidenceNote}>
                    <p className={styles.confidenceNoteText}>※ この予測は過去のデータに基づく参考値です。</p>
                    <p>実際の市場価格は様々な要因により変動します。</p>
                  </div>
                </div>
              </>
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
