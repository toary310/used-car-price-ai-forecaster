// Remove unnecessary commented out imports
import PredictionForm from '@/app/components/PredictionForm';
import styles from './page.module.css';

export default function Home() {
  return (
    // main 要素に CSS Module のクラスを適用
    <main className={styles.main}>
      <PredictionForm />
    </main>
  );
}
