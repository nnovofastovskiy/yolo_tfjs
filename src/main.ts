import './style.css';
import * as tf from '@tensorflow/tfjs';
import {
  loadModel,
  preprocessImage,
  processSegmentation,
  drawDetections
} from './utils/segmentation';
import { LABELS, COLORS } from './utils/labels';

let model: tf.GraphModel | null = null;
let isProcessing = false;

const elements = {
  status: document.getElementById('status') as HTMLDivElement,
  imageUpload: document.getElementById('imageUpload') as HTMLInputElement,
  sourceImage: document.getElementById('sourceImage') as HTMLImageElement,
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  processing: document.getElementById('processing') as HTMLDivElement,
  inferenceTime: document.getElementById('inferenceTime') as HTMLDivElement // ДОБАВЬТЕ ЭТО
};

async function initModel(): Promise<void> {
  try {
    console.log('Загрузка модели YOLO11n-seg...');
    const t0 = performance.now();
    model = await loadModel('/model/model.json');
    const t1 = performance.now();

    elements.status.textContent = `✅ Модель готова к работе (загружена за ${(t1 - t0).toFixed(0)}мс)`;
    elements.status.classList.add('ready');
    elements.imageUpload.disabled = false;

    console.log('Модель загружена успешно');
  } catch (error) {
    console.error('Ошибка загрузки модели:', error);
    elements.status.textContent = '❌ Ошибка загрузки модели';
    elements.status.classList.add('error');
  }
}

async function handleImageUpload(event: Event): Promise<void> {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];

  if (!file || !model || isProcessing) return;

  const reader = new FileReader();
  reader.onload = (e: ProgressEvent<FileReader>) => {
    if (!e.target?.result) return;

    elements.sourceImage.src = e.target.result as string;
    elements.sourceImage.onload = () => detectAndSegment(elements.sourceImage);
  };
  reader.readAsDataURL(file);
}

async function detectAndSegment(img: HTMLImageElement): Promise<void> {
  if (!model || isProcessing) return;

  isProcessing = true;
  elements.processing.style.display = 'block';
  elements.inferenceTime.style.display = 'none'; // ДОБАВЬТЕ ЭТО
  elements.imageUpload.disabled = true;

  const ctx = elements.canvas.getContext('2d');
  if (!ctx) return;

  elements.canvas.width = img.width;
  elements.canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  try {
    const totalStart = performance.now();

    const preprocessStart = performance.now();
    const { tensor, scale, padL, padT } = preprocessImage(img);
    const preprocessEnd = performance.now();

    const inferenceStart = performance.now();
    const predictions = await model.executeAsync(tensor);
    const inferenceEnd = performance.now();

    const postprocessStart = performance.now();
    const results = await processSegmentation(
      predictions,
      img.width,
      img.height,
      scale,
      padL,
      padT
    );
    const postprocessEnd = performance.now();

    const drawStart = performance.now();
    drawDetections(ctx, results, LABELS, COLORS, img.width, img.height, scale, padL, padT);
    const drawEnd = performance.now();

    const totalEnd = performance.now();

    // ДОБАВЬТЕ ЭТО - отображение времени
    const preprocessTime = preprocessEnd - preprocessStart;
    const inferenceTime = inferenceEnd - inferenceStart;
    const postprocessTime = postprocessEnd - postprocessStart;
    const drawTime = drawEnd - drawStart;
    const totalTime = totalEnd - totalStart;

    elements.inferenceTime.innerHTML = `
      ⚡ <strong>Время обработки:</strong><br>
      • Предобработка: ${preprocessTime.toFixed(1)}мс<br>
      • Инференс: ${inferenceTime.toFixed(1)}мс<br>
      • Постобработка: ${postprocessTime.toFixed(1)}мс<br>
      • Отрисовка: ${drawTime.toFixed(1)}мс<br>
      • <strong>Всего: ${totalTime.toFixed(1)}мс</strong> | Найдено объектов: ${results.length}
    `;
    elements.inferenceTime.style.display = 'block';

    console.log(`⚡ Детальная статистика:
      - Предобработка: ${preprocessTime.toFixed(1)}мс
      - Инференс: ${inferenceTime.toFixed(1)}мс
      - Постобработка: ${postprocessTime.toFixed(1)}мс
      - Отрисовка: ${drawTime.toFixed(1)}мс
      - Всего: ${totalTime.toFixed(1)}мс`);
    console.log(`🎯 Найдено объектов: ${results.length}`);

    tf.dispose([tensor, predictions]);
  } catch (error) {
    console.error('Ошибка при сегментации:', error);
    alert('Произошла ошибка при обработке изображения');
  } finally {
    isProcessing = false;
    elements.processing.style.display = 'none';
    elements.imageUpload.disabled = false;
  }
}

// Инициализация
elements.imageUpload.addEventListener('change', handleImageUpload);
initModel();
