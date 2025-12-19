import * as tf from '@tensorflow/tfjs';
import type { Detection, PreprocessResult, BoundingBox } from './types';

export async function loadModel(modelPath: string): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel(modelPath);

    const dummy = tf.zeros([1, 640, 640, 3]);
    const warmup = await model.executeAsync(dummy);
    tf.dispose([dummy, warmup]);

    return model;
}

export function preprocessImage(
    img: HTMLImageElement,
    inputSize: number = 640
): PreprocessResult {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img);
        const [h, w] = tensor.shape.slice(0, 2);

        const scale = Math.min(inputSize / w, inputSize / h);
        const nh = Math.round(h * scale);
        const nw = Math.round(w * scale);

        tensor = tf.image.resizeBilinear(tensor, [nh, nw]);

        const padX = inputSize - nw;
        const padY = inputSize - nh;
        const padL = Math.floor(padX / 2);
        const padT = Math.floor(padY / 2);
        const padR = padX - padL;
        const padB = padY - padT;

        tensor = tf.pad(tensor, [[padT, padB], [padL, padR], [0, 0]], 114);
        tensor = tensor.div(255.0).expandDims(0);

        return { tensor, scale, padL, padT };
    });
}

export async function processSegmentation(
    output: tf.Tensor | tf.Tensor[],
    imgWidth: number,
    imgHeight: number,
    scale: number,
    padL: number,
    padT: number,
    threshold: number = 0.5,
    enableMasks: boolean = true // ДОБАВЛЕНО
): Promise<Detection[]> {
    const boxesOut = Array.isArray(output) ? output[0] : output;
    const maskProtos = Array.isArray(output) && output.length > 1 ? output[1] : null;

    console.log('Output shape:', boxesOut.shape);
    if (maskProtos) {
        console.log('Mask protos shape:', maskProtos.shape);
    }

    const transposed = boxesOut.transpose([0, 2, 1]);
    const data = await transposed.array() as number[][][];
    const detections = data[0];

    const results: Detection[] = [];
    const numClasses = 1;

    for (let i = 0; i < detections.length; i++) {
        const det = detections[i];

        const xCenter = det[0];
        const yCenter = det[1];
        const w = det[2];
        const h = det[3];

        const classScores = det.slice(4, 4 + numClasses);
        const maxScore = Math.max(...classScores);
        const classId = classScores.indexOf(maxScore);

        if (maxScore < threshold) continue;

        const x1 = xCenter - w / 2;
        const y1 = yCenter - h / 2;
        const x2 = xCenter + w / 2;
        const y2 = yCenter + h / 2;

        const box = {
            x: (x1 - padL) / scale,
            y: (y1 - padT) / scale,
            width: (x2 - x1) / scale,
            height: (y2 - y1) / scale
        };

        const maskCoeffs = det.slice(4 + numClasses);

        results.push({
            box,
            score: maxScore,
            class: classId,
            maskCoeffs: maskCoeffs.length > 0 ? maskCoeffs : undefined
        });
    }

    transposed.dispose();

    console.log(`Filtered detections before NMS: ${results.length}`);

    const nmsResults = applyNMS(results, 0.45);

    // ИЗМЕНЕНО: Декодируем маски только если включен режим сегментации
    if (enableMasks && maskProtos && nmsResults.length > 0) {
        await decodeMasks(nmsResults, maskProtos, scale, padL, padT);
    }

    return nmsResults;
}

export function drawDetections(
    ctx: CanvasRenderingContext2D,
    detections: Detection[],
    labels: string[],
    colors: string[],
    imgWidth: number,
    imgHeight: number,
    scale: number,
    padL: number,
    padT: number,
    drawMasks: boolean = true
): void {
    ctx.lineWidth = 3;
    ctx.font = 'bold 16px Arial';

    detections.forEach((det) => {
        const color = colors[det.class % colors.length];

        // Отрисовка маски
        if (drawMasks && det.mask) {
            drawSegmentationMask(ctx, det, color, imgWidth, imgHeight, scale, padL, padT);
        }

        // ИЗМЕНЕНО: Отрисовка овала вместо прямоугольника
        const centerX = det.box.x + det.box.width / 2;
        const centerY = det.box.y + det.box.height / 2;
        const radiusX = det.box.width / 2;
        const radiusY = det.box.height / 2;

        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
        ctx.stroke();

        // Отрисовка метки
        const label = `${labels[det.class]}: ${(det.score * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(label).width;

        // Метка над верхней точкой овала
        const labelX = centerX - textWidth / 2;
        const labelY = det.box.y - 5;

        ctx.fillStyle = color;
        ctx.fillRect(labelX - 5, labelY - 20, textWidth + 10, 25);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, labelX, labelY - 2);
    });
}

// Функция для вычисления IoU (Intersection over Union)
function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionWidth = Math.max(0, x2 - x1);
    const intersectionHeight = Math.max(0, y2 - y1);
    const intersectionArea = intersectionWidth * intersectionHeight;

    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
}

// Non-Maximum Suppression
function applyNMS(detections: Detection[], iouThreshold: number = 0.5): Detection[] {
    if (detections.length === 0) return [];

    // Сортируем по score (от большего к меньшему)
    const sorted = [...detections].sort((a, b) => b.score - a.score);
    const selected: Detection[] = [];
    const suppressed = new Set<number>();

    for (let i = 0; i < sorted.length; i++) {
        if (suppressed.has(i)) continue;

        selected.push(sorted[i]);

        // Подавляем все боксы с высоким IoU с текущим
        for (let j = i + 1; j < sorted.length; j++) {
            if (suppressed.has(j)) continue;

            const iou = calculateIoU(sorted[i].box, sorted[j].box);
            if (iou > iouThreshold) {
                suppressed.add(j);
            }
        }
    }

    console.log(`NMS: ${detections.length} -> ${selected.length} детекций`);
    return selected;
}


// Функция для декодирования масок сегментации
async function decodeMasks(
    detections: Detection[],
    maskProtos: tf.Tensor,
    scale: number,
    padL: number,
    padT: number
): Promise<void> {
    return tf.tidy(() => {
        console.log('Original mask protos shape:', maskProtos.shape);

        // maskProtos обычно [1, 32, 160, 160] или [1, 160, 160, 32]
        let protosData = maskProtos.squeeze([0]); // Убираем batch dimension
        console.log('Squeezed protos shape:', protosData.shape);

        // Проверяем формат и при необходимости транспонируем
        const shape = protosData.shape;
        if (shape.length === 3) {
            // Если [160, 160, 32] - транспонируем в [32, 160, 160]
            if (shape[2] === 32 || shape[2] < shape[0]) {
                protosData = protosData.transpose([2, 0, 1]);
                console.log('Transposed protos shape:', protosData.shape);
            }
            // Если [32, 160, 160] - уже правильный формат
        }

        for (let i = 0; i < detections.length; i++) {
            const det = detections[i];
            if (!det.maskCoeffs || det.maskCoeffs.length === 0) continue;

            try {
                // Создаем тензор из коэффициентов [32]
                const coeffs = tf.tensor1d(det.maskCoeffs.slice(0, 32)); // Берем только первые 32

                // Умножаем coefficients на protos: [32] x [32, 160, 160] -> [160, 160]
                const maskTensor = tf.einsum('c,chw->hw', coeffs, protosData);

                // Применяем sigmoid для получения вероятностей
                const mask = tf.sigmoid(maskTensor);

                // Сохраняем маску в detection
                det.mask = mask.arraySync() as number[][];

                if (i === 0) {
                    console.log('Mask shape:', mask.shape);
                }

                coeffs.dispose();
                mask.dispose();
                maskTensor.dispose();
            } catch (error) {
                console.error(`Error decoding mask for detection ${i}:`, error);
            }
        }
    });
}


function drawSegmentationMask(
    ctx: CanvasRenderingContext2D,
    detection: Detection,
    color: string,
    imgWidth: number,
    imgHeight: number,
    scale: number,
    padL: number,
    padT: number
): void {
    if (!detection.mask) return;

    const mask = detection.mask;
    const maskHeight = mask.length; // 160
    const maskWidth = mask[0].length; // 160

    const box = detection.box;

    // Маска 160x160 соответствует изображению 640x640 (после letterbox)
    const maskScale = 640 / maskWidth; // обычно 4

    const rgb = hexToRgb(color);

    // Границы bbox для ограничения области
    const boxLeft = Math.max(0, Math.floor(box.x));
    const boxTop = Math.max(0, Math.floor(box.y));
    const boxRight = Math.min(imgWidth, Math.ceil(box.x + box.width));
    const boxBottom = Math.min(imgHeight, Math.ceil(box.y + box.height));

    // Создаем ImageData только для области bbox
    const regionWidth = boxRight - boxLeft;
    const regionHeight = boxBottom - boxTop;

    if (regionWidth <= 0 || regionHeight <= 0) return;

    const imageData = ctx.createImageData(regionWidth, regionHeight);
    const data = imageData.data;

    // Проходим только по пикселям внутри bbox
    for (let y = 0; y < regionHeight; y++) {
        for (let x = 0; x < regionWidth; x++) {
            // Координаты в исходном изображении
            const imgX = boxLeft + x;
            const imgY = boxTop + y;

            // Преобразуем в координаты 640x640 (с учетом letterbox)
            const x640 = imgX * scale + padL;
            const y640 = imgY * scale + padT;

            // Преобразуем в координаты маски 160x160
            const maskX = Math.floor(x640 / maskScale);
            const maskY = Math.floor(y640 / maskScale);

            if (maskX >= 0 && maskX < maskWidth && maskY >= 0 && maskY < maskHeight) {
                const maskValue = mask[maskY][maskX];

                // Увеличенный порог для лучшей фильтрации
                if (maskValue > 0.6) {
                    const idx = (y * regionWidth + x) * 4;
                    data[idx] = rgb.r;
                    data[idx + 1] = rgb.g;
                    data[idx + 2] = rgb.b;
                    // Прозрачность зависит от уверенности маски
                    data[idx + 3] = Math.floor(maskValue * 180);
                }
            }
        }
    }

    // Рисуем только в области bbox
    ctx.putImageData(imageData, boxLeft, boxTop);
}


// Вспомогательная функция для конвертации hex в RGB
function hexToRgb(hex: string): { r: number; g: number; b: number } {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 0, g: 255, b: 0 };
}
