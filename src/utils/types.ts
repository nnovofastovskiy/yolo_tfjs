import tf from '@tensorflow/tfjs';

export interface Detection {
    box: BoundingBox;
    score: number;
    class: number;
    maskCoeffs?: number[];
    mask?: number[][]; // ДОБАВЬТЕ ЭТО ПОЛЕ
}

export interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface PreprocessResult {
    tensor: tf.Tensor;
    scale: number;
    padL: number;
    padT: number;
}
