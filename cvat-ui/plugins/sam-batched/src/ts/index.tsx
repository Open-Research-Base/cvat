// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { Tensor } from 'onnxruntime-web';
import { LRUCache } from 'lru-cache';
import { CVATCore, MLModel, Job } from 'cvat-core-wrapper';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import { InitBody, DecodeBody, WorkerAction } from './inference.worker';

interface SAMPlugin {
    name: string;
    description: string;
    cvat: {
        lambda: {
            call: {
                enter: (
                    plugin: SAMPlugin,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<null | APIWrapperEnterOptions>;
                leave: (
                    plugin: SAMPlugin,
                    result: object,
                    taskID: number,
                    model: MLModel,
                    args: any,
                ) => Promise<any>;
            };
        };
        jobs: {
            get: {
                leave: (
                    plugin: SAMPlugin,
                    results: any[],
                    query: { jobID?: number }
                ) => Promise<any>;
            };
        };
    };
    data: {
        initialized: boolean;
        worker: Worker;
        core: CVATCore | null;
        jobs: Record<number, Job>;
        modelID: string;
        modelURL: string;
        embeddings: LRUCache<string, Tensor>;
        lowResMasks: LRUCache<string, Tensor>;
        lastClicks: ClickType[];
    };
    callbacks: {
        onStatusChange: ((status: string) => void) | null;
    };
}

interface ClickType {
    clickType: 0 | 1 | 2 | 3;
    x: number;
    y: number;
}

function getModelScale(w: number, h: number): number {
    // Input images to SAM must be resized so the longest side is 1024
    const LONG_SIDE_LENGTH = 1024;
    const scale = LONG_SIDE_LENGTH / Math.max(h, w);
    return scale;
}

function modelData(
    {
        clicks, tensor, modelScale, maskInput,
    }: {
        clicks: ClickType[];
        tensor: Tensor;
        modelScale: { height: number; width: number; scale: number };
        maskInput: Tensor | null;
    },
): DecodeBody {
    const imageEmbedding = tensor;

    const n = clicks.length;
    const pointCoords = new Float32Array(2 * n);
    const pointLabels = new Float32Array(n);

    // Scale and add clicks
    for (let i = 0; i < n; i++) {
        pointCoords[2 * i] = clicks[i].x * modelScale.scale;
        pointCoords[2 * i + 1] = clicks[i].y * modelScale.scale;
        pointLabels[i] = clicks[i].clickType;
    }

    // Create the tensor
    const pointCoordsTensor = new Tensor('float32', pointCoords, [1, n, 2]);
    const pointLabelsTensor = new Tensor('float32', pointLabels, [1, n]);
    const imageSizeTensor = new Tensor('float32', [modelScale.height, modelScale.width]);

    const prevMask = maskInput ||
        new Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMaskInput = new Tensor('float32', [maskInput ? 1 : 0]);

    return {
        image_embeddings: imageEmbedding,
        point_coords: pointCoordsTensor,
        point_labels: pointLabelsTensor,
        orig_im_size: imageSizeTensor,
        mask_input: prevMask,
        has_mask_input: hasMaskInput,
    };
}

const samPlugin: SAMPlugin = {
    name: 'Segment Anything Batch Mode',
    description: 'Handles batched SAM serverless function output',
    cvat: {
        jobs: {
            get: {
                async leave(
                    plugin: SAMPlugin,
                    results: any[],
                    query: { jobID?: number },
                ): Promise<any> {
                    if (typeof query.jobID === 'number') {
                        [plugin.data.jobs[query.jobID]] = results;
                    }
                    return results;
                },
            },
        },
        lambda: {
            call: {
                async enter(
                    plugin: SAMPlugin,
                    taskID: number,
                    model: MLModel, { frame }: { frame: number },
                ): Promise<null | APIWrapperEnterOptions> {
                    return new Promise((resolve, reject) => {
                        function resolvePromise(): void {
                            const key = `${taskID}_${frame}`;
                            if (plugin.data.embeddings.has(key)) {
                                resolve({ preventMethodCall: true });
                            } else {
                                resolve(null);
                            }
                        }

                        if (model.id === plugin.data.modelID) {
                            if (!plugin.data.initialized) {
                                samPlugin.data.worker.postMessage({
                                    action: WorkerAction.INIT,
                                    payload: {
                                        decoderURL: samPlugin.data.modelURL,
                                    } as InitBody,
                                });

                                samPlugin.data.worker.onmessage = (e: MessageEvent) => {
                                    if (e.data.action !== WorkerAction.INIT) {
                                        reject(new Error(
                                            `Caught unexpected action response from worker: ${e.data.action}`,
                                        ));
                                    }

                                    if (!e.data.error) {
                                        samPlugin.data.initialized = true;
                                        resolvePromise();
                                    } else {
                                        reject(new Error(`SAM worker was not initialized. ${e.data.error}`));
                                    }
                                };
                            } else {
                                resolvePromise();
                            }
                        } else {
                            resolve(null);
                        }
                    });
                },

                async leave(
                    plugin: SAMPlugin,
                    result: any,
                    taskID: number,
                    model: MLModel,
                    {
                        frame, pos_points, neg_points, obj_bbox,
                    }: {
                        frame: number, pos_points: number[][], neg_points: number[][], obj_bbox: number[][],
                    },
                ): Promise<
                    {
                        mask: number[][];
                        bounds: [number, number, number, number];
                    }> {
                    return new Promise((resolve, reject) => {
                        if (model.id !== plugin.data.modelID) {
                            resolve(result);
                            return;
                        }

                        const job = Object.values(plugin.data.jobs).find((_job) => (
                            _job.taskId === taskID && frame >= _job.startFrame && frame <= _job.stopFrame
                        )) as Job;

                        if (!job) {
                            throw new Error('Could not find a job corresponding to the request');
                        }

                        plugin.data.jobs = {
                            // we do not need to store old job instances
                            [job.id]: job,
                        };

                        job.frames.get(frame)
                            .then(({ height: imHeight, width: imWidth }: { height: number; width: number }) => {
                                const key = `${taskID}_${frame}`;

                                if (result) {
                                    // console.log('[1] Decoding base64 blob for batched embeddings');
                                    const bin = window.atob(result.blob);
                                    const uint8Array = new Uint8Array(bin.length);
                                    for (let i = 0; i < bin.length; i++) {
                                        uint8Array[i] = bin.charCodeAt(i);
                                    }
                                    const float32Arr = new Float32Array(uint8Array.buffer);
                                    // console.log('[2] Batch embeddings shape: (9, 256, 64, 64), total elements:', float32Arr.length);
                                    plugin.data.embeddings.set(key, new Tensor('float32', float32Arr, [9, 256, 64, 64]));
                                }

                                const modelScale = {
                                    width: imWidth,
                                    height: imHeight,
                                    scale: getModelScale(imWidth, imHeight),
                                };

                                const clicks: ClickType[] = [];
                                if (obj_bbox.length) {
                                    clicks.push({ clickType: 2, x: obj_bbox[0][0], y: obj_bbox[0][1] });
                                    clicks.push({ clickType: 3, x: obj_bbox[1][0], y: obj_bbox[1][1] });
                                }

                                pos_points.forEach((point) => {
                                    clicks.push({ clickType: 1, x: point[0], y: point[1] });
                                });

                                neg_points.forEach((point) => {
                                    clicks.push({ clickType: 0, x: point[0], y: point[1] });
                                });

                                const isLowResMaskSuitable = JSON
                                    .stringify(clicks.slice(0, -1)) === JSON.stringify(plugin.data.lastClicks);

                                // Select patch based on click position
                                const patchSize = 1024;
                                const bboxActive = obj_bbox.length > 0;

                                let clickX: number;
                                let clickY: number;

                                if (bboxActive) {
                                    // Use bbox center
                                    clickX = (obj_bbox[0][0] + obj_bbox[1][0]) / 2;
                                    clickY = (obj_bbox[0][1] + obj_bbox[1][1]) / 2;
                                } else {
                                    // Use mean position of all clicks
                                    clickX = clicks.reduce((sum, c) => sum + c.x, 0) / clicks.length;
                                    clickY = clicks.reduce((sum, c) => sum + c.y, 0) / clicks.length;
                                }

                                // Calculate patch positions (top-left corners)
                                const topY = 0;
                                const centerY = (imHeight - patchSize) / 2;
                                const bottomY = imHeight - patchSize;
                                const leftX = 0;
                                const centerX = (imWidth - patchSize) / 2;
                                const rightX = imWidth - patchSize;

                                // Define 9 patches with their top-left positions
                                const patches = [
                                    { index: 0, x: leftX, y: topY },     // top_left
                                    { index: 1, x: centerX, y: topY },   // top_center
                                    { index: 2, x: rightX, y: topY },    // top_right
                                    { index: 3, x: leftX, y: centerY },  // left_center
                                    { index: 4, x: centerX, y: centerY },// center
                                    { index: 5, x: rightX, y: centerY }, // right_center
                                    { index: 6, x: leftX, y: bottomY },  // bottom_left
                                    { index: 7, x: centerX, y: bottomY },// bottom_center
                                    { index: 8, x: rightX, y: bottomY }, // bottom_right
                                ];

                                let patchIndex = 4; // default center
                                let patchOffsetX = centerX;
                                let patchOffsetY = centerY;

                                if (bboxActive) {
                                    // Use IoU to select patch
                                    const bboxXtl = obj_bbox[0][0];
                                    const bboxYtl = obj_bbox[0][1];
                                    const bboxXbr = obj_bbox[1][0];
                                    const bboxYbr = obj_bbox[1][1];

                                    // console.log(`[3] BBox: [(${bboxXtl.toFixed(1)}, ${bboxYtl.toFixed(1)}), (${bboxXbr.toFixed(1)}, ${bboxYbr.toFixed(1)})], image (${imWidth}, ${imHeight})`);

                                    let maxIoU = 0;

                                    patches.forEach(patch => {
                                        const patchXtl = patch.x;
                                        const patchYtl = patch.y;
                                        const patchXbr = patch.x + patchSize;
                                        const patchYbr = patch.y + patchSize;

                                        // Calculate intersection
                                        const intersectXtl = Math.max(bboxXtl, patchXtl);
                                        const intersectYtl = Math.max(bboxYtl, patchYtl);
                                        const intersectXbr = Math.min(bboxXbr, patchXbr);
                                        const intersectYbr = Math.min(bboxYbr, patchYbr);

                                        const intersectWidth = Math.max(0, intersectXbr - intersectXtl);
                                        const intersectHeight = Math.max(0, intersectYbr - intersectYtl);
                                        const intersectArea = intersectWidth * intersectHeight;

                                        // Calculate union
                                        const bboxArea = (bboxXbr - bboxXtl) * (bboxYbr - bboxYtl);
                                        const patchArea = patchSize * patchSize;
                                        const unionArea = bboxArea + patchArea - intersectArea;

                                        const iou = unionArea > 0 ? intersectArea / unionArea : 0;

                                        if (iou > maxIoU) {
                                            maxIoU = iou;
                                            patchIndex = patch.index;
                                            patchOffsetX = patch.x;
                                            patchOffsetY = patch.y;
                                        }
                                    });

                                    // console.log(`[4] Selected patch ${patchIndex} (IoU: ${maxIoU.toFixed(4)}), offset: (${patchOffsetX}, ${patchOffsetY})`);
                                } else {
                                    // Use distance to select patch
                                    // console.log(`[3] Click at (${clickX.toFixed(1)}, ${clickY.toFixed(1)}), image (${imWidth}, ${imHeight})`);

                                    let minDistance = Infinity;

                                    patches.forEach(patch => {
                                        const patchCenterX = patch.x + patchSize / 2;
                                        const patchCenterY = patch.y + patchSize / 2;
                                        const distance = Math.sqrt(
                                            Math.pow(clickX - patchCenterX, 2) +
                                            Math.pow(clickY - patchCenterY, 2)
                                        );

                                        if (distance < minDistance) {
                                            minDistance = distance;
                                            patchIndex = patch.index;
                                            patchOffsetX = patch.x;
                                            patchOffsetY = patch.y;
                                        }
                                    });

                                    // console.log(`[4] Selected patch ${patchIndex} (distance: ${minDistance.toFixed(2)}px), offset: (${patchOffsetX}, ${patchOffsetY})`);
                                }

                                // Extract single patch embedding
                                const batchTensor = plugin.data.embeddings.get(key) as Tensor;
                                const embeddingSize = 256 * 64 * 64;
                                const patchData = (batchTensor.data as Float32Array).slice(
                                    patchIndex * embeddingSize,
                                    (patchIndex + 1) * embeddingSize
                                );
                                const patchTensor = new Tensor('float32', patchData, [1, 256, 64, 64]);
                                // console.log('[5] Extracted patch tensor shape: (1, 256, 64, 64)');

                                // Adjust clicks to patch coordinate space and clamp to patch bounds
                                const patchClicks = clicks.map(c => ({
                                    ...c,
                                    x: Math.max(0, Math.min(patchSize - 1, c.x - patchOffsetX)),
                                    y: Math.max(0, Math.min(patchSize - 1, c.y - patchOffsetY))
                                }));
                                // console.log('[6] Adjusted clicks to patch space:', patchClicks);

                                const feeds = modelData({
                                    clicks: patchClicks,
                                    tensor: patchTensor,
                                    modelScale: {
                                        width: patchSize,
                                        height: patchSize,
                                        scale: getModelScale(patchSize, patchSize),
                                    },
                                    maskInput: null, // Don't use low res masks for now
                                });

                                function toMatImage(input: number[], width: number, height: number): number[][] {
                                    const image = Array(height).fill(0);
                                    for (let i = 0; i < image.length; i++) {
                                        image[i] = Array(width).fill(0);
                                    }

                                    for (let i = 0; i < input.length; i++) {
                                        const row = Math.floor(i / width);
                                        const col = i % width;
                                        image[row][col] = input[i] > 0 ? 255 : 0;
                                    }

                                    return image;
                                }

                                function onnxToImage(input: any, width: number, height: number): number[][] {
                                    return toMatImage(input, width, height);
                                }

                                plugin.data.worker.postMessage({
                                    action: WorkerAction.DECODE,
                                    payload: feeds,
                                });

                                plugin.data.worker.onmessage = ((e) => {
                                    if (e.data.action !== WorkerAction.DECODE) {
                                        const error = 'Caught unexpected action response from worker: ' +
                                                `${e.data.action}, while "${WorkerAction.DECODE}" was expected`;
                                        reject(new Error(error));
                                    }

                                    if (!e.data.error) {
                                        const {
                                            masks, lowResMasks, xtl, ytl, xbr, ybr,
                                        } = e.data.payload;
                                        const imageData = onnxToImage(masks.data, masks.dims[3], masks.dims[2]);
                                        plugin.data.lowResMasks.set(key, lowResMasks);
                                        plugin.data.lastClicks = clicks;

                                        // console.log('[7] Mask bounds in patch space:', [xtl, ytl, xbr, ybr]);

                                        // Translate mask bounds from patch space to image space
                                        const translatedXtl = xtl + patchOffsetX;
                                        const translatedYtl = ytl + patchOffsetY;
                                        const translatedXbr = xbr + patchOffsetX;
                                        const translatedYbr = ybr + patchOffsetY;

                                        // console.log('[8] Mask bounds in image space:', [translatedXtl, translatedYtl, translatedXbr, translatedYbr]);

                                        resolve({
                                            mask: imageData,
                                            bounds: [translatedXtl, translatedYtl, translatedXbr, translatedYbr],
                                        });
                                    } else {
                                        reject(new Error(`Decoder error. ${e.data.error}`));
                                    }
                                });

                                plugin.data.worker.onerror = ((error) => {
                                    reject(error);
                                });
                            });
                    });
                },
            },
        },
    },
    data: {
        initialized: false,
        core: null,
        worker: new Worker(new URL('./inference.worker', import.meta.url)),
        jobs: {},
        modelID: 'pth-facebookresearch-sam-batched-vit-h',
        modelURL: '/assets/decoder.onnx',
        embeddings: new LRUCache({
            // float32 tensor [256, 64, 64] is 4 MB, max 128 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lowResMasks: new LRUCache({
            // float32 tensor [1, 256, 256] is 0.25 MB, max 8 MB
            max: 32,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lastClicks: [],
    },
    callbacks: {
        onStatusChange: null,
    },
};

const builder: ComponentBuilder = ({ core }) => {
    samPlugin.data.core = core;
    core.plugins.register(samPlugin);

    return {
        name: samPlugin.name,
        destructor: () => {},
    };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } })
            .cvatUI.registerComponent(builder);
    }
}

window.addEventListener('plugins.ready', register, { once: true });
