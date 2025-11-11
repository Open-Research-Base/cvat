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
    name: 'Segment Anything',
    description: 'Handles non-default SAM serverless function output',
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
                                    console.log('[1] Decoding base64 blob for batched embeddings');
                                    const bin = window.atob(result.blob);
                                    const uint8Array = new Uint8Array(bin.length);
                                    for (let i = 0; i < bin.length; i++) {
                                        uint8Array[i] = bin.charCodeAt(i);
                                    }
                                    const float32Arr = new Float32Array(uint8Array.buffer);
                                    console.log('[2] Batch embeddings shape: (9, 256, 64, 64), total elements:', float32Arr.length);
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
                                const firstClick = clicks.find(c => c.clickType === 1 || c.clickType === 2) || clicks[0];

                                // If bbox is active, use its center to select the patch
                                const bboxActive = obj_bbox.length > 0;
                                let clickX = firstClick.x;
                                let clickY = firstClick.y;

                                if(bboxActive) {
                                    clickX = (obj_bbox[0][0] + obj_bbox[1][0]) / 2;
                                    clickY = (obj_bbox[0][1] + obj_bbox[1][1]) / 2;
                                }

                                const centerX = (imWidth - patchSize) / 2;
                                const centerY = (imHeight - patchSize) / 2;
                                const rightX = imWidth - patchSize;
                                const bottomY = imHeight - patchSize;

                                let patchIndex = 4; // default center
                                let patchOffsetX = centerX;
                                let patchOffsetY = centerY;

                                if (clickY < centerY) { // Top row
                                    patchOffsetY = 0;
                                    if (clickX < centerX) { patchIndex = 0; patchOffsetX = 0; }
                                    else if (clickX < rightX) { patchIndex = 1; patchOffsetX = centerX; }
                                    else { patchIndex = 2; patchOffsetX = rightX; }
                                } else if (clickY < bottomY) { // Middle row
                                    patchOffsetY = centerY;
                                    if (clickX < centerX) { patchIndex = 3; patchOffsetX = 0; }
                                    else if (clickX < rightX) { patchIndex = 4; patchOffsetX = centerX; }
                                    else { patchIndex = 5; patchOffsetX = rightX; }
                                } else { // Bottom row
                                    patchOffsetY = bottomY;
                                    if (clickX < centerX) { patchIndex = 6; patchOffsetX = 0; }
                                    else if (clickX < rightX) { patchIndex = 7; patchOffsetX = centerX; }
                                    else { patchIndex = 8; patchOffsetX = rightX; }
                                }

                                console.log(`[3] Click at (${clickX}, ${clickY}), image (${imWidth}, ${imHeight})`);
                                console.log(`[4] Selected patch ${patchIndex}, offset (${patchOffsetX}, ${patchOffsetY})`);

                                // Extract single patch embedding
                                const batchTensor = plugin.data.embeddings.get(key) as Tensor;
                                const embeddingSize = 256 * 64 * 64;
                                const patchData = (batchTensor.data as Float32Array).slice(
                                    patchIndex * embeddingSize,
                                    (patchIndex + 1) * embeddingSize
                                );
                                const patchTensor = new Tensor('float32', patchData, [1, 256, 64, 64]);
                                console.log('[5] Extracted patch tensor shape: (1, 256, 64, 64)');

                                // Adjust clicks to patch coordinate space and clamp to patch bounds
                                const patchClicks = clicks.map(c => ({
                                    ...c,
                                    x: Math.max(0, Math.min(patchSize - 1, c.x - patchOffsetX)),
                                    y: Math.max(0, Math.min(patchSize - 1, c.y - patchOffsetY))
                                }));
                                console.log('[6] Adjusted clicks to patch space:', patchClicks);

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

                                        console.log('[7] Mask bounds in patch space:', [xtl, ytl, xbr, ybr]);

                                        // Translate mask bounds from patch space to image space
                                        const translatedXtl = xtl + patchOffsetX;
                                        const translatedYtl = ytl + patchOffsetY;
                                        const translatedXbr = xbr + patchOffsetX;
                                        const translatedYbr = ybr + patchOffsetY;

                                        console.log('[8] Mask bounds in image space:', [translatedXtl, translatedYtl, translatedXbr, translatedYbr]);

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
