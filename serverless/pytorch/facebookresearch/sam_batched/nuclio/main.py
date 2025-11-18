# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import numpy as np
import time
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    start = time.perf_counter()
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    image = np.array(image)

    # Extract image patches to compose a batch
    height, width = image.shape[:2]
    patch_size = 1024

    # Define positions for 9 patches
    positions = {}
    top_y = 0
    center_y = (height - patch_size) // 2
    bottom_y = height - patch_size
    left_x = 0
    center_x = (width - patch_size) // 2
    right_x = width - patch_size
    # Top row
    positions['top_left'] = (top_y, left_x)
    positions['top_center'] = (top_y, center_x)
    positions['top_right'] = (top_y, right_x)
    # Middle row
    positions['left_center'] = (center_y, left_x)
    positions['center'] = (center_y, center_x)
    positions['right_center'] = (center_y, right_x)
    # Bottom row
    positions['bottom_left'] = (bottom_y, left_x)
    positions['bottom_center'] = (bottom_y, center_x)
    positions['bottom_right'] = (bottom_y, right_x)

    # Create batch of patches
    batch = []
    for name, (y, x) in positions.items():
        patch = image[y:y+patch_size, x:x+patch_size, :]
        batch.append(patch)
    batch = np.stack(batch, axis=0)

    # Feed batch to the encoder model to retrieve the features
    features = context.user_data.model.handle_batch(batch)

    # Convert features to numpy
    features_numpy = features.cpu().numpy() if features.is_cuda else features.numpy()

    # Encode to base64
    features_encoded = base64.b64encode(features_numpy).decode()

    # Create response body
    response_data = {'blob': features_encoded}

    # Convert to JSON
    response_body = json.dumps(response_data)

    end = time.perf_counter()
    context.logger.info(f"Total processing time: {end - start:.2f} seconds")

    return context.Response(body=response_body,
        headers={},
        content_type='application/json',
        status_code=200
    )