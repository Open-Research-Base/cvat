# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from torch.cuda.amp import autocast
from segment_anything import sam_model_registry, SamPredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam_batched/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.latest_image = None
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.predictor = SamPredictor(sam_model)

    def handle(self, image):
        self.predictor.set_image(image)
        features = self.predictor.get_image_embedding()
        return features

    def handle_batch(self, batch):
        # Convert numpy array from shape (9, 2448, 2048, 3) to torch tensor (9, 3, 2448, 2048)
        # Transpose from HWC to CHW format for each image in the batch
        print("handle batch")
        with torch.no_grad():
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
            batch_tensor = batch_tensor.to(self.device)

            batch_features = []
            for idx, item in enumerate(batch_tensor):
                print(f"Processing item {idx + 1}/{len(batch_tensor)}")
                item = item.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

                # Use autocast to handle mixed precision automatically
                with autocast(dtype=torch.float16, cache_enabled=True, enabled=torch.cuda.is_available()):
                    # Preprocess the single item
                    preprocessed_item = self.predictor.model.preprocess(item)
                    print("preprocess done, shape:", preprocessed_item.shape)

                    # Feed to image encoder
                    features = self.predictor.model.image_encoder(preprocessed_item)

                # Move to CPU immediately to free GPU memory
                batch_features.append(features.float().cpu())  # Convert back to fp32 for CPU

                # Clear GPU cache
                del item, preprocessed_item, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Concatenate all features along batch dimension (on CPU)
            batch_features = torch.cat(batch_features, dim=0)
            print(f"Extracted features from batch with shape: {batch_features.shape}")

            # Move back to GPU if needed for downstream processing
            batch_features = batch_features.to(self.device)

        return batch_features