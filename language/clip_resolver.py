# language/clip_resolver.py
# ClIP-Based natural language → semantic label resolver.
# Takes a user query like "something I can drink from"
# and returns the best matching label from your semantic map.

import torch
import clip
from PIL import Image
import numpy as np

class CLIPResolver:
    def __init__(self, device=None):
        """
        Loads CLIP ViT-B/32, we dont need B/16 for finer details.
        """
        if device is None:
            if torch.backend.mps.is_available():
                self.device = 'cpu'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Loading CLIP on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print("CLIP loaded.")

        # Cache encoded label vectors so we don't re-encode every query
        self._label_cache = {}


    #---------------------------------------------------------------------------
    # Core Encoding Functions
    #---------------------------------------------------------------------------
    def encode_text(self, text):
        """
        Encode and input string into a normalized CLIP text vector.
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad(): # no gradient needed for inference
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features # Return a normalized vector of shape (1, 512)
    
    def encode_image(self, image):
        """
        Encode a raw RGB numpy frame into a normalized CLIP image vector.
        frame_rgb: numpy array (H, W, 3) in RGB.
        """
        pil_image = Image.fromarray(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features # Return a normalized vector of shape (1, 512)
    
    #---------------------------------------------------------------------------
    # Label Resolution Functions
    #---------------------------------------------------------------------------
    def resolve_label(self, query, candidate_labels):
        """
        Given a natural language query and a list of candidate labels,
        return the label that best matches the query using CLIP.
        """
        # Encode the query
        query_vector = self.encode_text(query)

        # Encode candidate labels (cache them for efficiency)
        label_vectors = []
        for label in candidate_labels:
            if label not in self._label_cache:
                self._label_cache[label] = self.encode_text(label)
            label_vectors.append(self._label_cache[label])
        
        # Stack label vectors into a single tensor
        label_vectors = torch.cat(label_vectors, dim=0)

        # Compute cosine similarity between query and each label
        similarities = (query_vector @ label_vectors.T).squeeze(0)  # shape: (num_labels,)

        # Find the index of the best matching label
        best_index = similarities.argmax().item()
        best_label = candidate_labels[best_index]
        
        return best_label, similarities[best_index].item()
