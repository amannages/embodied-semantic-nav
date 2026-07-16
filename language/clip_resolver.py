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
    def resolve_label(self, query, candidate_labels, verbose=True):
        """
        Given a natural language query and a list of candidate object labels,
        return the label that best matches the query semantically.

        It works by the following:
        1. Encode the query into a CLIP text vector.
        2. Encode each label into a CLIP text vector (with a prompt template).
        3. Compute cosine similarity between the query vector and each label vector.
        4. Return the label with the highest similarity score.

        candidate_labels: list of strings representing the semantic map's known labels
        Returns: (best_label, confidence_score, full_ranking)
        """
        if not candidate_labels:
            return None, 0.0, []
        
        # Encode the query
        query_vector = self.encode_text(query)

        # Encode each candidate label (with a prompt template)
        # We use a prompt template to give CLIP context:
        # "a photo of a mug" works better than just "mug"
        # because CLIP was trained on image captions, not bare nouns
        label_vectors = []
        for label in candidate_labels:
            if label not in self._label_cache:
                prompted = f"a photo of a {label.lower()} in a kitchen"
                self._label_cache[label] = self.encode_text(prompted)
            label_vectors.append(self._label_cache[label])
        
        # Stack into a single matrix: (num_labels, 512)
        label_matrix = torch.cat(label_vectors, dim=0)

        # Cosine similarity: query (1, 512) @ label_matrix.T (512, num_labels)
        similarities = (query_vector @ label_matrix.T).squeeze(0)
        similarities = similarities.cpu().numpy()

        # Build ranking
        ranking = sorted(
            zip(candidate_labels, similarities.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        best_label, best_score = ranking[0]

        if verbose:
            print(f"\nCLIP Query: '{query}'")
            print(f"Top matches:")
            for label, score in ranking[:5]:
                bar = "█" * int(score * 50)
                print(f"  {label:20s} {score:.3f} {bar}")

        return best_label, best_score, ranking
    
    #---------------------------------------------------------------------------
    # Image-Based Label Resolution Functions
    #---------------------------------------------------------------------------
    def resolve_from_image(self, image, candidate_labels, verbose=True):
        """
        Given a raw RGB image and a list of candidate object labels,
        return the label that best matches the image semantically.

        It works by the following:
        1. Encode the image into a CLIP image vector.
        2. Encode each label into a CLIP text vector (with a prompt template).
        3. Compute cosine similarity between the image vector and each label vector.
        4. Return the label with the highest similarity score.

        candidate_labels: list of strings representing the semantic map's known labels
        Returns: (best_label, confidence_score, full_ranking)
        """
        if not candidate_labels:
            return None, 0.0, []
        
        # Encode the image
        image_vector = self.encode_image(image)

        # Encode each candidate label (with a prompt template)
        label_vectors = []
        for label in candidate_labels:
            if label not in self._label_cache:
                prompted = f"a photo of a {label.lower()} in a kitchen"
                self._label_cache[label] = self.encode_text(prompted)
            label_vectors.append(self._label_cache[label])
        
        # Stack into a single matrix: (num_labels, 512)
        label_matrix = torch.cat(label_vectors, dim=0)

        # Cosine similarity: image (1, 512) @ label_matrix.T (512, num_labels)
        similarities = (image_vector @ label_matrix.T).squeeze(0)
        similarities = similarities.cpu().numpy()

        # Build ranking
        ranking = sorted(
            zip(candidate_labels, similarities.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        best_label, best_score = ranking[0]

        if verbose:
            print(f"\nImage-based resolution:")
            print(f"Top matches:")
            for label, score in ranking[:5]:
                bar = "█" * int(score * 50)
                print(f"  {label:20s} {score:.3f} {bar}")

        return best_label, best_score, ranking

    