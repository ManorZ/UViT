# UViT: Unstructured Vision Transformer.
Unstructured patch-based model, tokenizing image via interest points detection.


# [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
# [PyTorch implementation](https://huggingface.co/docs/transformers/model_doc/vit)


## 06/05/2022
1. Assumption: the information is not distributed uniformly across an image.
2. "Classic" convolution doesn't take advantage of (1), and acts over the all patches/pixels of the input image.
3. Transformers are able to act over inputs with varying length ("sentences").
4. Transformers are able to learn the relations of all the tokens in the input "sentence", even when tokens are not close to each other. \
4.1 In case of images; transformer is able to infer these relations in a fixed computational complexity (quadratic in the "sentence" length), 
  while CNNs require deeper architectures to increase the receptive field and allow far patches to "see" each other. \
5. "classic" CV algorithms are able to detect interest points in image. 
6. Assumption: interest points describe areas with more "relevant" information (requires a better formulation).
7. "Classic" convolution: locality, translation invariant. 
8. Natural images hold (7); near-by pixels are correlative, and many CV tasks require translation-invariance (e.g: classification). \
8.1 Some CV tasks may not benefit from the above, for e.g: image-to-text. specific e.g: Alice is looking down on Bob - location matters!
9. Transformers don't have the inductive bias as in (7) since they learn the "all tokens relations".
10. Regular ordering of patches of an image and/or using a spatial encoding (as in ViT) introduce an inductive bias as in (7) to transformer-based models.

**Motivation:** Can we train a transformer over patches on an input image describing "interest points" and constructing an unordered set? This including no kind of spatial encoding.
- Patches don't have to include the full input image - computationally efficient.
- Assuming the resulting model will be "free" of any inductive bias, which may benefit some CV tasks (classification probably not...).