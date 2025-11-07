# Generalized Sinusoidal Embeddings
We introduce generalized sinusoidal positional embeddings for arbitrary N-dimensional data with automatic optimization of frequency band allocation. This approach extends the original formulation from “Attention Is All You Need” (Vaswani et al., 2017) to 2D, 3D, and higher dimensions, making it applicable to images, videos, volumetric data, and even higher-dimensional modalities. The method jointly allocates frequency bands across dimensions and adapts the frequency schedule to the embedding size and maximal sequence length.

## Examples for 1D, 2D, 3D Positional Embeddings
<img width="640" height="480" alt="1d_tsne" src="https://github.com/user-attachments/assets/66a6915e-4623-42ab-a2b8-0352bf11cadd" />
<img width="640" height="480" alt="2d_tsne" src="https://github.com/user-attachments/assets/abc7e5cb-692c-45d8-a503-0089c03df37c" />
<img width="640" height="480" alt="3d_tsne" src="https://github.com/user-attachments/assets/d194c44b-165d-4685-9c10-d0e85ac98bd9" />

## More Robust than Original
By automatically optimizing frequency band allocation, we achieve significantly more robust positional encodings compared to the original formulation of Vaswani et al., as demonstrated by t-SNE visualizations of noise-perturbed embeddings.<img width="1008" height="533" alt="noise_resilience" src="https://github.com/user-attachments/assets/409769d1-9a28-4815-8000-abb63cbd69da" />

