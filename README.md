# DJL
AI DJ, Drop The Beat.

## Network To Try
### [MelNet](https://sjvasquez.github.io/blog/melnet/)
Able to generate high-fidelity audio samples that capture structure at timescales that time-domain models have yet to achieve. Powered by Recurrent Convolutional Neural Network.
- Paper: https://arxiv.org/pdf/1906.01083.pdf
- Instead of modeling a 1-D time-domain wave form, we can model a 2-D time-frequency representation: Spectrogram.
  - **Spectrogram**: a picture of sound. The x-axis is Time, the y-axis is Frequency, and the brightness (lightness or darkness) at any given point represents the energy at that point.

### [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
The result isn't ideal, the sound is still too robotic.
- Paper: https://arxiv.org/abs/1609.03499
