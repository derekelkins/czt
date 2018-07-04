## Chirp Z-transform

This is an implementation of the Chirp Z-transform in pure Kotlin. It works via FFT convolution.

The Chirp Z-transform computes

    X[k] = sum_{n=0}^{N-1} x[n] u^(-n) w^(nk) (k = 0,...,M-1)

for a complex number `u` and `w`. The effect is to evaluate the Z-transform:

    X(z) = sum_{n=0}^{N-1} x[n] z^(-n)
 
along a logarithmic spiral in the complex plane.

One major use of the Chirp Z-transform is to evaluate a subinterval of frequencies with
higher frequency resolution for the same number of inputs as an FFT.
