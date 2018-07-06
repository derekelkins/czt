package com.hedonisticlearning.czt
import kotlin.math.*

/**
A straightforward implementation of complex numbers.
*/
data class Complex(val real: Float, val imag: Float) {
    inline operator fun plus(other: Complex): Complex = Complex(real + other.real, imag + other.imag)
    inline operator fun plus(other: Float): Complex = Complex(real + other, imag)
    inline operator fun minus(other: Complex): Complex = Complex(real - other.real, imag - other.imag)
    inline operator fun minus(other: Float): Complex = Complex(real - other, imag)
    inline operator fun times(other: Complex): Complex = Complex(real * other.real - imag * other.imag, real * other.imag + imag * other.real)
    inline operator fun times(other: Float): Complex = Complex(real * other, imag * other)
    inline operator fun div(other:Complex): Complex = this * other.recip()
    inline operator fun div(other:Float): Complex = Complex(real / other, imag / other)
    inline operator fun unaryMinus(): Complex = Complex(-real, -imag)

    inline fun normSquared(): Float = real*real + imag*imag

    inline fun norm(): Float = sqrt(normSquared())

    inline fun normalize(): Complex = this / norm()

    inline fun conjugate(): Complex = Complex(real, -imag)

    inline fun recip(): Complex = (1.0f / normSquared()) * conjugate()

    inline fun sqrt(): Complex {
        val r = norm()
        return sqrt(r) * (this + r).normalize()
    }

    fun pow(n: Int): Complex {
        return when {
            n < 0 -> recip().pow(-n)
            n == 0 -> Complex(1.0f, 0.0f)
            n == 1 -> this
            n % 2 == 0 -> {
                val w = pow(n/2)
                w * w
            }
            else -> {
                val w = pow(n/2)
                w * w * this
            }
        }
    }

    companion object {
        inline fun cis(theta: Float): Complex = Complex(cos(theta), sin(theta))
        val zero = Complex(0.0f, 0.0f)
        val one = Complex(1.0f, 0.0f)
        val i = Complex(0.0f, 1.0f)
    }
}

inline operator fun Float.times(other: Complex): Complex = Complex(this * other.real, this * other.imag)
inline operator fun Float.plus(other: Complex): Complex = Complex(this + other.real, other.imag)
inline operator fun Float.minus(other: Complex): Complex = Complex(this - other.real, other.imag)

internal val fftTwiddleFactors = Array<Complex>(32, { Complex.cis(-2*PI.toFloat()/(1 shl it).toFloat()) })

// Output is in bit reversed order.
internal fun fft(x: Array<Complex>, pow2: Int) {
    assert(x.size == 1 shl pow2)
    val szm1 = x.size - 1
    for(s in pow2 downTo 1) {
        val m = 1 shl s
        val m2 = m shr 1
        val w_m = fftTwiddleFactors[s]
        for(k in 0 .. szm1 step m) {
            var w = Complex(1.0f, 0.0f)
            for(j in 0 .. m2-1) {
                val u = x[k + j]
                val t = x[k + j + m2]
                x[k + j] = u + t
                x[k + j + m2] = w*(u - t)
                w = w * w_m
            }
        }
    }
}

internal val ifftTwiddleFactors = Array<Complex>(32, { Complex.cis(2*PI.toFloat()/(1 shl it).toFloat()) })

// Expects input in bit reversed order.
internal fun ifft(x: Array<Complex>, pow2: Int) {
    assert(x.size == 1 shl pow2)
    val szm1 = x.size - 1
    for(s in 1 .. pow2) {
        val m = 1 shl s
        val m2 = m shr 1
        val w_m = ifftTwiddleFactors[s]
        for(k in 0 .. szm1 step m) {
            var w = Complex(1.0f, 0.0f)
            for(j in 0 .. m2-1) {
                val u = x[k + j]
                val t = w*x[k + j + m2]
                x[k + j] = u + t
                x[k + j + m2] = u - t
                w = w * w_m
            }
        }
    }
}

// Let N = x.size and M = size. Increasing M increases the frequency resolution
// of the output, but clearly this is also impacted by N. I believe that for a
// given M, the minimum N to achieve the desired resolution is the size of the DFT
// to achieve the desired frequency resolution, i.e. SAMPLE_RATE / resolution,
// divided by the proportion of the unit circle that the CZT covers. For example,
// the DFT of a real signal has redundant data as the upper half of the unit circle
// is mirrored in the lower half. We can double the frequency resolution by using a
// CZT that only traverses the upper half of the unit circle.

internal fun smallestPowerOf2GreaterThan(n: Int): Int {
    var p = 1
    while(1 shl p < n) { ++p }
    return p
}

// Chirp Z-transform
// The Z-transform of x is X(z) = sum_{n=0}^{N-1} x[n] z^(-n).
// If we define X[k] = X(uw^(-k)) for k = 0,...,M-1, we get
//      X[k] = sum_{n=0}^{N-1} x[n] u^(-n) w^nk (k = 0,...,M-1)
// Noting that nk = -(k-n)^2/2 + n^2/2 + k^2/2, we can turn that into
//      X[k] = w^(k^2/2) sum_{n=0}^{N-1} (x[n] u^(-n) w^(n^2/2)) w^(-(k-n)^2) (k = 0,...,M-1)
// which corresponds to modulating x by u^(-n) w^(n^2/2) producing a[n] = x[n] u^(-n) w^(-n^2/2)
// then convolving with b[n] = w^(-n^2/2) and finally modulating
// the result (a*b) (where * is convolution) by w^(n^2/2).
// That is, X[k] = b[k]^* (a * b)[k] (k = 0,...,M-1) where z^* is the conjugate of z.
// The convolution can be computed with the FFT convolution algorithm zero padding a
// to a length of L >= N+M-1 via a[n] = 0 for n >= N and extending b with b[n] = w^(-n^2/2) for
// n = 0,...,M-1, b[n] = 0 for n = M,...,L-N+1, and b[n] = w^(-(L-n)^2/2) for n = L-N+1,...,L-1.
// Due to the n^2, we must have b[-n] = b[n] and the negative indices get remapped to b[L-n].
// uw^(-k) describes points at uniform angular displacements on a logarithmic spiral starting at u.
// In particular, if |u|=1=|w|, then this describes M points on an arc of the unit circle starting
// at u and ending at uw^(1-M). That is, we can compute an interval of the DFT with higher angular
// resolution for about the same amount of work. Instead of placing M points uniformly around the
// entire unit circle when calculating the DFT, we place M points uniformly within an arc. (Another
// application, less important here, is that this lets us calculate DFTs of arbitrary size in terms
// of power-of-two sized FFTs.) Altogether, we modulate the first M points of the L point FFT convolution
// given an N point input.
/**
Creates an object that will transform arrays of Complex numbers or Floats of size n into an array of Complex numbers
of size m. It calculates:

    X[k] = sum_{i=0}^{n-1} x[i] u^(-i) w^(ik) (k = 0,...,m-1)

where x is the input and X is the output.

*/
class CZT(u: Complex, w: Complex, private val n: Int, private val m: Int) {
    private val pow2 = smallestPowerOf2GreaterThan(n+m-1)
    private val l = 1 shl pow2

    private val a = Array<Complex>(n, { u.pow(-it) * w.sqrt().pow(it*it) })
    private val b = Array<Complex>(l, {when { it < m -> w.sqrt().pow(-it*it); it > l-n -> w.sqrt().pow(-(l-it)*(l-it)); else -> Complex.zero }})
    private val c = Array<Complex>(m, { 1.0f/l.toFloat() * w.sqrt().pow(it*it) })

    init {
        fft(b, pow2)
    }

    private val buffer = Array<Complex>(l, { Complex.zero })

    /**
        Performs the Chirp Z-transform to the input which must be of size n and produces
        the an array of size m containing the result.
    */
    fun transform(x: Array<Complex>): Array<Complex> {
        assert(x.size == n)

        for(i in buffer.indices) {
            buffer[i] = if (i < n) a[i]*x[i] else Complex.zero
        }

        fft(buffer, pow2)

        for(i in buffer.indices) {
            buffer[i] *= b[i]
        }

        ifft(buffer, pow2)

        return Array(m, { buffer[it]*c[it] })
    }

    /**
        Performs the Chirp Z-transform to the input which must be of size n and produces
        the an array of size m containing the result.
    */
    fun transform(x: FloatArray): Array<Complex> {
        assert(x.size == n)

        for(i in buffer.indices) {
            buffer[i] = if (i < n) a[i]*x[i] else Complex.zero
        }

        fft(buffer, pow2)

        for(i in buffer.indices) {
            buffer[i] *= b[i]
        }

        ifft(buffer, pow2)

        return Array(m, { buffer[it]*c[it] })
    }

    fun transformInto(x: Array<Complex>, output: Array<Complex>) {
        assert(x.size == n)
        assert(output.size == m)

        for(i in buffer.indices) {
            buffer[i] = if (i < n) a[i]*x[i] else Complex.zero
        }

        fft(buffer, pow2)

        for(i in buffer.indices) {
            buffer[i] *= b[i]
        }

        ifft(buffer, pow2)

        for(i in output.indices) {
            output[i] = buffer[i] * c[i]
        }
    }

    fun transformInto(x: FloatArray, output: Array<Complex>) {
        assert(x.size == n)
        assert(output.size == m)

        for(i in buffer.indices) {
            buffer[i] = if (i < n) a[i]*x[i] else Complex.zero
        }

        fft(buffer, pow2)

        for(i in buffer.indices) {
            buffer[i] *= b[i]
        }

        ifft(buffer, pow2)

        for(i in output.indices) {
            output[i] = buffer[i] * c[i]
        }
    }

    companion object {
        /**
            Produces a CZT for the common case of evaluating along an arc of the unit circle in the complex plain.
            CZT.arc(0.0f, 2*PI.toFloat(), n) will compute the n-point discrete Fourier transform.
        */
        fun arc(startFreq: Float, endFreq: Float, n: Int): CZT = CZT(Complex.cis(startFreq), Complex.cis((startFreq-endFreq)/n.toFloat()), n, n)
    }
}

/*
fun main(args: Array<String>) {
    val x = floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f)
    dft(x.map { Complex(it, 0.0f) }.toTypedArray()).forEach { println(it) }
    val transformer = CZT.arc(0.0f, 2*PI.toFloat(), x.size)
    println("")
    transformer.transform(x).forEach { println(it) }
}

internal fun zt(x: Array<Complex>, z: Complex): Complex {
    val w = z.recip()
    var acc = Complex(1.0f, 0.0f)
    var y = Complex(0.0f, 0.0f)
    for(i in x.indices) {
        y += x[i]*acc
        acc *= w
    }
    return y
}

internal fun dft(x: Array<Complex>): Array<Complex> {
    return Array<Complex>(x.size, { i ->
        zt(x, Complex.cis(2*PI.toFloat()*i.toFloat()/x.size.toFloat()))
    })
}

internal fun czt(x: Array<Complex>, start: Float, end: Float, size: Int): Array<Complex> {
    val u = Complex.cis(start)
    val w = Complex.cis((end - start) / size.toFloat())
    return Array<Complex>(size, { zt(x, u*w.pow(-it)) })
}
*/
