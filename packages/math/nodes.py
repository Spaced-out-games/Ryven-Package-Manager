
from ryven.NENV import *
from math import *


class acos_Node(Node):
    """Return the arc cosine (measured in radians) of x.

The result is between 0 and pi."""
    title = 'acos_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(acos(*self.inputs))


class acosh_Node(Node):
    """Return the inverse hyperbolic cosine of x."""
    title = 'acosh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(acosh(*self.inputs))


class asin_Node(Node):
    """Return the arc sine (measured in radians) of x.

The result is between -pi/2 and pi/2."""
    title = 'asin_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(asin(*self.inputs))


class asinh_Node(Node):
    """Return the inverse hyperbolic sine of x."""
    title = 'asinh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(asinh(*self.inputs))


class atan_Node(Node):
    """Return the arc tangent (measured in radians) of x.

The result is between -pi/2 and pi/2."""
    title = 'atan_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(atan(*self.inputs))


class atan2_Node(Node):
    """Return the arc tangent (measured in radians) of y/x.

Unlike atan(y/x), the signs of both x and y are considered."""
    title = 'atan2_Node'
    init_inputs = [NodeInputBP(label = 'y'), NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(atan2(*self.inputs))


class atanh_Node(Node):
    """Return the inverse hyperbolic tangent of x."""
    title = 'atanh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(atanh(*self.inputs))


class ceil_Node(Node):
    """Return the ceiling of x as an Integral.

This is the smallest integer >= x."""
    title = 'ceil_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(ceil(*self.inputs))


class comb_Node(Node):
    """Number of ways to choose k items from n items without repetition and without order.

Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
to zero when k > n.

Also called the binomial coefficient because it is equivalent
to the coefficient of k-th term in polynomial expansion of the
expression (1 + x)**n.

Raises TypeError if either of the arguments are not integers.
Raises ValueError if either of the arguments are negative."""
    title = 'comb_Node'
    init_inputs = [NodeInputBP(label = 'n'), NodeInputBP(label = 'k')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(comb(*self.inputs))


class copysign_Node(Node):
    """Return a float with the magnitude (absolute value) of x but the sign of y.

On platforms that support signed zeros, copysign(1.0, -0.0)
returns -1.0."""
    title = 'copysign_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'y')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(copysign(*self.inputs))


class cos_Node(Node):
    """Return the cosine of x (measured in radians)."""
    title = 'cos_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(cos(*self.inputs))


class cosh_Node(Node):
    """Return the hyperbolic cosine of x."""
    title = 'cosh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(cosh(*self.inputs))


class degrees_Node(Node):
    """Convert angle x from radians to degrees."""
    title = 'degrees_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(degrees(*self.inputs))


class dist_Node(Node):
    """Return the Euclidean distance between two points p and q.

The points should be specified as sequences (or iterables) of
coordinates.  Both inputs must have the same dimension.

Roughly equivalent to:
    sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))"""
    title = 'dist_Node'
    init_inputs = [NodeInputBP(label = 'p'), NodeInputBP(label = 'q')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(dist(*self.inputs))


class erf_Node(Node):
    """Error function at x."""
    title = 'erf_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(erf(*self.inputs))


class erfc_Node(Node):
    """Complementary error function at x."""
    title = 'erfc_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(erfc(*self.inputs))


class exp_Node(Node):
    """Return e raised to the power of x."""
    title = 'exp_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(exp(*self.inputs))


class expm1_Node(Node):
    """Return exp(x)-1.

This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x."""
    title = 'expm1_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(expm1(*self.inputs))


class fabs_Node(Node):
    """Return the absolute value of the float x."""
    title = 'fabs_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(fabs(*self.inputs))


class factorial_Node(Node):
    """Find x!.

Raise a ValueError if x is negative or non-integral."""
    title = 'factorial_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(factorial(*self.inputs))


class floor_Node(Node):
    """Return the floor of x as an Integral.

This is the largest integer <= x."""
    title = 'floor_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(floor(*self.inputs))


class fmod_Node(Node):
    """Return fmod(x, y), according to platform C.

x % y may differ."""
    title = 'fmod_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'y')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(fmod(*self.inputs))


class frexp_Node(Node):
    """Return the mantissa and exponent of x, as pair (m, e).

m is a float and e is an int, such that x = m * 2.**e.
If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0."""
    title = 'frexp_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(frexp(*self.inputs))


class fsum_Node(Node):
    """Return an accurate floating point sum of values in the iterable seq.

Assumes IEEE-754 floating point arithmetic."""
    title = 'fsum_Node'
    init_inputs = [NodeInputBP(label = 'seq')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(fsum(*self.inputs))


class gamma_Node(Node):
    """Gamma function at x."""
    title = 'gamma_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(gamma(*self.inputs))


class gcd_Node(Node):
    """Greatest Common Divisor."""
    title = 'gcd_Node'
    init_inputs = []
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(gcd(*self.inputs))


class hypot_Node(Node):
    """hypot(*coordinates) -> value

Multidimensional Euclidean distance from the origin to a point.

Roughly equivalent to:
    sqrt(sum(x**2 for x in coordinates))

For a two dimensional point (x, y), gives the hypotenuse
using the Pythagorean theorem:  sqrt(x*x + y*y).

For example, the hypotenuse of a 3/4/5 right triangle is:

    >>> hypot(3.0, 4.0)
    5.0"""
    title = 'hypot_Node'
    
    
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(hypot(*self.inputs))

"""WARNING: isclose_Node arg count was found, but argument names were not. Solution has been implemented, should still work"""
class isclose_Node(Node):
    """Determine whether two floating point numbers are close in value.

  rel_tol
    maximum difference for being considered "close", relative to the
    magnitude of the input values
  abs_tol
    maximum difference for being considered "close", regardless of the
    magnitude of the input values

Return True if a is close in value to b, and False otherwise.

For the values to be considered close, the difference between them
must be smaller than at least one of the tolerances.

-inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
is, NaN is not close to anything, even itself.  inf and -inf are
only close to themselves."""
    title = 'isclose_Node'
    init_inputs = [NodeInputBP(), NodeInputBP(), NodeInputBP(), NodeInputBP()]
    init_outputs = NodeOutputBP()
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(isclose(*self.inputs))


class isfinite_Node(Node):
    """Return True if x is neither an infinity nor a NaN, and False otherwise."""
    title = 'isfinite_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(isfinite(*self.inputs))


class isinf_Node(Node):
    """Return True if x is a positive or negative infinity, and False otherwise."""
    title = 'isinf_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(isinf(*self.inputs))


class isnan_Node(Node):
    """Return True if x is a NaN (not a number), and False otherwise."""
    title = 'isnan_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(isnan(*self.inputs))


class isqrt_Node(Node):
    """Return the integer part of the square root of the input."""
    title = 'isqrt_Node'
    init_inputs = [NodeInputBP(label = 'n')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(isqrt(*self.inputs))


class lcm_Node(Node):
    """Least Common Multiple."""
    title = 'lcm_Node'
    init_inputs = []
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(lcm(*self.inputs))


class ldexp_Node(Node):
    """Return x * (2**i).

This is essentially the inverse of frexp()."""
    title = 'ldexp_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'i')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(ldexp(*self.inputs))


class lgamma_Node(Node):
    """Natural logarithm of absolute value of Gamma function at x."""
    title = 'lgamma_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(lgamma(*self.inputs))


class log_Node(Node):
    """log(x, [base=math.e])
Return the logarithm of x to the given base.

If the base not specified, returns the natural logarithm (base e) of x."""
    title = 'log_Node'
    
    
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(log(*self.inputs))


class log10_Node(Node):
    """Return the base 10 logarithm of x."""
    title = 'log10_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(log10(*self.inputs))


class log1p_Node(Node):
    """Return the natural logarithm of 1+x (base e).

The result is computed in a way which is accurate for x near zero."""
    title = 'log1p_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(log1p(*self.inputs))


class log2_Node(Node):
    """Return the base 2 logarithm of x."""
    title = 'log2_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(log2(*self.inputs))


class modf_Node(Node):
    """Return the fractional and integer parts of x.

Both results carry the sign of x and are floats."""
    title = 'modf_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(modf(*self.inputs))


class nextafter_Node(Node):
    """Return the next floating-point value after x towards y."""
    title = 'nextafter_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'y')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(nextafter(*self.inputs))


class perm_Node(Node):
    """Number of ways to choose k items from n items without repetition and with order.

Evaluates to n! / (n - k)! when k <= n and evaluates
to zero when k > n.

If k is not specified or is None, then k defaults to n
and the function returns n!.

Raises TypeError if either of the arguments are not integers.
Raises ValueError if either of the arguments are negative."""
    title = 'perm_Node'
    init_inputs = [NodeInputBP(label = 'n'), NodeInputBP(label = 'k')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(perm(*self.inputs))


class pow_Node(Node):
    """Return x**y (x to the power of y)."""
    title = 'pow_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'y')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(pow(*self.inputs))

"""WARNING: prod_Node arg count was found, but argument names were not. Solution has been implemented, should still work"""
class prod_Node(Node):
    """Calculate the product of all the elements in the input iterable.

The default start value for the product is 1.

When the iterable is empty, return the start value.  This function is
intended specifically for use with numeric values and may reject
non-numeric types."""
    title = 'prod_Node'
    init_inputs = [NodeInputBP(), NodeInputBP()]
    init_outputs = NodeOutputBP()
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(prod(*self.inputs))


class radians_Node(Node):
    """Convert angle x from degrees to radians."""
    title = 'radians_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(radians(*self.inputs))


class remainder_Node(Node):
    """Difference between x and the closest integer multiple of y.

Return x - n*y where n*y is the closest integer multiple of y.
In the case where x is exactly halfway between two multiples of
y, the nearest even value of n is used. The result is always exact."""
    title = 'remainder_Node'
    init_inputs = [NodeInputBP(label = 'x'), NodeInputBP(label = 'y')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(remainder(*self.inputs))


class sin_Node(Node):
    """Return the sine of x (measured in radians)."""
    title = 'sin_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(sin(*self.inputs))


class sinh_Node(Node):
    """Return the hyperbolic sine of x."""
    title = 'sinh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(sinh(*self.inputs))


class sqrt_Node(Node):
    """Return the square root of x."""
    title = 'sqrt_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(sqrt(*self.inputs))


class tan_Node(Node):
    """Return the tangent of x (measured in radians)."""
    title = 'tan_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(tan(*self.inputs))


class tanh_Node(Node):
    """Return the hyperbolic tangent of x."""
    title = 'tanh_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(tanh(*self.inputs))


class trunc_Node(Node):
    """Truncates the Real x to the nearest Integral toward 0.

Uses the __trunc__ magic method."""
    title = 'trunc_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(trunc(*self.inputs))


class ulp_Node(Node):
    """Return the value of the least significant bit of the float x."""
    title = 'ulp_Node'
    init_inputs = [NodeInputBP(label = 'x')]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    
    def update_event(self, inp = -1):
        		self.set_output_value(ulp(*self.inputs))

math_nodes = [acos_Node, acosh_Node, asin_Node, asinh_Node, atan_Node, atan2_Node, atanh_Node, ceil_Node, comb_Node, copysign_Node, cos_Node, cosh_Node, degrees_Node, dist_Node, erf_Node, erfc_Node, exp_Node, expm1_Node, fabs_Node, factorial_Node, floor_Node, fmod_Node, frexp_Node, fsum_Node, gamma_Node, gcd_Node, hypot_Node, isclose_Node, isfinite_Node, isinf_Node, isnan_Node, isqrt_Node, lcm_Node, ldexp_Node, lgamma_Node, log_Node, log10_Node, log1p_Node, log2_Node, modf_Node, nextafter_Node, perm_Node, pow_Node, prod_Node, radians_Node, remainder_Node, sin_Node, sinh_Node, sqrt_Node, tan_Node, tanh_Node, trunc_Node, ulp_Node]
export_nodes(*math_nodes)