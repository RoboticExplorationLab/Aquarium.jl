##################################################
## Gauss-Legendre quadrature (replaces adaptive quadgk)
##################################################

# 7-point Gauss-Legendre nodes and weights on [-1, 1].
# Exact for polynomials up to degree 13; highly accurate for smooth transcendental
# integrands (three-point delta products) between known breakpoints.
const _GL7_NODES = (
    -0.9491079123427585,
    -0.7415311855993945,
    -0.4058451513773972,
     0.0,
     0.4058451513773972,
     0.7415311855993945,
     0.9491079123427585,
)

const _GL7_WEIGHTS = (
    0.1294849661688697,
    0.2797053914892766,
    0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189,
    0.2797053914892766,
    0.1294849661688697,
)

# Fixed-order Gauss-Legendre quadrature on [lo, hi].
# `lo` and `hi` are Float64; `f` may return Float64 or ForwardDiff.Dual.
@inline function _gl7_integrate(f, lo::Float64, hi::Float64)
    half_w = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    s = _GL7_WEIGHTS[1] * f(half_w * _GL7_NODES[1] + mid)
    @inbounds for i in 2:7
        s += _GL7_WEIGHTS[i] * f(half_w * _GL7_NODES[i] + mid)
    end
    return half_w * s
end

##################################################
## one-point discrete deltas and derivatives
##################################################

# Analytical closed-form integral of the one-point delta product δ(ax+b)*δ(cx+d)
# over [0,1].  The integrand is piecewise quadratic between known breakpoints so
# the integral can be evaluated exactly with no numerical quadrature.
function one_point_delta_product_definite_integral(a, b, c, d)
    a_val = ForwardDiff.value(a)
    b_val = ForwardDiff.value(b)
    c_val = ForwardDiff.value(c)
    d_val = ForwardDiff.value(d)

    # Collect and sort breakpoints (Float64 for ordering, Dual-safe integration)
    breakpoints_val = sort!(Float64[0.0, 1.0,
        (-1 - b_val) / a_val, -b_val / a_val, (1 - b_val) / a_val,
        (-1 - d_val) / c_val, -d_val / c_val, (1 - d_val) / c_val])

    T = promote_type(typeof(a), typeof(b), typeof(c), typeof(d))
    result = zero(T)

    for i in 1:(length(breakpoints_val) - 1)
        x0 = breakpoints_val[i]
        x1 = breakpoints_val[i + 1]
        (x0 >= 1.0 || x1 <= 0.0 || x1 - x0 < 1e-14) && continue
        # Clamp to [0, 1]
        x0 = max(x0, 0.0)
        x1 = min(x1, 1.0)
        result += _integrate_hat_product_segment(a, b, c, d, x0, x1)
    end

    return result
end

# Integrate δ(ax+b)*δ(cx+d) over a single segment where the sign of each argument
# is constant.  The integrand is (Ax+B)(Cx+D) — a polynomial of degree ≤ 2.
function _integrate_hat_product_segment(a, b, c, d, x0, x1)
    xm = 0.5 * (x0 + x1)
    r1_val = ForwardDiff.value(a * xm + b)
    r2_val = ForwardDiff.value(c * xm + d)

    # If either delta is zero in this interval, the product is zero
    if abs(r1_val) >= 1.0 || abs(r2_val) >= 1.0
        return zero(promote_type(typeof(a), typeof(b), typeof(c), typeof(d)))
    end

    s1 = r1_val >= 0 ? 1.0 : -1.0
    s2 = r2_val >= 0 ? 1.0 : -1.0

    # f(x) = (1 - s1*(ax+b)) * (1 - s2*(cx+d))
    #      = (Ax + B)(Cx + D)  with  A=-s1*a, B=1-s1*b, C=-s2*c, D=1-s2*d
    A = -s1 * a
    B = 1 - s1 * b
    C = -s2 * c
    D = 1 - s2 * d

    AC   = A * C
    ADBC = A * D + B * C
    BD   = B * D

    return AC / 3 * (x1^3 - x0^3) + ADBC / 2 * (x1^2 - x0^2) + BD * (x1 - x0)
end

function one_point_delta_product_definite_integral_derivatives(a, b, c, d)
    # Compute all 4 derivatives in a single quadgk pass using ForwardDiff with a 4-element Dual
    # This is 4x faster than calling ForwardDiff.derivative 4 separate times

    # Create Dual numbers with unit partials for each variable
    # partials layout: [∂/∂a, ∂/∂b, ∂/∂c, ∂/∂d]
    a_dual = ForwardDiff.Dual(a, (1.0, 0.0, 0.0, 0.0))
    b_dual = ForwardDiff.Dual(b, (0.0, 1.0, 0.0, 0.0))
    c_dual = ForwardDiff.Dual(c, (0.0, 0.0, 1.0, 0.0))
    d_dual = ForwardDiff.Dual(d, (0.0, 0.0, 0.0, 1.0))

    # Compute the integral with all Dual numbers - quadgk evaluates once and propagates all partials
    integral_dual = one_point_delta_product_definite_integral(a_dual, b_dual, c_dual, d_dual)

    # Extract the partials (derivatives)
    partials = ForwardDiff.partials(integral_dual)

    return partials[1], partials[2], partials[3], partials[4]
end

function one_point_discrete_delta_product(r1, r2)
    return one_point_discrete_delta(r1) * one_point_discrete_delta(r2)
end

function one_point_discrete_delta_product_derivatives(r1, r2)
    ∂d1_∂r1 = one_point_discrete_delta_derivative(r1)
    ∂d2_∂r2 = one_point_discrete_delta_derivative(r2)

    ∂d_∂r1 = ∂d1_∂r1 * one_point_discrete_delta(r2)
    ∂d_∂r2 = one_point_discrete_delta(r1) * ∂d2_∂r2

    return ∂d_∂r1, ∂d_∂r2
end

function one_point_discrete_delta(r)
    abs_r = abs(r)
    # Branchless: multiply by boolean condition
    d = (abs_r <= 1.0) * (1.0 - abs_r)
    return d
end

function one_point_discrete_delta_derivative(r)
    abs_r = abs(r)
    # Match ForwardDiff's subgradient convention at the cusp r = 0: Julia's
    # `abs(Dual(0, 1))` returns `Dual(0, 1)` (picks the r >= 0 branch), so
    # d(1 - |r|)/dr at r = 0 is -1. Using `r >= 0 ? -1 : 1` reproduces this,
    # and matches the usual `-sign(r)` away from zero. Outside support the
    # derivative is zero.
    d_grad = (abs_r < 1.0) * (r >= 0.0 ? -1.0 : 1.0)
    return d_grad
end

##################################################
## three-point segment integrals
##################################################

function three_point_delta_product_definite_integral(a, b, c, d)
    integrand(x) = three_point_discrete_delta_product(a * x + b, c * x + d)

    a_val = ForwardDiff.value(a)
    b_val = ForwardDiff.value(b)
    c_val = ForwardDiff.value(c)
    d_val = ForwardDiff.value(d)

    # Breakpoints from three-point delta piecewise regions: |r| = 0.5 and |r| = 1.5
    breakpoints_val = sort!(Float64[0.0, 1.0,
        (-1.5 - b_val) / a_val, (-0.5 - b_val) / a_val, (0.5 - b_val) / a_val, (1.5 - b_val) / a_val,
        (-1.5 - d_val) / c_val, (-0.5 - d_val) / c_val, (0.5 - d_val) / c_val, (1.5 - d_val) / c_val,
    ])

    # Sum fixed-order GL quadrature over each subinterval between breakpoints
    T = promote_type(typeof(a), typeof(b), typeof(c), typeof(d))
    integral = zero(T)
    for i in 1:(length(breakpoints_val) - 1)
        lo = breakpoints_val[i]
        hi = breakpoints_val[i + 1]
        (lo >= 1.0 || hi <= 0.0 || hi - lo < 1e-14) && continue
        lo = max(lo, 0.0)
        hi = min(hi, 1.0)
        integral += _gl7_integrate(integrand, lo, hi)
    end

    return integral
end

function three_point_delta_product_definite_integral_derivatives(a, b, c, d)
    a_dual = ForwardDiff.Dual(a, (1.0, 0.0, 0.0, 0.0))
    b_dual = ForwardDiff.Dual(b, (0.0, 1.0, 0.0, 0.0))
    c_dual = ForwardDiff.Dual(c, (0.0, 0.0, 1.0, 0.0))
    d_dual = ForwardDiff.Dual(d, (0.0, 0.0, 0.0, 1.0))

    integral_dual = three_point_delta_product_definite_integral(a_dual, b_dual, c_dual, d_dual)

    partials = ForwardDiff.partials(integral_dual)

    return partials[1], partials[2], partials[3], partials[4]
end

##################################################
## dispatcher functions
##################################################

function delta_support_radius(kind::Symbol)
    kind === :one_point && return 1.0
    kind === :three_point && return 1.5
    error("Unknown discrete_delta_kind: $kind")
end

function discrete_delta_product(kind::Symbol, r1, r2)
    kind === :one_point && return one_point_discrete_delta_product(r1, r2)
    kind === :three_point && return three_point_discrete_delta_product(r1, r2)
    error("Unknown discrete_delta_kind: $kind")
end

function discrete_delta_product_derivatives(kind::Symbol, r1, r2)
    kind === :one_point && return one_point_discrete_delta_product_derivatives(r1, r2)
    kind === :three_point && return three_point_discrete_delta_product_derivatives(r1, r2)
    error("Unknown discrete_delta_kind: $kind")
end

##################################################
## three-point discrete deltas and derivatives
##################################################

function three_point_discrete_delta_product(r1, r2)
    return three_point_discrete_delta(r1) * three_point_discrete_delta(r2)
end

function three_point_discrete_delta_product_derivatives(r1, r2)
    ∂d1_∂r1 = three_point_discrete_delta_derivative(r1)
    ∂d2_∂r2 = three_point_discrete_delta_derivative(r2)

    ∂d_∂r1 = ∂d1_∂r1 * three_point_discrete_delta(r2)
    ∂d_∂r2 = three_point_discrete_delta(r1) * ∂d2_∂r2

    return ∂d_∂r1, ∂d_∂r2
end

function three_point_discrete_delta(r)
    abs_r = abs(r)
    arg1 = max(0.0, 1.0 - 3.0 * r^2)
    arg2 = max(0.0, 1.0 - 3.0 * (1.0 - abs_r)^2)
    region1 = (abs_r <= 0.5) * ((1.0/3.0) * (1.0 + sqrt(arg1)))
    region2 = (0.5 < abs_r <= 1.5) * ((1.0/6.0) * (5.0 - 3.0 * abs_r - sqrt(arg2)))
    return region1 + region2
end

function three_point_discrete_delta_derivative(r)
    abs_r = abs(r)
    arg1 = max(1e-300, 1.0 - 3.0 * r^2)
    arg2 = max(1e-300, -2.0 - 3.0 * r^2 + 6.0 * abs_r)
    region1 = (abs_r <= 0.5) * (-r / sqrt(arg1))
    region2 = (0.5 < abs_r <= 1.5) * ((sign(r) * 0.5) * (-1.0 + (-1.0 + abs_r) / sqrt(arg2)))
    return region1 + region2
end

@testitem "One-point discrete delta" begin
    using AquariumClosed
    using ForwardDiff

    # Partition of unity: sum of delta over all grid nodes = 1 for any point
    # For a point at fractional position r from a grid node, the one-point hat
    # function has support [-1, 1], so summing over integer shifts gives 1.
    for r in [0.0, 0.25, 0.5, 0.73, -0.4]
        total = sum(one_point_discrete_delta(r - k) for k in -3:3)
        @test total ≈ 1.0 atol=1e-14
    end

    # Support: zero outside [-1, 1]
    @test one_point_discrete_delta(1.5) == 0.0
    @test one_point_discrete_delta(-1.5) == 0.0
    @test one_point_discrete_delta(0.0) == 1.0

    # Symmetry
    @test one_point_discrete_delta(0.3) == one_point_discrete_delta(-0.3)

    # Product symmetry
    @test one_point_discrete_delta_product(0.3, 0.7) == one_point_discrete_delta_product(0.7, 0.3)

    # Positivity
    for r in range(-1.0, 1.0, length=21)
        @test one_point_discrete_delta(r) >= 0.0
    end

    # Derivative vs ForwardDiff (match ForwardDiff's subgradient at the r=0 cusp)
    for r in [0.0, 0.3, -0.7, 0.99]
        fd_deriv = ForwardDiff.derivative(one_point_discrete_delta, r)
        analytical = one_point_discrete_delta_derivative(r)
        @test analytical ≈ fd_deriv atol=1e-12
    end

    # Product derivatives vs ForwardDiff, including the r=0 cusp
    for (r1, r2) in [(0.3, -0.6), (0.0, 0.5), (0.4, 0.0), (0.0, 0.0)]
        ∂_∂r1, ∂_∂r2 = one_point_discrete_delta_product_derivatives(r1, r2)
        fd_∂r1 = ForwardDiff.derivative(x -> one_point_discrete_delta_product(x, r2), r1)
        fd_∂r2 = ForwardDiff.derivative(x -> one_point_discrete_delta_product(r1, x), r2)
        @test ∂_∂r1 ≈ fd_∂r1 atol=1e-12
        @test ∂_∂r2 ≈ fd_∂r2 atol=1e-12
    end

    # Derivative is -1 at the cusp r=0 (matching ForwardDiff's r>=0 branch)
    # and zero at |r|>=1 (outside support)
    @test one_point_discrete_delta_derivative(0.0) == -1.0
    @test one_point_discrete_delta_derivative(1.5) == 0.0
end

@testitem "Three-point discrete delta" begin
    using AquariumClosed
    using ForwardDiff

    # Partition of unity: sum of delta over all grid nodes = 1
    for r in [0.0, 0.25, 0.5, 0.73, -0.4, 1.1]
        total = sum(three_point_discrete_delta(r - k) for k in -4:4)
        @test total ≈ 1.0 atol=1e-12
    end

    # Support: zero outside [-1.5, 1.5]
    @test three_point_discrete_delta(2.0) == 0.0
    @test three_point_discrete_delta(-2.0) == 0.0
    @test three_point_discrete_delta(1.5) ≈ 0.0 atol=1e-14

    # Symmetry
    @test three_point_discrete_delta(0.3) ≈ three_point_discrete_delta(-0.3)

    # Positivity within support
    for r in range(-1.5, 1.5, length=31)
        @test three_point_discrete_delta(r) >= 0.0
    end

    # Product symmetry
    @test three_point_discrete_delta_product(0.3, 0.7) ≈ three_point_discrete_delta_product(0.7, 0.3)

    # Derivative vs ForwardDiff
    for r in [0.3, -0.7, 1.2, 0.1]
        fd_deriv = ForwardDiff.derivative(three_point_discrete_delta, r)
        analytical = three_point_discrete_delta_derivative(r)
        @test analytical ≈ fd_deriv atol=1e-10
    end

    # Product derivatives vs ForwardDiff
    r1, r2 = 0.3, -0.6
    ∂_∂r1, ∂_∂r2 = three_point_discrete_delta_product_derivatives(r1, r2)
    fd_∂r1 = ForwardDiff.derivative(x -> three_point_discrete_delta_product(x, r2), r1)
    fd_∂r2 = ForwardDiff.derivative(x -> three_point_discrete_delta_product(r1, x), r2)
    @test ∂_∂r1 ≈ fd_∂r1 atol=1e-10
    @test ∂_∂r2 ≈ fd_∂r2 atol=1e-10
end

@testitem "Discrete delta dispatchers" begin
    using AquariumClosed
    # delta_support_radius
    @test delta_support_radius(:one_point) == 1.0
    @test delta_support_radius(:three_point) == 1.5
    @test_throws Exception delta_support_radius(:bogus)

    # discrete_delta_product dispatches to correct underlying function
    r1, r2 = 0.3, -0.6
    @test discrete_delta_product(:one_point, r1, r2) == one_point_discrete_delta_product(r1, r2)
    @test discrete_delta_product(:three_point, r1, r2) == three_point_discrete_delta_product(r1, r2)
    @test_throws Exception discrete_delta_product(:bogus, r1, r2)

    # discrete_delta_product_derivatives dispatches correctly
    ∂1_op, ∂2_op = discrete_delta_product_derivatives(:one_point, r1, r2)
    ∂1_op_ref, ∂2_op_ref = one_point_discrete_delta_product_derivatives(r1, r2)
    @test ∂1_op == ∂1_op_ref
    @test ∂2_op == ∂2_op_ref

    ∂1_tp, ∂2_tp = discrete_delta_product_derivatives(:three_point, r1, r2)
    ∂1_tp_ref, ∂2_tp_ref = three_point_discrete_delta_product_derivatives(r1, r2)
    @test ∂1_tp == ∂1_tp_ref
    @test ∂2_tp == ∂2_tp_ref

    @test_throws Exception discrete_delta_product_derivatives(:bogus, r1, r2)
end

@testitem "Delta product segment integrals" begin
    using AquariumClosed
    using ForwardDiff
    using QuadGK

    # --- One-point (closed-form) vs quadgk ground truth ---
    @testset "one-point vs quadgk" begin
        test_cases = [
            (1.0, 0.0, 1.0, 0.0),    # centered on grid node
            (1.2, 0.3, 0.8, -0.2),    # general case
            (0.8, -0.2, 1.2, 0.3),    # swapped
            (1.0, 0.5, 1.0, -0.5),    # offset
            (2.0, 0.1, 0.5, 0.3),     # different scales
            (1.0, 5.0, 1.0, -5.0),    # no overlap
        ]
        for (a, b, c, d) in test_cases
            analytical = one_point_delta_product_definite_integral(a, b, c, d)
            # quadgk itself has ~1e-8 tolerance at cusps; the analytical formula is exact
            ref, _ = quadgk(x -> one_point_discrete_delta_product(a*x+b, c*x+d), 0.0, 1.0)
            @test analytical ≈ ref atol=1e-8
        end
    end

    # Known exact value: ∫₀¹ (1-x)^2 dx = 1/3
    val = one_point_delta_product_definite_integral(1.0, 0.0, 1.0, 0.0)
    @test val ≈ 1/3 atol=1e-14

    # When delta products don't overlap (far apart), integral should be zero
    val_zero = one_point_delta_product_definite_integral(1.0, 5.0, 1.0, -5.0)
    @test val_zero ≈ 0.0 atol=1e-14

    # Derivatives vs ForwardDiff
    a, b, c, d = 1.2, 0.3, 0.8, -0.2
    ∂a, ∂b, ∂c, ∂d = one_point_delta_product_definite_integral_derivatives(a, b, c, d)

    fd_∂a = ForwardDiff.derivative(x -> one_point_delta_product_definite_integral(x, b, c, d), a)
    fd_∂b = ForwardDiff.derivative(x -> one_point_delta_product_definite_integral(a, x, c, d), b)
    fd_∂c = ForwardDiff.derivative(x -> one_point_delta_product_definite_integral(a, b, x, d), c)
    fd_∂d = ForwardDiff.derivative(x -> one_point_delta_product_definite_integral(a, b, c, x), d)

    @test ∂a ≈ fd_∂a atol=1e-8
    @test ∂b ≈ fd_∂b atol=1e-8
    @test ∂c ≈ fd_∂c atol=1e-8
    @test ∂d ≈ fd_∂d atol=1e-8

    # Symmetry: swapping (a,b) <-> (c,d) should give same integral
    val1 = one_point_delta_product_definite_integral(1.2, 0.3, 0.8, -0.2)
    val2 = one_point_delta_product_definite_integral(0.8, -0.2, 1.2, 0.3)
    @test val1 ≈ val2 atol=1e-12
end

@testitem "Three-point delta product segment integrals" begin
    using AquariumClosed
    using ForwardDiff
    using QuadGK

    # --- Gauss-Legendre vs quadgk ground truth ---
    @testset "three-point vs quadgk" begin
        test_cases = [
            (1.0, 0.0, 1.0, 0.0),
            (1.2, 0.3, 0.8, -0.2),
            (0.8, -0.2, 1.2, 0.3),
            (1.0, 0.5, 1.0, -0.5),
            (2.0, 0.1, 0.5, 0.3),
            (1.0, 5.0, 1.0, -5.0),
        ]
        for (a, b, c, d) in test_cases
            gl_val = three_point_delta_product_definite_integral(a, b, c, d)
            # Both GL7 and quadgk are approximate; agree to ~1e-6
            ref, _ = quadgk(x -> three_point_discrete_delta_product(a*x+b, c*x+d), 0.0, 1.0)
            @test gl_val ≈ ref atol=1e-6
        end
    end

    # Positive for overlapping case
    val = three_point_delta_product_definite_integral(1.0, 0.0, 1.0, 0.0)
    @test val > 0.0

    # Zero when supports don't overlap
    val_zero = three_point_delta_product_definite_integral(1.0, 5.0, 1.0, -5.0)
    @test val_zero ≈ 0.0 atol=1e-14

    # Symmetry: swapping (a,b) <-> (c,d)
    val1 = three_point_delta_product_definite_integral(1.2, 0.3, 0.8, -0.2)
    val2 = three_point_delta_product_definite_integral(0.8, -0.2, 1.2, 0.3)
    @test val1 ≈ val2 atol=1e-12

    # All 4 partial derivatives vs ForwardDiff
    a, b, c, d = 1.2, 0.3, 0.8, -0.2
    ∂a, ∂b, ∂c, ∂d = three_point_delta_product_definite_integral_derivatives(a, b, c, d)

    fd_∂a = ForwardDiff.derivative(x -> three_point_delta_product_definite_integral(x, b, c, d), a)
    fd_∂b = ForwardDiff.derivative(x -> three_point_delta_product_definite_integral(a, x, c, d), b)
    fd_∂c = ForwardDiff.derivative(x -> three_point_delta_product_definite_integral(a, b, x, d), c)
    fd_∂d = ForwardDiff.derivative(x -> three_point_delta_product_definite_integral(a, b, c, x), d)

    @test ∂a ≈ fd_∂a atol=1e-8
    @test ∂b ≈ fd_∂b atol=1e-8
    @test ∂c ≈ fd_∂c atol=1e-8
    @test ∂d ≈ fd_∂d atol=1e-8
end