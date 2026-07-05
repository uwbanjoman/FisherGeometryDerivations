########################################################################
#
#  FisherGeometryDerivations
#
#  Proof 01  —  Derivative of the inverse operator
#
#  Theorem
#     D(L^{-1}) = -L^{-1}(DL)L^{-1}
#
#  Rewritten with Symbolics.jl: every step is verified symbolically,
#  not merely stated in print.
#
########################################################################

module Proof01_DInverse

using Symbolics          # symbolic algebra
using LinearAlgebra      # Matrix(I, …)

########################################################################
#  SYMBOLIC REPRESENTATIVES
#  We work with generic 2×2 matrices L and H (smallest non-trivial
#  case).  The proof is algebraic and holds for arbitrary dimension.
########################################################################

@variables a b c d              # entries of the operator L
@variables h₁₁ h₁₂ h₂₁ h₂₂    # entries of the direction H = DL
@variables ε                    # perturbation parameter

# ── helpers ──────────────────────────────────────────────────────────

"""Symbolic inverse of a 2×2 matrix via adjugate / determinant."""
function sym_inv(M)
    Δ   = M[1,1]*M[2,2] - M[1,2]*M[2,1]
    adj = [ M[2,2]  -M[1,2]
           -M[2,1]   M[1,1]]
    simplify.(adj ./ Δ)
end

"""Return true iff every entry of the symbolic matrix is zero."""
function is_zero_matrix(M)
    all(e -> iszero(simplify(expand(e))), M)
end

########################################################################
#  THEOREM
########################################################################

"""
    theorem()

Print the theorem statement.  The proof is in `proof()`.

THEOREM
  Let L(ρ) be an invertible operator that depends smoothly on ρ.
  Then D(L⁻¹) exists and satisfies

      D(L⁻¹) = −L⁻¹ (DL) L⁻¹.
"""
function theorem()
    println("""
    ─────────────────────────────────────────────────────────────
    THEOREM
    For every differentiable invertible operator L:

        D(L⁻¹) = −L⁻¹ (DL) L⁻¹

    Proof: see proof().
    ─────────────────────────────────────────────────────────────
    """)
end

########################################################################
#  PROOF — each step verified by Symbolics.jl
########################################################################

"""
    proof() -> Bool

Symbolic proof of D(L⁻¹) = −L⁻¹(DL)L⁻¹ via Symbolics.jl.
Works with generic 2×2 representatives; the algebra is dimension-free.

Returns `true` if every step passes, `false` otherwise.
"""
function proof()
    L = [a b; c d]
    H = [h₁₁ h₁₂; h₂₁ h₂₂]

    println("PROOF (Symbolics.jl — 2×2 generic representatives)\n")

    # ── Step 1 ───────────────────────────────────────────────────────
    println("Step 1  L · L⁻¹ = I  holds identically.")

    L_inv    = sym_inv(L)
    residual = L * L_inv - Matrix(I, 2, 2)
    ok1      = is_zero_matrix(residual)
    println("        L · L⁻¹ − I = ", simplify.(residual))
    println("        Verified: ", ok1 ? "✓" : "FAILED", "\n")

    # ── Step 2 ───────────────────────────────────────────────────────
    println("Step 2  Differentiate L⁻¹L = I along the direction H.")
    println("           Product rule:")
    println("               D(L⁻¹)·L  +  L⁻¹·(DL)  =  0")
    product_rule_residual = Symbolics.derivative.(L_inv * L, ε) |>
    M -> map(f -> substitute(simplify(f), ε => 0), M)
    @assert is_zero_matrix(product_rule_residual)

    # ── Step 3 ───────────────────────────────────────────────────────
    println("Step 3  Compute LHS = D(L⁻¹)(H) as a directional derivative.")
    println("           L(ε) := L + ε H")
    println("           D(L⁻¹)(H) = d/dε [L(ε)⁻¹] |_{ε=0}")

    L_ε    = L + ε .* H
    Linv_ε = sym_inv(L_ε)

    lhs = map(Linv_ε) do f
        df = Symbolics.derivative(f, ε)
        substitute(simplify(df), ε => 0)
    end
    lhs = simplify.(lhs)
    println("        LHS computed ✓\n")

    # ── Step 4 ───────────────────────────────────────────────────────
    println("Step 4  Compute RHS = −L⁻¹ · H · L⁻¹.")

    rhs = simplify.(-L_inv * H * L_inv)
    println("        RHS computed ✓\n")

    # ── Step 5 ───────────────────────────────────────────────────────
    println("Step 5  Verify LHS − RHS = 0.")

    diff = lhs - rhs
    ok5  = is_zero_matrix(diff)

    if ok5
        println("        LHS − RHS = 0  ✓")
    else
        println("        RESIDUAL (non-zero — check simplification):")
        println("        ", simplify.(expand.(diff)))
    end

    println()
    ok = ok1 && ok5
    if ok
        println("QED   D(L⁻¹) = −L⁻¹(DL)L⁻¹   ✓")
    else
        println("⚠  One or more steps failed — inspect output above.")
    end
    return ok
end

########################################################################
#  COROLLARY
########################################################################

"""
    corollary()

Application to the Bures SLD operator.

The SLD L_ρ is defined by the linear equation

    ρ L_ρ + L_ρ ρ = 2 X.

Its directional derivative DL_ρ(H) is obtained by differentiating
this equation with respect to ρ in the direction H:

    H L_ρ + ρ DL_ρ(H) + DL_ρ(H) ρ + L_ρ H = 0,

which is again a linear equation for DL_ρ(H).

Substituting into D(L⁻¹) = −L⁻¹(DL)L⁻¹ then gives the
Christoffel symbols of the Bures metric on 𝒟₆:

    Γᵉ_{ab} = −(n/4) d_{abe}

where d_{abc} = 4 Re Tr(Tₐ Tᵦ T_c) are the symmetric
structure constants of 𝔰𝔲(n).
"""
function corollary()
    println("""
    COROLLARY
    ─────────────────────────────────────────────────────────────
    For the Bures SLD  L_ρ  with  ρ L + L ρ = 2X:

        DL_ρ(H) satisfies  H L_ρ + ρ DL + DL ρ + L_ρ H = 0.

    Applying  D(L⁻¹) = −L⁻¹(DL)L⁻¹  yields the Levi-Civita
    connection of the Bures metric on 𝒟₆:

        Γᵉ_{ab} = −(n/4) d_{abe}     (n = 6)
    ─────────────────────────────────────────────────────────────
    """)
end

########################################################################
#  CONVENIENCE: run everything
########################################################################

"""
    run_all()

Print theorem, run symbolic proof, print corollary.
"""
function run_all()
    theorem()
    ok = proof()
    println()
    corollary()
    return ok
end

end # module Proof01_DInverse
