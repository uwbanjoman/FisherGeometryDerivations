# srv/Operators.jl
using LinearAlgebra

export
    jordan,
    Lρ,
    Rρ,
    Lρ_matrix,
    Rρ_matrix,
    Lρ_inv,
    Lρ_inv_matrix,
    Lρ_sqrt_inv,
    dLρ,
    dLρ_inv

# ============================================================
# Jordan product
#
# Jρ(X) = (ρX + Xρ)/2
# ============================================================

"""
    jordan(A,B)

Jordan product

    A ∘ B = (AB + BA)/2
"""
function jordan(A::AbstractMatrix,B::AbstractMatrix)
    return (A*B + B*A)/2
end

# ============================================================
# Left Jordan operator
# ============================================================

"""
    Lρ(ρ,X)

Lρ(X) = (ρX + Xρ)/2
"""
function Lρ(ρ::AbstractMatrix,
            X::AbstractMatrix)

    return jordan(ρ,X)

end

# ============================================================
# Right multiplication
# (mainly useful for diagnostics)
# ============================================================

"""
    Rρ(ρ,X)

Rρ(X)=Xρ
"""
function Rρ(ρ::AbstractMatrix,
            X::AbstractMatrix)

    return X*ρ

end

# ============================================================
# Matrix representation
#
# vec(Lρ(X))
#
# ============================================================

"""
    Lρ_matrix(ρ)

Matrix representation of the Jordan operator

vec(Lρ(X)) = L * vec(X)
"""
function Lρ_matrix(ρ::AbstractMatrix)

    n=size(ρ,1)

    Iₙ=Matrix{eltype(ρ)}(I,n,n)

    return 0.5*(
        kron(ρ,Iₙ)
        +
        kron(Iₙ,transpose(ρ))
    )

end

"""
Right multiplication matrix.
"""
function Rρ_matrix(ρ::AbstractMatrix)

    n=size(ρ,1)

    Iₙ=Matrix{eltype(ρ)}(I,n,n)

    return kron(Iₙ,transpose(ρ))

end

# ============================================================
# Inverse Jordan operator
# ============================================================

"""
    Lρ_inv(ρ,X)

Solve

Lρ(Y)=X
"""
function Lρ_inv(ρ::AbstractMatrix,
                X::AbstractMatrix)

    n=size(ρ,1)

    L=Lρ_matrix(ρ)

    Y=L\vec(X)

    return reshape(Y,n,n)

end

"""
Matrix representation of Lρ^{-1}.
"""
function Lρ_inv_matrix(ρ)

    return inv(Lρ_matrix(ρ))

end

# ============================================================
# Positive square root
#
# L^{-1/2}
# ============================================================

"""
    Lρ_sqrt_inv(ρ)

Positive square root of Lρ^{-1}.
"""
function Lρ_sqrt_inv(ρ)

    Linv=Lρ_inv_matrix(ρ)

    E=eigen(Hermitian(Linv))

    return E.vectors *
           Diagonal(sqrt.(E.values)) *
           E.vectors'

end

# ============================================================
# Differential
#
# DLρ(H)(X)
#
# ============================================================

"""
    dLρ(H,X)

Directional derivative

DLρ(H)(X)
"""
function dLρ(H::AbstractMatrix,
             X::AbstractMatrix)

    return jordan(H,X)

end

# ============================================================
# Differential of inverse
#
# D(L^{-1})
#
# ============================================================

"""
    dLρ_inv(ρ,H,X)

Directional derivative of Lρ^{-1}

DL^{-1}(H)
=
-L^{-1} DL(H) L^{-1}
"""
function dLρ_inv(ρ,
                 H,
                 X)

    Y=Lρ_inv(ρ,X)

    Z=dLρ(H,Y)

    return -Lρ_inv(ρ,Z)

end
