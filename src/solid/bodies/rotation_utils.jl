using Rotations

#############################################################################################
## Rotation matrix and its Jacobian
#############################################################################################

function rotation_matrix(θ::Real)
    
    return [cos(θ) -sin(θ); sin(θ) cos(θ)]

end

function rotation_matrix_jacobian(θ::Real)

    return [-sin(θ) -cos(θ); cos(θ) -sin(θ)]

end