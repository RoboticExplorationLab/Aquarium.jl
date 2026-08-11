mutable struct PDController <: Controller
    # Gains
    Kp::Real
    Kd::Real
    
    # Saturation limits
    output_min::Real
    output_max::Real

end

function PDController(
    Kp::Real,
    Kd::Real;
    output_min::Real = -Inf,
    output_max::Real = Inf
)
    return PDController(Kp, Kd, output_min, output_max)
end

function calculate_control_output(
    controller::PDController,
    setpoint::Real,
    setpoint_derivative::Real,
    input::Real,
    input_derivative::Real
)
    # Compute raw control output
    output = controller.Kp * (setpoint - input) +
        controller.Kd * (setpoint_derivative - input_derivative)
    
    # Apply saturation limits
    output = clamp(output, controller.output_min, controller.output_max)

    return output
end