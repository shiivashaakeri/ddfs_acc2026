# control module
from .excitation import (
    generate_excitation,
    generate_excitation_sequence,
    generate_sinusoidal_excitation,
    verify_pe_condition,
    compute_singular_values,
    estimate_excitation_quality,
    ExcitationGenerator,
)

from .segmented_controller import (
    SegmentedFunnelController,
    design_initial_controller,
    create_controller_from_params,
)
