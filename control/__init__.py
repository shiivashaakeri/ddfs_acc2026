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

# segmented_controller will be imported when implemented
# from .segmented_controller import SegmentedFunnelController, design_initial_controller
