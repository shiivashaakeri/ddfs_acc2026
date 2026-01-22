"""
2-DoF Robot Arm Dynamics.

Implements the continuous-time dynamics for a planar 2-DoF robot arm
with standard rigid-body dynamics (Spong et al., 2006).

State: x = [q1, q2, q1_dot, q2_dot]^T (4-dim)
Input: u = [tau1, tau2]^T (2-dim)

Key equations:
    x_dot = [q_dot; M(q)^{-1}(tau - C(q,q_dot)q_dot - G(q) - B*q_dot)]

    M(q) = [[a+2b*cos(q2), d+b*cos(q2)], [d+b*cos(q2), d]]
    where:
        a = I1 + I2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
        b = m2*l1*lc2
        d = I2 + m2*lc2^2
"""

# TODO: Implement RobotArm2DoF class
# class RobotArm2DoF:
#     def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, b1, b2, g=9.81):
#     def mass_matrix(self, q) -> np.ndarray:
#     def coriolis_matrix(self, q, qdot) -> np.ndarray:
#     def gravity_vector(self, q) -> np.ndarray:
#     def friction_matrix(self) -> np.ndarray:
#     def dynamics(self, x, u) -> np.ndarray:
