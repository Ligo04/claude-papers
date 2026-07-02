"""
ReActor Bilevel Optimization - Simplified Demo

This demonstrates the core bilevel optimization concept from ReActor.
Simplified for educational purposes - not production code.
"""

import numpy as np
from typing import Tuple, Dict


class SimplifiedReActor:
    """Simplified ReActor implementation for understanding bilevel optimization."""

    def __init__(self, n_joints: int = 10, n_frames: int = 100):
        """
        Args:
            n_joints: Number of joints in the character
            n_frames: Number of frames in the motion
        """
        self.n_joints = n_joints
        self.n_frames = n_frames

        # Retargeting parameters (upper level)
        # Constant offset per joint (axis-angle representation)
        self.theta = np.zeros((n_joints, 3))

        # Policy parameters (lower level) - simplified as linear policy
        self.policy_weights = np.random.randn(n_joints * 2, n_joints) * 0.01

        # Original motion (source)
        self.q_original = self._generate_sample_motion()

    def _generate_sample_motion(self) -> np.ndarray:
        """Generate sample sinusoidal motion for demo."""
        t = np.linspace(0, 4 * np.pi, self.n_frames)
        motion = np.zeros((self.n_frames, self.n_joints))
        for i in range(self.n_joints):
            motion[:, i] = np.sin(t + i * 0.5) * 0.5
        return motion

    def retarget_motion(self, theta: np.ndarray) -> np.ndarray:
        """
        Apply retargeting with current parameters.

        Args:
            theta: Retargeting parameters (n_joints, 3)

        Returns:
            Adapted motion (n_frames, n_joints)
        """
        # Nominal retargeting (simplified: just copy)
        q_nominal = self.q_original.copy()

        # Apply learnable offsets (constant per joint)
        # In real ReActor, this would be axis-angle rotation
        # Here simplified as additive offset
        q_adapted = q_nominal + theta[:, 0]  # Use first component

        return q_adapted

    def compute_motion_loss(self, q_adapted: np.ndarray) -> float:
        """
        Upper level loss: how different is adapted motion from original?

        Args:
            q_adapted: Adapted motion

        Returns:
            Motion loss (scalar)
        """
        # Pose deviation
        pose_loss = np.mean((q_adapted - self.q_original) ** 2)

        # Velocity smoothness
        vel_adapted = np.diff(q_adapted, axis=0)
        vel_original = np.diff(self.q_original, axis=0)
        vel_loss = np.mean((vel_adapted - vel_original) ** 2)

        return pose_loss + 0.1 * vel_loss

    def train_policy(self, q_ref: np.ndarray, n_steps: int = 100) -> Dict:
        """
        Lower level optimization: train policy to track reference.

        Args:
            q_ref: Reference motion to track
            n_steps: Number of training steps

        Returns:
            Training info dict
        """
        learning_rate = 0.01

        for step in range(n_steps):
            # Simplified policy: linear mapping from (q, q_ref) to action
            tracking_errors = []

            for t in range(len(q_ref) - 1):
                # Current state (simplified)
                q_current = q_ref[t] + np.random.randn(self.n_joints) * 0.1
                state = np.concatenate([q_current, q_ref[t]])

                # Policy output
                action = self.policy_weights.T @ state

                # Tracking error
                error = np.mean((action - q_ref[t]) ** 2)
                tracking_errors.append(error)

                # Gradient descent (simplified)
                grad = 2 * np.outer(state, action - q_ref[t])
                self.policy_weights -= learning_rate * grad

        avg_error = np.mean(tracking_errors)
        return {'tracking_error': avg_error}

    def compute_simplified_gradient(
        self,
        theta: np.ndarray,
        lambda_param: float = 0.1
    ) -> np.ndarray:
        """
        Compute simplified gradient estimate (Eq. 7 in paper).

        Args:
            theta: Current retargeting parameters
            lambda_param: Balance parameter

        Returns:
            Gradient w.r.t. theta
        """
        # Get adapted motion
        q_adapted = self.retarget_motion(theta)

        # Direct motion gradient (∇_θ L_motion)
        # Simplified: finite differences
        eps = 1e-5
        grad_motion = np.zeros_like(theta)

        for i in range(self.n_joints):
            for j in range(3):
                theta_plus = theta.copy()
                theta_plus[i, j] += eps
                q_plus = self.retarget_motion(theta_plus)
                loss_plus = self.compute_motion_loss(q_plus)

                theta_minus = theta.copy()
                theta_minus[i, j] -= eps
                q_minus = self.retarget_motion(theta_minus)
                loss_minus = self.compute_motion_loss(q_minus)

                grad_motion[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Policy gradient term (simplified)
        # In real ReActor: λ · E[∇_θ log π(a|s) · A(s,a)]
        # Here: approximate with tracking performance sensitivity
        policy_info = self.train_policy(q_adapted, n_steps=50)
        tracking_error = policy_info['tracking_error']

        # Approximate policy gradient contribution
        grad_policy = -tracking_error * np.sign(grad_motion)

        # Combined gradient
        grad_total = grad_motion + lambda_param * grad_policy

        return grad_total

    def bilevel_optimization(
        self,
        n_iterations: int = 20,
        lambda_param: float = 0.1,
        learning_rate: float = 0.01
    ) -> Dict:
        """
        Run bilevel optimization.

        Args:
            n_iterations: Number of upper-level updates
            lambda_param: Balance parameter
            learning_rate: Learning rate for theta

        Returns:
            Optimization history
        """
        history = {
            'motion_loss': [],
            'tracking_error': [],
            'theta_norm': []
        }

        print("Starting bilevel optimization...")
        print(f"Lambda: {lambda_param}, LR: {learning_rate}\n")

        for iteration in range(n_iterations):
            # Get adapted motion with current theta
            q_adapted = self.retarget_motion(self.theta)

            # Lower level: train policy
            policy_info = self.train_policy(q_adapted, n_steps=100)

            # Upper level: compute gradient and update theta
            grad = self.compute_simplified_gradient(self.theta, lambda_param)
            self.theta -= learning_rate * grad

            # Compute losses
            motion_loss = self.compute_motion_loss(q_adapted)
            tracking_error = policy_info['tracking_error']
            theta_norm = np.linalg.norm(self.theta)

            # Record history
            history['motion_loss'].append(motion_loss)
            history['tracking_error'].append(tracking_error)
            history['theta_norm'].append(theta_norm)

            if iteration % 5 == 0:
                print(f"Iter {iteration:2d}: "
                      f"Motion Loss = {motion_loss:.4f}, "
                      f"Tracking Error = {tracking_error:.4f}, "
                      f"||θ|| = {theta_norm:.4f}")

        print("\nOptimization complete!")
        return history


def main():
    """Run demo."""
    print("=" * 60)
    print("ReActor Bilevel Optimization Demo")
    print("=" * 60)
    print()

    # Create simplified ReActor
    reactor = SimplifiedReActor(n_joints=10, n_frames=100)

    # Run bilevel optimization
    history = reactor.bilevel_optimization(
        n_iterations=20,
        lambda_param=0.1,
        learning_rate=0.01
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Initial motion loss: {history['motion_loss'][0]:.4f}")
    print(f"Final motion loss: {history['motion_loss'][-1]:.4f}")
    print(f"Initial tracking error: {history['tracking_error'][0]:.4f}")
    print(f"Final tracking error: {history['tracking_error'][-1]:.4f}")
    print(f"Final ||θ||: {history['theta_norm'][-1]:.4f}")

    # Visualize (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history['motion_loss'])
        axes[0].set_title('Motion Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].grid(True)

        axes[1].plot(history['tracking_error'])
        axes[1].set_title('Tracking Error')
        axes[1].set_xlabel('Iteration')
        axes[1].grid(True)

        axes[2].plot(history['theta_norm'])
        axes[2].set_title('||θ|| (Parameter Norm)')
        axes[2].set_xlabel('Iteration')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('bilevel_optimization_results.png', dpi=150)
        print("\nPlot saved to: bilevel_optimization_results.png")

    except ImportError:
        print("\nMatplotlib not available - skipping visualization")


if __name__ == "__main__":
    main()
