
import sys
import torch
torch.random.manual_seed(0)
import numpy as np


def random_SSM(N):
    A = torch.randn(N, N, dtype=torch.float32) 
    B = torch.randn(N, 1, dtype=torch.float32)
    C = torch.randn(1, N, dtype=torch.float32)
    return A, B, C

def discretize(A, B, C, step):
    I = torch.eye(A.shape[0])
    BL = torch.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C

def scan_SSM(Ab, Bb, Cb, u, x0):
    """
    Simulate the state-space model using a recurrence relation, i.e. compute the state and output at each time step.
    In training, we would use the convulutional form of the state-space model, instead of this recurrence form, as we 
    can compute the state and output at each time step in parallel (here during training, we have access to the 
    entire sequence at once, so we can compute the state and output at each time step in parallel). But in
    inference, we would use this recurrence form to compute the state and output at each time step.

    Parameters:
    - Ab (torch.Tensor): The state transition matrix.
    - Bb (torch.Tensor): The input matrix.
    - Cb (torch.Tensor): The output matrix.
    - u (torch.Tensor): The input sequence (time steps, input_dim).
    - x0 (torch.Tensor): The initial state.

    Returns:
    - x_out (torch.Tensor): Sequence of states.
    - y_out (torch.Tensor): Sequence of outputs.
    """
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    x_out = [x0]
    y_out = []

    x_k = x0
    for u_k in u:
        x_k, y_k = step(x_k, u_k)
        x_out.append(x_k)
        y_out.append(y_k)

    return torch.stack(x_out[1:]), torch.stack(y_out)

def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [ (Cb @ torch.matrix_power(Ab, l) @ Bb) for l in range(L)]
    )


def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, None], torch.zeros(N,))[1]

def causal_convolution(u, K, nofft=False):
    u = u.flatten()  # Ensure `u` is 1D
    K = K.flatten()  # Ensure `K` is 1D

    if nofft:
        return np.convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0]))) # compiles to frequency domain
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0]))) # compiles to frequency domain
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(N)
    u = torch.randn((L,), dtype=torch.float32)
    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    conv = causal_convolution(u, K_conv(*ssmb, L), nofft=True)

    # Check
    assert np.allclose(rec.ravel(), conv.ravel())

#### example_force, example_mass, example_ssm
"""
Example of a mass-spring-damper system. Here we approx the position of the mass, given its force only, i.e.
we are solvin a ode with a state-space model. The force is a sine wave with a frequency of 10 Hz. 
We solve the given ode using the run_SSM function and animate the results.
"""

def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)

def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 0]])
    return A, B, C

def example_ssm():
    # SSM
    # k = spring constant, b = friction, m = mass
    ssm = example_mass(k=100, b=1, m=0.05)

    # L samples of u(t).
    # we currently have 300 samples, i.e. 300 time steps/sequence length, we can increase this to get a smoother animation but with
    # more computational cost. Here we are using the recurrence computation to solve the ode.
    L = 300

    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t).
    y = run_SSM(*ssm, u)

    # Plotting ---
    import matplotlib.pyplot as plt
    import seaborn
    from celluloid import Camera

    seaborn.set_context("paper")
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])

    # Animate plot over time
    for k in range(0, L, 2):
        ax1.plot(ks[:k], u[:k], color="red")
        ax2.plot(ks[:k], y[:k], color="blue")
        ax3.boxplot(
            [[y[k, 0] - 0.04, y[k, 0], y[k, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("images/line.gif", dpi=150, writer="imagemagick")






if __name__ == "__main__":
    example_ssm()