"""Define how to simulate a Gaussian Plume."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import jit
from scipy import special

EPSILON = 1e-9  # used to ensure we don't divide by 0


@dataclass
class GaussianPlume:
    """Defines how to generate Gaussian Plumes.

    Parameters
    ----------
    spatial_resolution : float      # metres per pixel (e.g. 20)
    temporal_resolution : float     # seconds per time step (e.g. 0.1)
    duration : int                  # seconds to simulate (e.g. 600)
    crop_size : int                 # raster size, e.g. 512 (→ 512 x 512)
    dispersion_coeff : float        # puff dispersion rate (m/s)
    emission_rate : float           # emission rate (kg / h)
    OU_sigma_fluctuations : float   # Ornstein-Uhlenbeck parameter  (default 0.5 m/s)
    OU_correlation_time : float     # Ornstein-Uhlenbeck parameter  (default 60 s)
    """

    spatial_resolution: float
    temporal_resolution: float
    duration: int
    crop_size: int
    dispersion_coeff: float
    emission_rate: float
    OU_sigma_fluctuations: float = 0.5
    OU_correlation_time: float = 60.0

    # derived fields - calculated automatically after init
    source_x: int = field(init=False)
    source_y: int = field(init=False)

    def __post_init__(self) -> None:
        """Operations to run post initialization."""
        # centre pixel becomes the plume source coordinate
        self.source_x = self.crop_size // 2
        self.source_y = self.crop_size // 2

    def generate(
        self,
        wind_df: pd.DataFrame,
        OU_sigma_fluctuations: float | None = None,
        OU_correlation_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a single plume.

        Simulates a plume using Gaussian diffusion with Ornstein-Uhlenbeck processes
        for wind perturbations.

        Parameters
        ----------
            wind_df (dataframe with [timestamp, speed_x, speed_y] columns): DataFrame containing wind data
            OU_sigma_fluctuations (m/s): Optional override for wind speed fluctuations
            OU_correlation_time (s): Optional override for time scale of wind fluctuations

        Returns
        -------
            np.ndarray: The concentration field of the simulated plume
        """
        # Grid parameters
        dt = self.temporal_resolution

        # Compute puff centre positions over time
        times = (wind_df["timestamp"] - wind_df["timestamp"].min()).dt.total_seconds()
        times_interp, x_positions, y_positions = self.compute_positions(
            times=times.to_numpy(),
            u=wind_df.speed_x.to_numpy(),
            v=wind_df.speed_y.to_numpy(),
            dt_interp=dt,
        )

        # Now we add a disturbance to the positions to emulate turbulence
        sigma_fluctuations = self.OU_sigma_fluctuations if OU_sigma_fluctuations is None else OU_sigma_fluctuations
        correlation_time = self.OU_correlation_time if OU_correlation_time is None else OU_correlation_time
        x_disturbance, y_disturbance = self.simulate_OU_2D(
            n_steps=len(times_interp),
            dt=dt,
            sigma_fluctuations=sigma_fluctuations,  # m/s
            correlation_time=correlation_time,
        )

        # Add a Gaussian around each puff and sumup to get the concentration
        concentration, X, Y = self.create_concentration_field(
            x_pos=x_positions + x_disturbance[:-1],
            y_pos=y_positions + y_disturbance[:-1],
            times=times_interp,
            pixel_size=self.spatial_resolution,
            dispersion_coeff=self.dispersion_coeff,
            emission_rate=self.emission_rate,
            raster_size=self.crop_size,
        )

        return concentration, X, Y

    @staticmethod
    @jit(nopython=True)
    def add_outer(out: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Add the outer product of a and b to the input array out.

        This function computes the outer product of vectors a and b and adds the result
        to the existing values in out. It is optimized with Numba's JIT compilation.

        It is equivalent to out += np.outer(a, b), but much faster.

        Parameters
        ----------
        out : ndarray
            2D array with shape (len(a), len(b)) to which the outer product will be added
        a : ndarray
            1D array of length nx
        b : ndarray
            1D array of length ny

        Returns
        -------
        ndarray
            The modified input array with the outer product added
        """
        nx = len(a)
        ny = len(b)
        for i in range(nx):
            for j in range(ny):
                out[i, j] += a[i] * b[j]
        return out

    def create_concentration_field(
        self,
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        times: np.ndarray,
        pixel_size: float = 2.0,
        dispersion_coeff: float = 2.0,
        emission_rate: float = 1.0,
        raster_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a concentration field (raster grid) from a series of Gaussian puffs.

        Args:
            x_pos: Array of x positions for each puff (meters)
            y_pos: Array of y positions for each puff (meters)
            times: Array of times since release for each puff (seconds)
            pixel_size: Size of each grid cell (meters)
            dispersion_coeff: Coefficient controlling puff dispersion rate (m/s)
            emission_rate: Emission rate of the puff (kg/hr)
            raster_size: Size of the raster grid (nx = ny = raster_size)

        Returns
        -------
            tuple containing:
                concentration: 2D array of concentration values (mol/m2)
                X: 2D array of x coordinates for each grid cell
                Y: 2D array of y coordinates for each grid cell
        """
        # Set up the raster grid
        nx, ny = raster_size, raster_size
        dx = pixel_size  # shorthand
        x_grid = np.linspace(-nx * dx / 2, nx * dx / 2, nx)
        y_grid = np.linspace(-ny * dx / 2, ny * dx / 2, ny)
        X, Y = np.meshgrid(y_grid, x_grid)

        dt = times[1] - times[0]
        # 57.75286 is the conversion factor from mol/s to kg/hr for CH₄, given:
        #    1 mol CH₄/s = 16.04246 g/s ≈ 0.01604 kg/s = 57.75 kg/hr.
        puff_size = emission_rate * dt / 57.75286  # mol
        pixel_area = pixel_size**2  # m2

        # Initialize concentration field
        concentration = np.zeros((ny, nx))

        x_edges = np.concatenate([x_grid - dx / 2, [x_grid[-1] + dx / 2]])
        y_edges = np.concatenate([y_grid - dx / 2, [y_grid[-1] + dx / 2]])

        # Add contribution from each puff
        for i in range(len(x_pos)):
            # Calculate sigma based on time
            sigma = dispersion_coeff * times[i] + 0.01

            # X direction
            x_cdf = special.ndtr((x_edges - x_pos[i]) / sigma)
            x_gaussian = np.diff(x_cdf)

            # Y direction
            y_cdf = special.ndtr((y_edges - y_pos[i]) / sigma)
            y_gaussian = np.diff(y_cdf)

            # Multiply the 1D distributions to get 2D
            self.add_outer(concentration, y_gaussian, x_gaussian)

        concentration *= puff_size / pixel_area  # mol/m2
        return concentration, X, Y

    def compute_positions(
        self, times: np.ndarray, u: np.ndarray, v: np.ndarray, dt_interp: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute x,y positions over time with interpolated wind velocities.

        Args:
            times: 1D Array of timestamps (seconds since start)
            u: 1D Array of x-direction wind velocities (m/s)
            v: 1D Array of y-direction wind velocities (m/s)
            dt_interp: Time step for interpolation (seconds)

        Returns
        -------
            tuple containing:
                times_interp: Array of interpolated timestamps
                x: Array of x positions (meters)
                y: Array of y positions (meters)
        """
        # Create finer time array for interpolation
        assert len(times) == len(u) == len(v), "Times, u, and v must have the same length"
        t_start = times[0]
        t_end = times[-1]
        times_interp = np.arange(t_start, t_end + dt_interp, dt_interp)

        # Interpolate velocities to finer time resolution
        u_interp = np.interp(times_interp, times, u)
        v_interp = np.interp(times_interp, times, v)

        # Compute positions by integrating velocities over time
        x = np.cumsum(u_interp) * dt_interp
        y = np.cumsum(v_interp) * dt_interp

        return times_interp, x, y

    def simulate_OU_2D(
        self, n_steps: int, dt: float, sigma_fluctuations: float, correlation_time: float, p0: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a 2D Ornstein-Uhlenbeck process.

        Parameters
        ----------
            n_steps: Number of simulation steps
            dt: Time step size
            sigma_fluctuations: Size of the random fluctuations (m/s)
            correlation_time: Correlation time of the random fluctuations (s)
            p0: Initial position as a 2D array [x0, y0], defaults to [0, 0]

        Returns
        -------
            Tuple of (x_positions, y_positions) arrays
        """
        if p0 is None:
            p0 = np.zeros(2)

        p = np.zeros((n_steps + 1, 2))  # Array for (x, y) positions
        p[0] = p0  # Initial position

        for i in range(n_steps):  # Euler-Maruyama step
            p[i + 1] = p[i] + (-p[i] * dt / correlation_time) + (sigma_fluctuations * dt * np.random.randn(2))
        return p[:, 0], p[:, 1]  # Return x and y paths
