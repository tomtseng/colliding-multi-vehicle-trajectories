import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pydrake.solvers.mathematicalprogram
import pydrake.systems.primitives
import pydrake.systems.trajectory_optimization
import pydrake.trajectories


NUM_CARS = 2
# Color for each car in the animation.
CAR_COLORS = ["r", "b"]
# control for a car is [acceleration]
NUM_CONTROL_DIMENSIONS = 1
# state for a car is [position, speed]
NUM_STATE_DIMENSIONS = 2

CAR_RADIUS = 0.9  # meters
MAX_ACCELERATION = 3.9  # m/s^2
# Coefficient of restitution, determining the elasticity of car collisions
RESTITUTION = 0.1

SECONDS_TO_MILLISECONDS = 1000


def plot_trajectory(state_trajectory, goal_state):
    """Plots and animates a state trajectory.

    Arguments:
      state_trajectory: (num_time_steps x NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array
        Trajectory to plot.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired final state.
    """
    FRAME_RATE = 25  # s^-1
    TIMESTEP = 1 / FRAME_RATE  # s

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    times = np.arange(
        state_trajectory.start_time(), state_trajectory.end_time(), TIMESTEP
    )
    state_values = np.array([state_trajectory.value(t) for t in times]).reshape(
        len(times), NUM_CARS, NUM_STATE_DIMENSIONS
    )
    positions = state_values[:, :, 0]
    ax.set(
        xlim=(np.min(positions) - CAR_RADIUS, np.max(positions) + CAR_RADIUS),
        ylim=(-2 * CAR_RADIUS, 2 * CAR_RADIUS),
    )

    # Draw goal state.
    for i in range(NUM_CARS):
        goal = plt.Circle(
            (goal_state[i, 0], 0), radius=CAR_RADIUS, color=CAR_COLORS[i], alpha=0.2
        )
        ax.add_patch(goal)

    # Draw the cars.
    cars = [
        plt.Circle((0, 0), radius=CAR_RADIUS, color=CAR_COLORS[i])
        for i in range(NUM_CARS)
    ]
    for car in cars:
        ax.add_patch(car)

    def animate(t):
        for i in range(NUM_CARS):
            cars[i].center = (positions[t, i], 0)
        return cars

    SECONDS_TO_MILLISECONDS = 1000
    matplotlib.animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: [],
        frames=len(times),
        interval=TIMESTEP * SECONDS_TO_MILLISECONDS,
        blit=True,
    )

    plt.show()


def solve(start_state, goal_state, num_time_samples):
    """Solves for vehicle trajectory from the start state to the goal state.

    Arguments:
      start_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired start state of the cars.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of Optional[float]
        Desired final state of the cars.
    """

    A = np.kron(np.eye(NUM_CARS), np.array([[0, 1], [0, 0]]))
    B = np.kron(np.eye(NUM_CARS), np.array([[0], [1]]))
    C = np.eye(NUM_STATE_DIMENSIONS * NUM_CARS)
    D = np.zeros((NUM_STATE_DIMENSIONS * NUM_CARS, NUM_CONTROL_DIMENSIONS * NUM_CARS))
    linear_system = pydrake.systems.primitives.LinearSystem(
        A=A, B=B, C=C, D=D, time_period=0.0
    )

    MIN_TIMESTEP = 0.01
    MAX_TIMESTEP = 0.1
    solver = pydrake.systems.trajectory_optimization.DirectCollocation(
        linear_system,
        linear_system.CreateDefaultContext(),
        num_time_samples=num_time_samples,
        minimum_timestep=MIN_TIMESTEP,
        maximum_timestep=MAX_TIMESTEP,
    )
    flat_start_state = start_state.flatten()
    flat_goal_state = goal_state.flatten()
    solver.AddBoundingBoxConstraint(
        flat_start_state, flat_start_state, solver.initial_state()
    )
    solver.AddBoundingBoxConstraint(
        flat_goal_state, flat_goal_state, solver.final_state()
    )
    control_vars = solver.input()
    for c in range(NUM_CARS):
        solver.AddConstraintToAllKnotPoints(control_vars[c] <= MAX_ACCELERATION)
        solver.AddConstraintToAllKnotPoints(control_vars[c] >= -MAX_ACCELERATION)
    # Penalize solutions that use large accelerations.
    solver.AddRunningCost(control_vars.dot(control_vars))

    solver_result = pydrake.solvers.mathematicalprogram.Solve(solver)
    if solver_result.is_success():
        state_trajectory = solver.ReconstructStateTrajectory(solver_result)
        plot_trajectory(
            state_trajectory=state_trajectory, goal_state=goal_state,
        )
    else:
        infeasible_constraints = pydrake.solvers.mathematicalprogram.GetInfeasibleConstraints(
            solver, solver_result
        )
        print("Failed to solve.")
        print("Infeasible constraints:")
        for constraint in infeasible_constraints:
            print(constraint)


START_STATE = np.array([[0, 0], [4, -3]])
# Don't enforce heading on goal state --- there are undesirable results if the
# cars turn more than 360 degrees in one direction.
GOAL_STATE = np.array([[3, 0], [6, 0]])
NUM_TIME_SAMPLES = 30

solve(
    start_state=START_STATE, goal_state=GOAL_STATE, num_time_samples=NUM_TIME_SAMPLES,
)
