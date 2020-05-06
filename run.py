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

CAR_MASSES = [1.0, 1.0]  # kg
CAR_RADIUS = 0.9  # meters
MAX_ACCELERATION = 3.9  # m/s^2
# Coefficient of restitution, determining the elasticity of car collisions
RESTITUTION = 0.1

SECONDS_TO_MILLISECONDS = 1000


def add_collision_constraint(solver, prev_state, next_state, car_1, car_2):
    """Adds solver constraint that next_state is equal to prev_state after a
    collision between car_1 and car_2."""
    for i in range(NUM_CARS):
        if i != car_1 and i != car_2:
            solver.AddConstraint(pydrake.math.eq(prev_state[i], next_state[i]))

    solver.AddConstraint(prev_state[car_1, 0] == next_state[car_1, 0])
    solver.AddConstraint(prev_state[car_2, 0] == next_state[car_2, 0])

    mass_1, mass_2 = CAR_MASSES[car_1], CAR_MASSES[car_2]
    velocity_1, velocity_2 = prev_state[car_1, 1], prev_state[car_2, 1]
    solver.AddConstraint(
        next_state[car_1, 1]
        == (
            RESTITUTION * mass_2 * (velocity_2 - velocity_1)
            + mass_1 * velocity_1
            + mass_2 * velocity_2
        )
        / (mass_1 + mass_2)
    )
    solver.AddConstraint(
        next_state[car_2, 1]
        == (
            RESTITUTION * mass_1 * (velocity_1 - velocity_2)
            + mass_1 * velocity_1
            + mass_2 * velocity_2
        )
        / (mass_1 + mass_2)
    )


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


def solve(start_state, goal_state, num_time_samples, collision_sequence=[]):
    """Solves for vehicle trajectory from the start state to the goal state.

    Arguments:
      start_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired start state of the cars.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of Optional[float]
        Desired final state of the cars.
      num_time_samples: int
        Number of time samples in the solution. Increasing this number will make
        the solver take longer but will make the output trajectory approximately
        follow the dynamics and state/control constraints more closely.
      num_collisions: List[Tuple[int, int]]
        The collisions that will happen in this trajectory in order. A collision
        is specified by a pair of cars that will collide.
    """
    MIN_TIMESTEP = 0.01
    MAX_TIMESTEP = 0.5

    # Model the dynamics of the system.
    A = np.kron(np.eye(NUM_CARS), np.array([[0, 1], [0, 0]]))
    B = np.kron(np.eye(NUM_CARS), np.array([[0], [1]]))
    C = np.eye(NUM_STATE_DIMENSIONS * NUM_CARS)
    D = np.zeros((NUM_STATE_DIMENSIONS * NUM_CARS, NUM_CONTROL_DIMENSIONS * NUM_CARS))
    linear_system = pydrake.systems.primitives.LinearSystem(
        A=A, B=B, C=C, D=D, time_period=0.0
    )
    system_context = linear_system.CreateDefaultContext()
    collocation_constraint = pydrake.systems.trajectory_optimization.DirectCollocationConstraint(
        system=linear_system, context=system_context
    )

    solver = pydrake.solvers.mathematicalprogram.MathematicalProgram()

    # Solve for the trajectory as a sequence of collision-free sub-trajectories.
    # Each sub-trajectory is a cubic spline.
    num_trajectories = len(collision_sequence) + 1
    time_samples_per_trajectory = num_time_samples // num_trajectories

    # Since this is in one dimension, cars cannot pass each other. Pre-compute
    # the order of cars for collision checking later.
    car_order = list(range(NUM_CARS))
    car_order.sort(key=lambda c: start_state[c, 0])

    # The initial state trajectory guess will be a linear interpolation from the
    # start state to the goal state.
    TIMESTEP_GUESS = (MAX_TIMESTEP - MIN_TIMESTEP) / 2
    state_trajectory_guess = pydrake.trajectories.PiecewisePolynomial.FirstOrderHold(
        [0.0, (time_samples_per_trajectory * num_trajectories) * TIMESTEP_GUESS],
        np.column_stack((start_state.flatten(), goal_state.flatten())),
    )

    state_vars = []  # represents state of cars at each time sample
    control_vars = []  # represents control of cars at each time sample
    time_vars = []  # represents interval between each time sample
    for traj_idx in range(num_trajectories):
        state_vars.append(
            solver.NewContinuousVariables(
                (time_samples_per_trajectory + 1) * NUM_CARS * NUM_STATE_DIMENSIONS,
                name="state" + str(traj_idx),
            ).reshape((time_samples_per_trajectory + 1, NUM_CARS, NUM_STATE_DIMENSIONS))
        )
        control_vars.append(
            solver.NewContinuousVariables(
                (time_samples_per_trajectory + 1) * NUM_CARS * NUM_CONTROL_DIMENSIONS,
                name="control" + str(traj_idx),
            ).reshape(
                (time_samples_per_trajectory + 1, NUM_CARS, NUM_CONTROL_DIMENSIONS)
            )
        )
        time_vars.append(
            solver.NewContinuousVariables(
                time_samples_per_trajectory, name="time" + str(traj_idx)
            )
        )

        solver.AddBoundingBoxConstraint(MIN_TIMESTEP, MAX_TIMESTEP, time_vars[-1])
        solver.AddBoundingBoxConstraint(
            -MAX_ACCELERATION, MAX_ACCELERATION, control_vars[-1][:, :, 0]
        )

        if traj_idx == 0:
            solver.AddConstraint(pydrake.math.eq(state_vars[-1][0], start_state))
        else:
            # Constrain start of the sub-trajectory to be the result of a
            # collision at the end of the previous trajectory.
            car_1, car_2 = collision_sequence[traj_idx - 1]
            add_collision_constraint(
                solver,
                prev_state=state_vars[-2][-1],
                next_state=state_vars[-1][0],
                car_1=car_1,
                car_2=car_2,
            )

        if traj_idx == num_trajectories - 1:
            solver.AddConstraint(pydrake.math.eq(state_vars[-1][-1], goal_state))
        else:
            # Constrain sub-trajectory to end with a collision.
            car_1, car_2 = collision_sequence[traj_idx]
            position_1 = state_vars[-1][-1, car_1, 0]
            position_2 = state_vars[-1][-1, car_2, 0]
            # NOTE(tomtseng): For the 1-D case, I assume that car_1 hits car_2
            # from the left.
            solver.AddConstraint(position_2 == position_1 + 2 * CAR_RADIUS)

        for t in range(time_samples_per_trajectory):
            pydrake.systems.trajectory_optimization.AddDirectCollocationConstraint(
                constraint=collocation_constraint,
                timestep=[time_vars[-1][t]],
                state=state_vars[-1][t].flatten(),
                next_state=state_vars[-1][t + 1].flatten(),
                input=control_vars[-1][t].flatten(),
                next_input=control_vars[-1][t + 1].flatten(),
                prog=solver,
            )

        # Don't allow collisions within a sub-trajectory.
        for i in range(1, NUM_CARS):
            car_prev = car_order[i - 1]
            car_next = car_order[i]
            position_prev = state_vars[-1][:, car_prev, 0]
            position_next = state_vars[-1][:, car_next, 0]
            solver.AddConstraint(
                pydrake.math.ge(position_next, position_prev + 2 * CAR_RADIUS)
            )

        solver.SetInitialGuess(
            time_vars[-1], np.full_like(a=time_vars[-1], fill_value=TIMESTEP_GUESS)
        )
        solver.SetInitialGuess(
            state_vars[-1].reshape((time_samples_per_trajectory + 1, -1)),
            np.vstack(
                [
                    state_trajectory_guess.value(
                        ((time_samples_per_trajectory * traj_idx) + t) * TIMESTEP_GUESS
                    ).T
                    for t in range(time_samples_per_trajectory + 1)
                ]
            ),
        )

    # Penalize solutions that use a lot of time.
    solver.AddCost(sum(sum(ts) for ts in time_vars))

    solver_result = pydrake.solvers.mathematicalprogram.Solve(solver)
    if solver_result.is_success():
        state_trajectory = pydrake.trajectories.PiecewisePolynomial()

        cumulative_time = 0
        for traj_idx in range(num_trajectories):
            # Get the state trajectory as a cubic spline. This follows the
            # implementation of
            # pydrake.systems.trajectory_optimization.DirectCollocation::ReconstructStateTrajectory
            # from
            # https://github.com/RobotLocomotion/drake/blob/709ff3ed522f158c4dfbb5c6cfb50190e47af588/systems/trajectory_optimization/direct_collocation.cc#L224
            timesteps = solver_result.GetSolution(time_vars[traj_idx])
            states = solver_result.GetSolution(
                state_vars[traj_idx].reshape((time_samples_per_trajectory + 1, -1))
            )
            controls = solver_result.GetSolution(
                control_vars[traj_idx].reshape((time_samples_per_trajectory + 1, -1))
            )
            times = np.empty(time_samples_per_trajectory + 1)
            derivatives = np.empty_like(states)
            for t in range(time_samples_per_trajectory + 1):
                times[t] = (
                    cumulative_time if t == 0 else times[t - 1] + timesteps[t - 1]
                )
                system_context.SetContinuousState(states[t])
                system_context.FixInputPort(index=0, data=controls[t])
                derivatives[t] = linear_system.EvalTimeDerivatives(
                    system_context
                ).CopyToVector()
            subtrajectory = pydrake.trajectories.PiecewisePolynomial.CubicHermite(
                breaks=times, samples=states.T, samples_dot=derivatives.T
            )
            cumulative_time = times[-1]
            state_trajectory.ConcatenateInTime(subtrajectory)

        print("Cost: {}".format(solver_result.get_optimal_cost()))
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


START_STATE = np.array([[0, 0], [4, -5]])
# Don't enforce heading on goal state --- there are undesirable results if the
# cars turn more than 360 degrees in one direction.
GOAL_STATE = np.array([[3, 0], [6, 0]])
NUM_TIME_SAMPLES = 50
COLLISION_SEQUENCE = [(0, 1)]

solve(
    start_state=START_STATE,
    goal_state=GOAL_STATE,
    num_time_samples=NUM_TIME_SAMPLES,
    collision_sequence=COLLISION_SEQUENCE,
)
