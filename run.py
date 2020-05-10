import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pydrake.math
import pydrake.solvers.mathematicalprogram
import pydrake.systems.framework
import pydrake.systems.scalar_conversion
import pydrake.systems.trajectory_optimization
import pydrake.trajectories

NUM_CARS = 2
# Color for each car in the animation.
CAR_COLORS = ["r", "b"]
# control for a car is [acceleration, steering angle]
NUM_CONTROL_DIMENSIONS = 2
# state for a car is [x position, y position, heading, linear speed, steering
# angle]
NUM_STATE_DIMENSIONS = 5

CAR_RADIUS = 0.9  # meters
MAX_ACCELERATION = 3.9  # m/s^2
MAX_STEERING_ANGLE = np.pi / 4  # radians
MAX_STEERING_VELOCITY = np.pi / 2  # radians/s

EPSILON = 1e-2
SECONDS_TO_MILLISECONDS = 1000


@pydrake.systems.scalar_conversion.TemplateSystem.define("Cars")
def CarsSystem_(T):
    """Returns a Drake system representing a simplified dynamics of cars.

    Code modeled after
    https://github.com/RussTedrake/underactuated/blob/ae1832c9c3048a0a154d201e4ab6c4fe9555f666/underactuated/quadrotor2d.py
    """

    class Impl(pydrake.systems.framework.LeafSystem_[T]):
        def _construct(self, converter=None):
            pydrake.systems.framework.LeafSystem_[T].__init__(self, converter)

            self.DeclareVectorInputPort(
                "control",
                pydrake.systems.framework.BasicVector_[T](
                    NUM_CARS * NUM_CONTROL_DIMENSIONS
                ),
            )
            self.DeclareVectorOutputPort(
                "state",
                pydrake.systems.framework.BasicVector_[T](
                    NUM_CARS * NUM_STATE_DIMENSIONS
                ),
                self.CopyStateOut,
            )
            self.DeclareContinuousState(NUM_CARS * NUM_STATE_DIMENSIONS)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            """Compute state derivatives with respect to time using a bicycle
            model.
            """
            state = (
                context.get_continuous_state_vector()
                .CopyToVector()
                .reshape((NUM_CARS, NUM_STATE_DIMENSIONS))
            )
            control = (
                self.get_input_port(0)
                .Eval(context)
                .reshape((NUM_CARS, NUM_CONTROL_DIMENSIONS))
            )

            change_in_state = np.empty_like(state)
            for i in range(NUM_CARS):
                heading, speed, steering_angle = state[i, 2:5]
                change_in_state[i, 0] = speed * np.cos(heading)
                change_in_state[i, 1] = speed * np.sin(heading)
                change_in_state[i, 2] = speed * np.tan(steering_angle)
            change_in_state[:, 3:] = control
            derivatives.get_mutable_vector().SetFromVector(change_in_state.flatten())

    return Impl


def plot_trajectory(state_trajectory, goal_state):
    """Plots and animates a state trajectory.

    Arguments:
      state_trajectory: pydrake.trajectories.PiecewisePolynomial
        Trajectory to plot.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired final state.
    """
    FRAME_RATE = 25  # hertz
    TIMESTEP = 1 / FRAME_RATE  # seconds

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    times = np.arange(
        state_trajectory.start_time(), state_trajectory.end_time(), TIMESTEP
    )
    state_values = np.array([state_trajectory.value(t) for t in times]).reshape(
        len(times), NUM_CARS, NUM_STATE_DIMENSIONS
    )

    xs = state_values[:, :, 0]
    ys = state_values[:, :, 1]
    # Set axis limits.
    ax.set(
        xlim=(np.min(xs) - CAR_RADIUS, np.max(xs) + CAR_RADIUS),
        ylim=(np.min(ys) - CAR_RADIUS, np.max(ys) + CAR_RADIUS),
    )

    # Draw goal state.
    for i in range(NUM_CARS):
        x, y = goal_state[i, :2]
        goal = plt.Circle((x, y), radius=CAR_RADIUS, color=CAR_COLORS[i], alpha=0.2)
        ax.add_patch(goal)

    # Draw the cars.
    cars = [
        plt.Circle((0, 0), radius=CAR_RADIUS, color=CAR_COLORS[i])
        for i in range(NUM_CARS)
    ]
    for car in cars:
        ax.add_patch(car)
    car_headings = [matplotlib.lines.Line2D([], [], color="k") for i in range(NUM_CARS)]
    for car_heading in car_headings:
        ax.add_line(car_heading)
    animated_objects = cars + car_headings

    def animate(t):
        for i in range(NUM_CARS):
            x, y, heading = state_values[t, i, :3]
            cars[i].center = (x, y)
            car_headings[i].set_xdata([x, x + np.cos(heading) * CAR_RADIUS])
            car_headings[i].set_ydata([y, y + np.sin(heading) * CAR_RADIUS])
        return animated_objects

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
      collision_sequence: List[Tuple[int, int]]
        The collisions that will happen in this trajectory in order. A collision
        is specified by a pair of cars that will collide.
    """
    # Bounds on time step on each time sample
    MIN_TIMESTEP = 0.01  # seconds
    MAX_TIMESTEP = 0.5  # seconds

    cars_system = CarsSystem_[None]()
    system_context = cars_system.CreateDefaultContext()
    collocation_constraint = pydrake.systems.trajectory_optimization.DirectCollocationConstraint(
        system=cars_system, context=system_context
    )

    solver = pydrake.solvers.mathematicalprogram.MathematicalProgram()
    solver.SetSolverOption(pydrake.solvers.mathematicalprogram.SolverType.kSnopt, "Print file", "/tmp/snopt-output.txt")
    solver.SetSolverOption(pydrake.solvers.mathematicalprogram.SolverType.kSnopt, "Major iterations limit", 10000)
    solver.SetSolverOption(pydrake.solvers.mathematicalprogram.SolverType.kSnopt, "Iterations limit", 100000)

    # Solve for the trajectory as a sequence of collision-free sub-trajectories.
    num_trajectories = len(collision_sequence) + 1
    time_samples_per_trajectory = num_time_samples // num_trajectories

    # The initial state trajectory guess will be a linear interpolation from the
    # start state to the goal state.
    TIMESTEP_GUESS = (MAX_TIMESTEP - MIN_TIMESTEP) / 2
    state_trajectory_guesses = np.empty_like(goal_state)
    for c in range(NUM_CARS):
        for j in range(NUM_STATE_DIMENSIONS):
            if goal_state[c, j] is None:
                state_trajectory_guesses[c, j] = None
            else:
                state_trajectory_guesses[
                    c, j
                ] = pydrake.trajectories.PiecewisePolynomial.FirstOrderHold(
                    [
                        0.0,
                        (time_samples_per_trajectory * num_trajectories)
                        * TIMESTEP_GUESS,
                    ],
                    np.column_stack((start_state[c, j], goal_state[c, j])),
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
            -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, state_vars[-1][:, :, 4]
        )
        solver.AddBoundingBoxConstraint(
            -MAX_ACCELERATION, MAX_ACCELERATION, control_vars[-1][:, :, 0]
        )
        solver.AddBoundingBoxConstraint(
            -MAX_STEERING_VELOCITY, MAX_STEERING_VELOCITY, control_vars[-1][:, :, 1]
        )

        if traj_idx == 0:
            solver.AddConstraint(pydrake.math.eq(state_vars[-1][0], start_state))
        else:
            # Constrain start of the sub-trajectory to be the result of a
            # collision at the end of the previous trajectory.
            # TODO(tomtseng)
            pass

        if traj_idx == num_trajectories - 1:
            for c in range(NUM_CARS):
                for j in range(NUM_STATE_DIMENSIONS):
                    if goal_state[c, j] is not None:
                        solver.AddConstraint(
                            state_vars[-1][-1, c, j] <= goal_state[c, j] + EPSILON
                        )
                        solver.AddConstraint(
                            state_vars[-1][-1, c, j] >= goal_state[c, j] - EPSILON
                        )
        else:
            # Constrain sub-trajectory to end with a collision.
            # TODO(tomtseng)
            pass

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
            for j in range(i + 1, NUM_CARS):
                # TODO(tomtseng)
                pass

        solver.SetInitialGuess(
            time_vars[-1], np.full_like(a=time_vars[-1], fill_value=TIMESTEP_GUESS)
        )
        for c in range(NUM_CARS):
            for j in range(NUM_STATE_DIMENSIONS):
                if state_trajectory_guesses[c, j] is not None:
                    solver.SetInitialGuess(
                        state_vars[-1][:, c, j],
                        np.array(
                            [
                                state_trajectory_guesses[c, j].value(
                                    ((time_samples_per_trajectory * traj_idx) + t)
                                    * TIMESTEP_GUESS
                                )
                                for t in range(time_samples_per_trajectory + 1)
                            ]
                        ).flatten(),
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
                derivatives[t] = cars_system.EvalTimeDerivatives(
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
        print(solver_result.get_solution_result())
        print("Infeasible constraints:")
        for constraint in infeasible_constraints:
            print(constraint)


if __name__ == "__main__":
    START_STATE = np.array([[0, 0, 0, 0, 0], [4, -3, np.pi / 4, 0, 0]])
    # Don't enforce heading on goal state since headings modulo 2 * pi are
    # indistinguishable.
    GOAL_STATE = np.array([[6, 2, None, 0, 0], [3, 3, None, 0, 0]])
    NUM_TIME_SAMPLES = 20
    COLLISION_SEQUENCE = []

    solve(
        start_state=START_STATE,
        goal_state=GOAL_STATE,
        num_time_samples=NUM_TIME_SAMPLES,
        collision_sequence=COLLISION_SEQUENCE,
    )
