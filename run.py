"""
Searches for an approximate trajectory for several circular cars to get from a
start location to an end location.
"""
#!/usr/bin/env python3

import time

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
# angle].
# The state uses the rear axle as its reference frame; the heading is the
# direction of the rear wheel, the linear speed is the speed of the rear wheel
# in the heading direction, and the steering angle is the front axle angle
# relative to the heading.
NUM_STATE_DIMENSIONS = 5
NUM_POSITION_DIMENSIONS = 2  # x, y

CAR_RADIUS = 0.9  # meters
# COG = center of gravity. The center of gravity of a car is assumed to be at
# the center of the circle representing the car.
COG_TO_REAR_AXLE_LENGTH = 0.6  # meters

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


def get_center_of_gravity(x_rear_axle, y_rear_axle, heading):
    """Computes the position of the center of gravity of a car given the position and heading
    of the rear axle.
    """
    # If the state and control are symbolic variables used for optimization,
    # then we need to call pydrake.symbolic functions rather than numpy
    # functions to represent mathematical operations.
    m = pydrake.symbolic if type(x_rear_axle) == pydrake.symbolic.Variable else np

    return (
        x_rear_axle + COG_TO_REAR_AXLE_LENGTH * m.cos(heading),
        y_rear_axle + COG_TO_REAR_AXLE_LENGTH * m.sin(heading),
    )


def plot_trajectory(state_trajectory, goal_position):
    """Plots and animates a state trajectory.

    Arguments:
      state_trajectory: pydrake.trajectories.PiecewisePolynomial
        Trajectory to plot.
      goal_position: (NUM_CARS x NUM_POSITION_DIMENSIONS)-dimensional array of floats
        Desired final position.
    """
    FRAME_RATE = 25  # 1/s
    TIMESTEP = 1 / FRAME_RATE  # s

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
        xlim=(np.min(xs) - 2 * CAR_RADIUS, np.max(xs) + 2 * CAR_RADIUS),
        ylim=(np.min(ys) - 2 * CAR_RADIUS, np.max(ys) + 2 * CAR_RADIUS),
    )

    # Draw goal state.
    for i in range(NUM_CARS):
        x, y = goal_position[i]
        goal = plt.Circle((x, y), radius=CAR_RADIUS, color=CAR_COLORS[i], alpha=0.2)
        ax.add_patch(goal)

    # Draw the cars.
    cars = [
        plt.Circle((0, 0), radius=CAR_RADIUS, color=CAR_COLORS[i])
        for i in range(NUM_CARS)
    ]
    for car in cars:
        ax.add_patch(car)
    front_axles = [matplotlib.lines.Line2D([], [], color="k") for i in range(NUM_CARS)]
    for front_axle in front_axles:
        ax.add_line(front_axle)
    rear_axles = [matplotlib.lines.Line2D([], [], color="k") for i in range(NUM_CARS)]
    for rear_axle in rear_axles:
        ax.add_line(rear_axle)
    animated_objects = cars + rear_axles + front_axles

    def animate(t):
        for i in range(NUM_CARS):
            x_rear_axle, y_rear_axle, heading, _, steering_angle = state_values[t, i, :]
            x_cog, y_cog = get_center_of_gravity(
                x_rear_axle=x_rear_axle, y_rear_axle=y_rear_axle, heading=heading
            )
            cars[i].center = (x_cog, y_cog)
            front_axles[i].set_xdata(
                [x_cog, x_cog + np.cos(heading + steering_angle) * CAR_RADIUS]
            )
            front_axles[i].set_ydata(
                [y_cog, y_cog + np.sin(heading + steering_angle) * CAR_RADIUS]
            )
            rear_axles[i].set_xdata([x_cog, x_cog - np.cos(heading) * CAR_RADIUS])
            rear_axles[i].set_ydata([y_cog, y_cog - np.sin(heading) * CAR_RADIUS])
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


def solve(start_state, goal_position, num_time_samples, collision_sequence=[]):
    """Solves for vehicle trajectory from the start state to the goal state.

    Arguments:
      start_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired start state of the cars.
      goal_position: (NUM_CARS x NUM_POSITION_DIMENSIONS)-dimensional array of floats
        Desired final position of the cars. The position is relative to the
        center of gravity of the each car, not the rear axle of reach car.
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
    MAX_TIMESTEP = 0.2  # seconds
    SOLVER_OUTPUT_FILE = "/tmp/snopt-output.txt"
    SNOPT_ITERATIONS_LIMIT = 100000  # default is 10000
    SNOPT_MAJOR_ITERATIONS_LIMIT = 10000  # default is 1000

    cars_system = CarsSystem_[None]()
    system_context = cars_system.CreateDefaultContext()
    collocation_constraint = pydrake.systems.trajectory_optimization.DirectCollocationConstraint(
        system=cars_system, context=system_context
    )

    solver = pydrake.solvers.mathematicalprogram.MathematicalProgram()
    # See section 7.7 of "User’s Guide for SNOPT Version 7.5" for SNOPT option
    # descriptions.
    solver.SetSolverOption(
        pydrake.solvers.mathematicalprogram.SolverType.kSnopt,
        "Print file",
        SOLVER_OUTPUT_FILE,
    )
    solver.SetSolverOption(
        pydrake.solvers.mathematicalprogram.SolverType.kSnopt,
        "Iterations limit",
        SNOPT_ITERATIONS_LIMIT,
    )
    solver.SetSolverOption(
        pydrake.solvers.mathematicalprogram.SolverType.kSnopt,
        "Major iterations limit",
        SNOPT_MAJOR_ITERATIONS_LIMIT,
    )

    # Solve for the trajectory as a sequence of collision-free sub-trajectories.
    num_trajectories = len(collision_sequence) + 1
    time_samples_per_trajectory = num_time_samples // num_trajectories

    # The initial state trajectory guess will be a linear interpolation from the
    # start state to the goal state.
    TIMESTEP_GUESS = (MAX_TIMESTEP - MIN_TIMESTEP) / 2
    position_trajectory_guess = pydrake.trajectories.PiecewisePolynomial.FirstOrderHold(
        [0.0, (time_samples_per_trajectory * num_trajectories) * TIMESTEP_GUESS,],
        np.column_stack((start_state[:, :2].flatten(), goal_position.flatten())),
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
                x, y, heading, speed = state_vars[-1][-1, c, :4]
                x_cog, y_cog = get_center_of_gravity(
                    x_rear_axle=x, y_rear_axle=y, heading=heading
                )
                for coord, goal_coord in [
                    (x_cog, goal_position[c, 0]),
                    (y_cog, goal_position[c, 1]),
                    (speed, 0),
                ]:
                    solver.AddConstraint(coord >= goal_coord - EPSILON)
                    solver.AddConstraint(coord <= goal_coord + EPSILON)

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
        for t in range(time_samples_per_trajectory):
            for i in range(NUM_CARS):
                x_1, y_1, heading_1 = state_vars[-1][t, i, :3]
                x_cog_1, y_cog_1 = get_center_of_gravity(
                    x_rear_axle=x_1, y_rear_axle=y_1, heading=heading_1
                )
                for j in range(i + 1, NUM_CARS):
                    x_2, y_2, heading_2 = state_vars[-1][t, j, :3]
                    x_cog_2, y_cog_2 = get_center_of_gravity(
                        x_rear_axle=x_2, y_rear_axle=y_2, heading=heading_2
                    )
                    distance_squared = (x_cog_1 - x_cog_2) ** 2 + (
                        y_cog_1 - y_cog_2
                    ) ** 2
                    solver.AddConstraint(distance_squared >= (2 * CAR_RADIUS) ** 2)

        solver.SetInitialGuess(
            time_vars[-1], np.full_like(a=time_vars[-1], fill_value=TIMESTEP_GUESS)
        )
        solver.SetInitialGuess(
            state_vars[-1][:, :, :NUM_POSITION_DIMENSIONS].reshape(
                (time_samples_per_trajectory + 1, NUM_CARS * NUM_POSITION_DIMENSIONS)
            ),
            np.vstack(
                [
                    position_trajectory_guess.value(
                        ((time_samples_per_trajectory * traj_idx) + t) * TIMESTEP_GUESS
                    ).T
                    for t in range(time_samples_per_trajectory + 1)
                ]
            ),
        )

    # Penalize solutions that use a lot of time.
    solver.AddCost(sum(sum(ts) for ts in time_vars))

    print("Solving...")
    start_time = time.time()
    solver_result = pydrake.solvers.mathematicalprogram.Solve(solver)
    end_time = time.time()
    print("Done solving: {} seconds".format(end_time - start_time))
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
            state_trajectory=state_trajectory, goal_position=goal_position,
        )
    else:
        infeasible_constraints = pydrake.solvers.mathematicalprogram.GetInfeasibleConstraints(
            solver, solver_result
        )
        print("Failed to solve: {}".format(solver_result.get_solution_result()))
        print("Infeasible constraints:")
        for constraint in infeasible_constraints:
            print(constraint)


if __name__ == "__main__":
    START_STATE = np.array([[0, 0, 0, 0, 0], [4, -3, np.pi / 4, 0, 0]])
    GOAL_CENTER_OF_GRAVITY_POSITION = np.array([[6, 2], [3, 3]])
    NUM_TIME_SAMPLES = 30
    COLLISION_SEQUENCE = []

    solve(
        start_state=START_STATE,
        goal_position=GOAL_CENTER_OF_GRAVITY_POSITION,
        num_time_samples=NUM_TIME_SAMPLES,
        collision_sequence=COLLISION_SEQUENCE,
    )
