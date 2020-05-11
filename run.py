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
# state for a car is [x position of rear axle, y position of rear axle, heading,
# speed in direction of heading, speed perpendicular to the heading, steering
# angle of front axle relative to heading].
NUM_STATE_DIMENSIONS = 6
NUM_POSITION_DIMENSIONS = 2  # x, y

CAR_RADIUS = 0.9  # meters
CAR_MASSES = [1300, 1300]  # kg
# COG = center of gravity. The center of gravity of a car is assumed to be at
# the center of the circle representing the car.
COG_TO_FRONT_AXLE_LENGTH = 0.6  # meters
COG_TO_REAR_AXLE_LENGTH = 0.6  # meters
AXLE_TO_AXLE_LENGTH = COG_TO_FRONT_AXLE_LENGTH + COG_TO_REAR_AXLE_LENGTH  # meters

MAX_ACCELERATION = 3.9  # m/s^2
MAX_STEERING_ANGLE = np.pi / 4  # radians
MAX_STEERING_VELOCITY = np.pi / 2  # radians/s
# Determines how fast lateral speed decays.
CORNERING_COEFFICIENT = 5
# Coefficient of restitution; determines the elasticity of car collisions
RESTITUTION_COEFFICIENT = 0.1

EPSILON = 1e-2
SECONDS_TO_MILLISECONDS = 1000


@pydrake.systems.scalar_conversion.TemplateSystem.define("Cars")
def CarsSystem_(T):
    """Returns a Drake system representing a simplified dynamics of cars.

    Code modeled after
    https://github.com/RussTedrake/underactuated/blob/ae1832c9c3048a0a154d201e4ab6c4fe9555f666/underactuated/quadrotor2d.py

    The TemplateSystem template is to support automatic differentiation.
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
                heading, forward_speed, lateral_speed, steering_angle = state[i, 2:]
                forward_acceleration, steering_velocity = control[i]

                change_in_x = forward_speed * np.cos(heading) - lateral_speed * np.sin(
                    heading
                )
                change_in_y = forward_speed * np.sin(heading) + lateral_speed * np.cos(
                    heading
                )
                change_in_heading = (
                    forward_speed * np.tan(steering_angle) / AXLE_TO_AXLE_LENGTH
                )
                change_in_forward_speed = (
                    forward_acceleration + lateral_speed * change_in_heading
                )
                # The EPSILON term is to resolve numerical difficulties in using arctan2, which has a branch cut, in a solver
                change_in_lateral_speed = -CORNERING_COEFFICIENT * np.arctan2(
                    lateral_speed, EPSILON + np.abs(forward_speed)
                )
                change_in_steering_angle = steering_velocity
                change_in_state[i, :] = np.array(
                    [
                        change_in_x,
                        change_in_y,
                        change_in_heading,
                        change_in_forward_speed,
                        change_in_lateral_speed,
                        change_in_steering_angle,
                    ]
                )
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


def squared_distance_of_centers_of_gravity(current_state, car_1, car_2):
    """Computes the squared distance between the centers of two cars in the
    current state.
    """
    x_1, y_1, heading_1 = current_state[car_1, :3]
    x_cog_1, y_cog_1 = get_center_of_gravity(
        x_rear_axle=x_1, y_rear_axle=y_1, heading=heading_1
    )
    x_2, y_2, heading_2 = current_state[car_2, :3]
    x_cog_2, y_cog_2 = get_center_of_gravity(
        x_rear_axle=x_2, y_rear_axle=y_2, heading=heading_2
    )
    return (x_cog_1 - x_cog_2) ** 2 + (y_cog_1 - y_cog_2) ** 2


def add_collision_constraint(solver, prev_state, next_state, car_1, car_2):
    """Adds solver constraint that next_state is equal to prev_state after a
    collision between car_1 and car_2.
    """
    for i in range(NUM_CARS):
        if i != car_1 and i != car_2:
            solver.AddConstraint(pydrake.math.eq(prev_state[i], next_state[i]))

    UNCHANGED_COORDINATES = [0, 1, 2, 5]
    for car in [car_1, car_2]:
        solver.AddConstraint(
            pydrake.math.eq(
                prev_state[car, UNCHANGED_COORDINATES],
                next_state[car, UNCHANGED_COORDINATES],
            )
        )

    x_1, y_1, heading_1, forward_speed_1, lateral_speed_1 = prev_state[car_1, :5]
    x_cog_1, y_cog_1 = get_center_of_gravity(
        x_rear_axle=x_1, y_rear_axle=y_1, heading=heading_1
    )
    x_2, y_2, heading_2, forward_speed_2, lateral_speed_2 = prev_state[car_2, :5]
    x_cog_2, y_cog_2 = get_center_of_gravity(
        x_rear_axle=x_2, y_rear_axle=y_2, heading=heading_2
    )

    absolute_collision_angle = np.arctan2(y_cog_2 - y_cog_1, x_cog_2 - x_cog_1)

    collision_angle_1 = absolute_collision_angle - heading_1
    prev_speed_towards_collision_1 = forward_speed_1 * np.cos(
        collision_angle_1
    ) + lateral_speed_1 * np.sin(collision_angle_1)
    prev_speed_lateral_collision_1 = -forward_speed_1 * np.sin(
        collision_angle_1
    ) + lateral_speed_1 * np.cos(collision_angle_1)
    collision_angle_2 = np.pi + absolute_collision_angle - heading_2
    prev_speed_towards_collision_2 = forward_speed_2 * np.cos(
        collision_angle_2
    ) + lateral_speed_2 * np.sin(collision_angle_2)
    prev_speed_lateral_collision_2 = -forward_speed_2 * np.sin(
        collision_angle_2
    ) + lateral_speed_2 * np.cos(collision_angle_2)

    mass_1, mass_2 = CAR_MASSES[car_1], CAR_MASSES[car_2]
    relative_mass_1 = mass_1 / (mass_1 + mass_2)
    relative_mass_2 = mass_2 / (mass_1 + mass_2)
    next_speed_towards_collision_1 = (
        RESTITUTION_COEFFICIENT
        * relative_mass_2
        * (-prev_speed_towards_collision_2 - prev_speed_towards_collision_1)
        + relative_mass_1 * prev_speed_towards_collision_1
        + relative_mass_2 * -prev_speed_towards_collision_2
    )
    next_speed_lateral_collision_1 = prev_speed_lateral_collision_1
    next_speed_towards_collision_2 = (
        RESTITUTION_COEFFICIENT
        * relative_mass_1
        * (-prev_speed_towards_collision_1 - prev_speed_towards_collision_2)
        + relative_mass_1 * -prev_speed_towards_collision_1
        + relative_mass_2 * prev_speed_towards_collision_2
    )
    next_speed_lateral_collision_2 = prev_speed_lateral_collision_2

    next_forward_speed_1 = next_speed_towards_collision_1 * np.cos(
        collision_angle_1
    ) - next_speed_lateral_collision_1 * np.sin(collision_angle_1)
    next_lateral_speed_1 = next_speed_towards_collision_1 * np.sin(
        collision_angle_1
    ) + next_speed_lateral_collision_1 * np.cos(collision_angle_1)
    next_forward_speed_2 = next_speed_towards_collision_2 * np.cos(
        collision_angle_2
    ) - next_speed_lateral_collision_2 * np.sin(collision_angle_2)
    next_lateral_speed_2 = next_speed_towards_collision_2 * np.sin(
        collision_angle_2
    ) + next_speed_lateral_collision_2 * np.cos(collision_angle_2)

    solver.AddConstraint(next_state[car_1, 3] == next_forward_speed_1)
    solver.AddConstraint(next_state[car_1, 4] == next_lateral_speed_1)
    solver.AddConstraint(next_state[car_2, 3] == next_forward_speed_2)
    solver.AddConstraint(next_state[car_2, 4] == next_lateral_speed_2)


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
    time_text = ax.text(0.01, 0.01, "", fontsize=12, transform=ax.transAxes)
    animated_objects = cars + rear_axles + front_axles + [time_text]

    def animate(t):
        time_text.set_text("time = {:.2f}s".format(t * TIMESTEP))
        for i in range(NUM_CARS):
            x_rear_axle, y_rear_axle, heading, _, _, steering_angle = state_values[
                t, i, :
            ]
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
    MIN_TIMESTEP = 0.005  # seconds
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
    # Normalizes constraints and variables. This seems to give better solutions.
    solver.SetSolverOption(
        pydrake.solvers.mathematicalprogram.SolverType.kSnopt, "Scale option", 2
    )

    # Solve for the trajectory as a sequence of collision-free sub-trajectories.
    num_trajectories = len(collision_sequence) + 1
    time_samples_per_trajectory = num_time_samples // num_trajectories

    # The initial state trajectory guess will be a linear interpolation from the
    # start state to the goal state.
    TIMESTEP_GUESS = (MAX_TIMESTEP - MIN_TIMESTEP) / 2
    position_trajectory_guess = pydrake.trajectories.PiecewisePolynomial.FirstOrderHold(
        [0.0, (time_samples_per_trajectory * num_trajectories) * TIMESTEP_GUESS],
        np.column_stack((start_state[:, :2].flatten(), goal_position.flatten())),
    )

    state_vars = []  # represents state of cars at each time sample
    control_vars = []  # represents control of cars at each time sample
    time_vars = []  # represents interval between each time sample
    for traj_idx in range(num_trajectories):
        state_vars.append(
            solver.NewContinuousVariables(
                (time_samples_per_trajectory + 1) * NUM_CARS * NUM_STATE_DIMENSIONS,
                name="traj {}: state".format(traj_idx),
            ).reshape((time_samples_per_trajectory + 1, NUM_CARS, NUM_STATE_DIMENSIONS))
        )
        control_vars.append(
            solver.NewContinuousVariables(
                (time_samples_per_trajectory + 1) * NUM_CARS * NUM_CONTROL_DIMENSIONS,
                name="traj {}: control".format(traj_idx),
            ).reshape(
                (time_samples_per_trajectory + 1, NUM_CARS, NUM_CONTROL_DIMENSIONS)
            )
        )
        time_vars.append(
            solver.NewContinuousVariables(
                time_samples_per_trajectory, name="traj {}: time".format(traj_idx)
            )
        )

        solver.AddBoundingBoxConstraint(
            MIN_TIMESTEP, MAX_TIMESTEP, time_vars[-1]
        ).evaluator().set_description("traj {}: timestep bounds".format(traj_idx))
        solver.AddBoundingBoxConstraint(
            -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, state_vars[-1][:, :, 5]
        ).evaluator().set_description("traj {}: steering angle bounds".format(traj_idx))
        solver.AddBoundingBoxConstraint(
            -MAX_ACCELERATION, MAX_ACCELERATION, control_vars[-1][:, :, 0]
        ).evaluator().set_description("traj {}: acceleration bounds".format(traj_idx))
        solver.AddBoundingBoxConstraint(
            -MAX_STEERING_VELOCITY, MAX_STEERING_VELOCITY, control_vars[-1][:, :, 1]
        ).evaluator().set_description(
            "traj {}: steering velocity bounds".format(traj_idx)
        )

        if traj_idx == 0:
            solver.AddConstraint(
                pydrake.math.eq(state_vars[-1][0], start_state)
            ).evaluator().set_description("traj {}: start".format(traj_idx))
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
            # Constrain last sub-trajectory to end at goal state
            for c in range(NUM_CARS):
                x, y, heading, forward_speed, lateral_speed = state_vars[-1][-1, c, :5]
                x_cog, y_cog = get_center_of_gravity(
                    x_rear_axle=x, y_rear_axle=y, heading=heading
                )
                for coord, goal_coord in [
                    (x_cog, goal_position[c, 0]),
                    (y_cog, goal_position[c, 1]),
                    (forward_speed, 0),
                    (lateral_speed, 0),
                ]:
                    solver.AddConstraint(
                        coord >= goal_coord - EPSILON
                    ).evaluator().set_description(
                        "traj {}: finish at goal - lower bound".format(traj_idx)
                    )
                    solver.AddConstraint(
                        coord <= goal_coord + EPSILON
                    ).evaluator().set_description(
                        "traj {}: finish at goal - upper bound".format(traj_idx)
                    )

        else:
            # Constrain sub-trajectory to end with a collision.
            car_1, car_2 = collision_sequence[traj_idx]
            squared_distance = squared_distance_of_centers_of_gravity(
                current_state=state_vars[-1][-1], car_1=car_1, car_2=car_2
            )
            solver.AddConstraint(
                squared_distance == (2 * CAR_RADIUS) ** 2
            ).evaluator().set_description(
                "traj {}: enforce collision {}<->{} at finish".format(
                    traj_idx, car_1, car_2
                )
            )

        for t in range(time_samples_per_trajectory):
            pydrake.systems.trajectory_optimization.AddDirectCollocationConstraint(
                constraint=collocation_constraint,
                timestep=[time_vars[-1][t]],
                state=state_vars[-1][t].flatten(),
                next_state=state_vars[-1][t + 1].flatten(),
                input=control_vars[-1][t].flatten(),
                next_input=control_vars[-1][t + 1].flatten(),
                prog=solver,
            ).evaluator().set_description(
                "traj {}: time sample {}: direct collocation".format(traj_idx, t)
            )

        # Don't allow collisions within a sub-trajectory.
        for t in range(time_samples_per_trajectory):
            for i in range(NUM_CARS):
                for j in range(i + 1, NUM_CARS):
                    squared_distance = squared_distance_of_centers_of_gravity(
                        current_state=state_vars[-1][t], car_1=i, car_2=j
                    )
                    solver.AddConstraint(
                        squared_distance >= (2 * CAR_RADIUS) ** 2
                    ).evaluator().set_description(
                        "traj {}: time sample {}: avoid collision {}<->{}".format(
                            traj_idx, t, i, j
                        )
                    )

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
    START_STATE = np.array([[0, 0, 0, 0, 0, 0], [4, -3, np.pi / 2, 0, 0, 0]])
    GOAL_CENTER_OF_GRAVITY_POSITION = np.array([[6, 2], [3, 3]])
    NUM_TIME_SAMPLES = 40
    COLLISION_SEQUENCE = [(0, 1)]

    solve(
        start_state=START_STATE,
        goal_position=GOAL_CENTER_OF_GRAVITY_POSITION,
        num_time_samples=NUM_TIME_SAMPLES,
        collision_sequence=COLLISION_SEQUENCE,
    )
