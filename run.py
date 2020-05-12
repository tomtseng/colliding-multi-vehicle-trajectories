"""
Searches for an approximate trajectory for several circular cars to get from a
start location to an end location, then displays an animation illustrating the
trajectory.

To change the trajectory that this program searches for, change the constants in
`run_main()`.

The trajectory optimization in this program leaves a lot to be desired. It often
returns local optima that clearly could be improved, and it often hits numerical
issues and fails to solve.

The trajectory optimization can handle more cars if the `NUM_CARS` constant and
the constants in `run_main()` are adjusted appropriately, but such trajectories
take a long time to solve for, and the solver seems to encounter numerical
issues annoyingly often.

Here's an example of a situation with three cars for which the solver can find a
trajectory:
    START_STATE = np.array(
        [
            [-1, 0, 0, 2, 0, 0],
            [2.5, -2, np.pi / 2, 0, 0, 0],
            [4, 4, 3 * np.pi / 2, 0, 0, 0],
        ]
    )
    GOAL_CENTER_OF_GRAVITY_POSITION = np.array([[0, 0], [7, 3], [2, 1]])
    COLLISION_SEQUENCE = [(0, 1), (1, 2)]
    NUM_TIME_SAMPLES = 40
"""
#!/usr/bin/env python3

import time
import sys

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
CAR_COLORS = ["r", "b", "g"]
# control for a car is [acceleration, steering angle]
NUM_CONTROL_DIMENSIONS = 2
# State for a car is [x position of rear axle, y position of rear axle, heading,
# speed in direction of heading, speed perpendicular to the heading, steering
# angle of front axle relative to heading].
NUM_STATE_DIMENSIONS = 6
NUM_POSITION_DIMENSIONS = 2  # x, y

CAR_RADIUS = 0.9  # meters
CAR_MASSES = [1300, 1300, 1300]  # kg
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


def solve(start_state, goal_position, num_time_samples, collision_sequence=[]):
    """Solves for an approximate vehicle trajectory from the start state to the
    goal state.

    The solver tries to minimize the time for all cars to get to their goal
    position. (It can be mildly entertaining to remove the `solver.AddCost(...)`
    statement and observe the meandering trajectories produced. Removing that
    statement is also useful for more quickly checking if there's any hope of
    the solver finding a feasible trajectory.)

    Arguments:
      start_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired start state of the cars.
      goal_position: (NUM_CARS x NUM_POSITION_DIMENSIONS)-dimensional array of floats
        Desired final position of the cars. The position is relative to the
        center of gravity of each car, not the rear axle of each car like in the
        car state.
      num_time_samples: int
        Number of time samples in the solution. Increasing this number will make
        the solver take longer but will make the output trajectory follow the
        vehicle dynamics and constraints more closely. Also, there is a minimum
        interval between time samples, so setting this argument to a huge number
        will result in slow trajectories.
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
        LATERAL_SPEED_GUESS = 0
        solver.SetInitialGuess(
            state_vars[-1][:, :, 4],
            np.full_like(a=state_vars[-1][:, :, 4], fill_value=LATERAL_SPEED_GUESS),
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
        print("Success! Cost: {}".format(solver_result.get_optimal_cost()))
    else:
        print("Failed to solve: {}".format(solver_result.get_solution_result()))

STARTS=np.array([
[[0.48813503927324753, 2.151893663724195, 1.2913626692990459, 0.4488318299689684, 0.0, -0.11992276076686781], [1.4589411306665614, -0.6241278873730751, 4.92316472452724, 4.636627605010293, 0.0, -0.18308963408526047]],
[[-0.8297799529742598, 2.2032449344215808, -6.281748030835881, -1.9766742736816023, 0.0, -0.5548745491664142], [-4.0766140523120225, -3.137397886223291, -1.94074114138924, -1.0323252576933006, 0.0, 0.060973183190647595]],
[[-0.6400509785799624, -4.740737681721087, 0.624077102651273, -0.6467760738172315, 0.0, -0.12508596397557648], [-1.6966517899612588, -2.953513659621575, 1.4988031666952022, -2.0034532632547686, 0.0, -0.366266859777174]],
[[0.5079790257457546, 2.081478226181048, -2.6275685445261354, 0.10827605197663015, 0.0, 0.617239632514538], [3.9629308893343804, -3.7441468953616375, -3.678894493308657, -4.485327966991701, 0.0, -0.09297568017599589]],
[[4.670298390136766, 0.4723224917572235, 5.939926850929895, 2.148159936743647, 0.0, 0.31059171137886543], [-2.839105044196236, 4.762744547762418, -6.2048936112566695, -2.4701763761655604, 0.0, -0.10242922137502064]],
[[-2.780068289102605, 3.7073230617737636, -3.685475788097069, 4.186109079379216, 0.0, -0.018203662073001947], [1.1174386290264575, 2.6590785648031554, 0.23144726158217743, -2.0319949842377802, 0.0, -0.49052634695492936]],
[[-4.236917106260428, 2.7991879224011464, -0.7739724241369528, 2.234651778309412, 0.0, 0.7508241696907487], [0.38495870410433675, 0.011204636599378759, -5.377764062196205, -2.3156101989812883, 0.0, -0.00018456727161186848]],
[[3.7342940279181622, 4.68540662820932, 4.63943542112605, 0.3085569155559895, 0.0, -0.4198293606696021], [-4.886011957225701, -0.6953118170750949, -1.2270889997585979, 0.22674671352569575, 0.0, -0.0339420877822203]],
[[2.7132064326674596, -4.792480506405985, 1.6794732520385187, 2.488038825386118, 0.0, -0.002345179591040969], [-2.7520335446915234, -3.01937135240376, 3.2739254859150613, -3.3088916343746453, 0.0, -0.6466343077831692]],
[[-3.197303111232308, -4.805247585123754, -0.4622096277640333, 2.249339291921478, 0.0, -0.12534388480507086], [-0.145729018322176, -4.872191854093914, -0.1586930626821541, 4.418066523433662, 0.0, 0.5510276378507315]],
[[-3.4583715762032763, 2.4004969651540478, -2.9742712380375704, 0.33739393380297766, 0.0, -0.7625038658621777], [4.18747008099885, 4.007148541170123, -5.863199261165958, 4.569493362751167, 0.0, -0.5698702654093029]],
[[2.7770241057382012, -2.624587799650877, 4.075004223703333, 4.657491980429997, 0.0, 0.7423600937609829], [-0.4655075258268777, 1.0904246276127791, 3.462368296607149, 1.416133447590692, 0.0, 0.348745419406727]],
[[0.1394334377269022, 2.731650520792968, 4.65493158463736, -4.919530514701821, 0.0, -0.29886610933745594], [4.576037393658909, 0.1311671226788711, -2.283505263458636, 0.3919993742564811, 0.0, -0.43785171254670674]],
[[3.488176972685787, -3.211040750790015, -5.600037008417136, -1.3846155391177706, 0.0, -0.3527993963465928], [0.30000224895425287, -1.9408108426408397, -2.4570476674791215, -3.882587238244053, 0.0, -0.3928577103248976]],
[[-2.0533499731289027, 0.3058675560529416, -3.876464118043643, -4.320996418087086, 0.0, 0.45079570641140765], [1.5633352177585547, 1.3752089604363587, 0.9500539824186447, -4.609370838111335, 0.0, -0.2233458677974477]],
[[1.5037424173959169, 0.0545337373484287, 4.757646398741748, -3.1815977462846554, 0.0, 0.5532864104497517], [2.501362861060107, 1.6610166748532258, 6.131075024094647, -2.4303157739616985, 0.0, -0.7409353199528774]],
[[0.8813080107727425, 3.9771372790941797, 4.920120253489582, 3.15837477307684, 0.0, -0.7290229341403042], [1.9175758175888387, -1.2131905799459188, 0.2326154009227661, 1.57951465558813, 0.0, -0.4808989531985227]],
[[0.1729788384658928, 4.4696260381481405, 3.3358657197801467, -2.1760415603288727, 0.0, -0.43818091873100085], [1.8622208523746684, -3.3286079689994374, -1.3516078287667899, 1.1805234725280922, 0.0, -0.13833988349705673]],
[[4.6001730333591855, 1.9951204994957603, 6.281517657107994, -2.7993270021714824, 0.0, -0.21825216882474796], [2.3984099020943708, 4.964557250890969, -2.3078519415303513, -3.634554201764754, 0.0, -0.18224377388864776]],
[[3.701241366272118, 0.8227692867255971, -2.779191836614632, -3.1408876790982077, 0.0, -0.13964359250776837], [-3.8262445286528446, 1.8496874437413506, -0.7840025465539764, 0.5622932517197023, 0.0, -0.20878974255912497]],
[[-1.9206504737502916, 0.19391479305300763, 3.3715278498520718, 2.8922073617085973, 0.0, 0.5820775265865645], [-3.1207860837793113, -2.3049475438308176, -0.04785096759376284, 2.3912174711371073, 0.0, -0.4791682910254099]],
[[2.290137422891191, 0.6123960232764114, -4.712802514392742, -1.0240763129988286, 0.0, 0.4418778971116373], [0.10992981653113176, -3.17306641584175, 4.442373807324804, 4.553718926348093, 0.0, 0.7606007373145116]],
[[3.637599855700829, -2.150940347723732, -5.362618384650034, 2.6323720433028495, 0.0, -0.07426873065024575], [0.4229687400877449, 2.2663578282750727, 4.384470891530725, 2.681999778089822, 0.0, 0.3662213042063691]],
[[1.44143536068335, -1.1925151036488346, 2.048920406342962, -3.3634927389724667, 0.0, 0.7266626544662378], [-1.5333815962023434, 4.917509922936077, -3.3293606751676954, 0.8569426852165769, 0.0, -0.14657084562073985]],
[[3.588892672930397, -1.2728884502472004, 0.6927686558470967, 4.556565489917677, 0.0, 0.371759735220055], [3.162051391124809, -3.9891343969137525, 5.384539896577763, 1.0910916899130925, 0.0, 0.15166578280186593]],
[[-2.5148987256227007, -0.5002457894920456, -1.1191508761626947, -2.397003091131001, 0.0, 0.5818161867160554], [-3.14960072838095, -4.8033857456995595, 5.695733009586846, 1.8045080473103923, 0.0, -0.02106732154986879]],
[[-4.614383191185904, 2.8010046052467494, -5.1182355603079905, 1.3289268648447727, 0.0, -0.7635784698799837], [4.361585360849233, -0.7342561826446659, -3.102807169480159, -3.100346937063301, 0.0, -0.1337931537163265]],
[[-0.4194505177590946, -1.9165039303508835, -3.3742333068398582, -2.2257544824387407, 0.0, 0.4983112772681837], [-3.8865336390750183, 1.264372289542525, -2.804966142139915, 1.8217467418849527, 0.0, 0.2811266499705982]],
[[2.285071915957256, 1.0161421237021333, 5.682269851272249, -3.5682655227896953, 0.0, 0.48279859220693044], [-1.3678540347897106, -3.1344318771068416, -3.026594801683135, 1.2976234261194932, 0.0, -0.7342515874470413]],
[[4.444966028573068, -0.35901825695592393, -3.860451520697835, 0.8189487687999195, 0.0, 0.1886278362467999], [1.8422401746317405, -3.9656244498000195, 3.0847394559740042, -2.1802092992576405, 0.0, 0.3980784106057531]],
[[-1.1522688048325627, 3.59707846590619, 5.58197725031941, 2.0282489361072944, 0.0, 0.2099119489713981], [1.0596128199059809, -2.9987315714009553, -1.4151255392460804, -2.410168362485684, 0.0, -0.6682053197911237]],
[[0.46889156145354693, 2.9789902128260524, 4.026288768340455, -3.779501331697989, 0.0, 0.1602243775936989], [0.25514583364583654, -0.3609159418993162, -0.3588233581786664, 1.3271284287733138, 0.0, 0.6686312598957218]],
[[-0.9231297191971386, -4.4463395988814, 3.6258362056672517, -2.126948152106541, 0.0, -0.0779891156193332], [-1.9608769465923448, 0.26399524297145405, 1.5558701536929087, 2.7677545772495735, 0.0, 0.29254769243786893]],
[[-1.254598811526375, 4.507143064099161, 2.9153218530881926, 0.986584841970366, 0.0, -0.5403246560789353], [-3.4400547966379733, -4.419163878318005, 4.6015051579454465, 1.0111501174320878, 0.0, 0.3268396409087737]],
[[-3.8494543361022107, 1.090665392794814, -4.606945014611746, -2.5941038003465122, 0.0, -0.2715293361781592], [3.5913749094859764, 1.6609021309802579, 0.5172596148601798, -4.709861755756396, 0.0, 0.3671709651916335]],
[[3.3484214866564947, -3.9520389563013025, 3.0742429600601664, -1.3949916374371432, 0.0, -0.2209940191909321], [1.092383806181517, -1.0622044891174864, -1.1426272834843747, 0.09902409589606442, 0.0, 0.3300996960014855]],
[[4.890115134756002, 0.4954472684472542, -2.746414202400812, -4.227104339208273, 0.0, -0.08722711175256093], [-0.27192030244514687, -4.5147799909355415, -4.230789760608797, -3.8404928850205557, 0.0, 0.20010638725633334]],
[[2.838323508057506, 1.3483370605273013, -3.153617520246917, 2.580758646498177, 0.0, -0.29361806290300335], [4.372373574883975, -4.571345516477361, -0.7430846371645838, 4.1272228993519775, 0.0, -0.07067967886998117]],
[[-3.8651152810635048, 4.7448309443645655, 2.8743641769145167, -1.4853219410729857, 0.0, 0.32610538860111116], [2.9960460166005385, 1.4556185450829249, -1.073172986685357, 2.0603101127300913, 0.0, -0.39796222049960867]],
[[-4.825097290688474, 3.9157326718025836, -2.703514303063292, -2.010236225410421, 0.0, 0.4587263439358118], [-1.7552940252177116, 3.647103866902583, -0.6595757161726343, 0.48229906350094165, 0.0, -0.224353705884495]],
[[-1.9903554414408542, -2.5293816803818556, 5.357485313076474, 3.9160343827776405, 0.0, 0.2878904613555795], [0.6688465949967224, 0.46965880243031677, -3.638921398074758, 2.6975446818543567, 0.0, 0.6223967871487848]],
[[-0.053983544619785384, -2.719168955506638, -3.07280529894193, -1.036700902772231, 0.0, -0.1927129938999731], [4.965742301546493, -0.9180279966680294, 3.4167207069745, 2.6053668800888428, 0.0, -0.29843661721045045]],
[[1.7573141550355231, -4.552878177970539, -1.9691041247387702, 1.4401972701983095, 0.0, -0.33895742894883896], [4.493378066500052, -3.423298312262361, -1.4077732730243095, 0.8999367647231438, 0.0, -0.01917032617722836]],
[[-0.7981703285922617, -1.3676050303491438, -3.9599563308140233, 0.18282728167476225, 0.0, -0.7718807509916638], [4.689362067194862, 3.0138125601646095, 3.233477812374117, 1.7148430184182395, 0.0, -0.7388130428715352]],
[[4.841918505425767, -1.6658773398534663, 2.1827989919086956, -3.0360958278547545, 0.0, -0.22863495869389128], [3.133659531820605, -2.5215044648873897, -0.5201054274498853, 3.7730142149708765, 0.0, -0.17018249123754203]],
[[-4.126503570978249, -2.69522900448272, -1.1176395483957329, -1.89217301439913, 0.0, 0.1036032709386584], [0.4506370378385052, 3.0709944283037665, 5.254692052412807, 0.2209075284367712, 0.0, -0.11830097317214605]],
[[-1.3489441697730609, -0.48794084106662083, -0.04950711747527414, -4.2437793340486305, 0.0, 0.11272282992537352], [-2.103659942260414, -1.911955412115752, -1.5723524643847657, 3.1585667443101446, 0.0, 0.7364438147494576]],
[[4.2403511557442535, -3.4212941946520803, 4.610790839263373, -4.158430591385418, 0.0, 0.1155692724779942], [-2.1806023313708724, -2.437194421899945, 0.8040015687275446, -2.8017664818132393, 0.0, 0.2334530060696709]],
[[-1.9912666995338126, -3.13054183658122, -2.22195201225622, 1.6574957028830903, 0.0, 0.10519748816495245], [-1.0174604110123528, -1.2058508447540337, -6.15021380565942, -3.296343990062879, 0.0, -0.5915723099822998]],
[[3.244694759132818, -3.1865026750280023, 4.741628012008221, -0.004486926533608049, 0.0, 0.730304733536036], [-4.964544531902692, 4.236443466383852, 1.6266401874782348, -1.9570244500064815, 0.0, -0.24987429677493045]]])
GOALS=np.array([
[[2.917250380826646, 0.28894919752904435], [0.6804456109393229, 4.25596638292661]],
[[-0.8080548559670522, 1.8521950039675952], [-2.9554775026848255, 3.7811743639094537]],
[[1.2113383276929488, 0.2914209427703902], [-3.6542005465506646, 0.13578121265746468]],
[[-4.701237891214331, -0.43166775605288876], [1.4914404761476074, -2.215127173520247]],
[[2.7938292179375246, -3.0231492539974694], [3.6299323559922225, 4.834006771753128]],
[[-4.192587312351251, 2.3844029619897], [-0.5869077710404689, -3.416901322873488]],
[[1.7922999612094053, 3.037390361043755], [-1.190588668514616, -4.340636530940949]],
[[0.5535647382233044, 0.4338601754254032], [2.6089557560292898, 2.1237457408209126]],
[[1.8535981836779722, 4.533933461949365], [-4.960517336720855, 0.12192263385776592]],
[[2.2996447022081856, -3.9126392813102506], [3.939041702850812, 3.5715424707282963]],
[[-2.161716470420542, 1.0608318435882893], [4.442251360530422, 3.5273554110928487]],
[[-4.649634758985627, -2.015505291108206], [-4.414875081179252, 3.5706094258719894]],
[[3.064813579242699, -1.5774537438531566], [0.38888849067144626, -4.941262144489417]],
[[4.176298978365624, -2.358531467466264], [2.1777368747355608, 3.6571503419015414]],
[[4.456831868184075, -4.3995531968461465], [3.6404210354621096, 3.772905261480302]],
[[1.3571911546100166, 3.4731238748662623], [2.361746251160107, -4.791928880765096]],
[[-2.2768359793341606, 2.186059336011658], [2.8300360945901044, 3.5032763977499535]],
[[-4.975351187991138, 3.840321823721048], [3.8494753837647355, -1.9959031063351906]],
[[-1.794807164348069, -1.3358524691648488], [2.096515625881274, 4.001424305233735]],
[[-0.9763427118579715, -3.869592993056037], [-0.5296915373251068, 0.8544511651103566]],
[[-3.2025475327408803, 0.3882625857515336], [-2.1723439538902243, 2.609398694620987]],
[[-3.072990261726317, 4.707951013702171], [-2.651916453985571, -4.736461514901293]],
[[-2.580135752961149, 2.203086925547387], [-4.416226941882789, -0.9033107384221575]],
[[-3.6376567798095305, 0.44136287638179805], [0.18176346825845435, 2.6685510629850544]],
[[-4.082158652037979, -1.5481375669464237], [1.6275252318558762, -0.5828651153876416]],
[[4.650268198659541, -1.0660126087836663], [-4.204424287131058, -1.4859257558620378]],
[[-4.865793902605975, 1.4847278482319046], [3.6936874594903415, 1.090357034477199]],
[[2.967174185877428, -4.541978374772669], [4.125982744103684, -2.8618401366293336]],
[[4.472320731772605, 4.043182436182063], [0.5013015562183227, -3.0299171235925284]],
[[2.9270562957071604, 1.2740059627935976], [-0.5651115622159857, 4.634653524074658]],
[[-2.1904302532220923, -0.615658488539502], [-0.16750956410536766, 3.684894862873948]],
[[3.155029157504419, 4.444417523833376], [4.195851420595398, -0.8516920364897897]],
[[4.8093886318780505, 1.0081609221591314], [3.1396851994299553, 2.0864515216319868]],
[[-4.7941550570419755, 4.699098521619943], [3.324426408004218, -2.8766088932172384]],
[[-1.0504998156899417, 3.020471186286665], [-2.455788741401207, -4.431150633488482]],
[[4.605262252056674, -0.43378891061062763], [-0.7234847893839103, -3.865362988800296]],
[[3.5618204859562432, 1.501024210875526], [4.907216847578001, -0.29649251596214565]],
[[0.07934340703791598, -4.150920452052999], [-0.736502089344623, 2.457169596938721]],
[[-2.440075687020377, -4.759886458546391], [-4.0127404515118945, -1.9956356463871163]],
[[-3.876879746072902, -3.581028451517029], [-0.5504092328781507, 2.3198022518691186]],
[[2.216504565659716, -0.012196500917341702], [-0.5778721710580577, 3.0243052780442543]],
[[-1.534587954373027, -1.482351825034376], [-3.54533143503845, 4.726646847174504]],
[[-3.2565661397587875, -2.8181743881678933], [1.4674554420505856, -2.501685908576654]],
[[-1.0843797001966848, -0.9520596100336522], [-3.103689856156975, 2.671056444827393]],
[[-1.7386657964968997, -4.099048897562287], [3.342177499238341, -4.037883950579696]],
[[-4.281960454503622, 3.985288501792178], [-0.7948617394495372, 0.8216978793839562]],
[[2.8888121916447007, -1.1931817284682564], [-3.3728378366218426, 0.4738767673029729]],
[[1.2775224240968273, -0.8902558963869778], [-0.974566421346549, -0.4983792799615774]],
[[1.9240127640408602, 3.7444156057269122], [-1.6260309536137663, 4.924592256795675]],
[[-3.5926481928598344, -0.1299822414343783], [3.8104672024478035, -0.4700496357395201]]])

def run_main():
    for i in range(30):
        print("## NO COLLISIONS iter {}".format(i))
        sys.stdout.flush()
        # Start state for each car. See comment above `NUM_STATE_DIMENSIONS` for a
        # description of the state variables.
        START_STATE = STARTS[i]
        # Goal position for each car.
        GOAL_CENTER_OF_GRAVITY_POSITION = GOALS[i]
        # The collisions between cars must be specified up front. See function
        # comment of `solve()`.
        COLLISION_SEQUENCE = []
        # See function comment of `solve()`.
        NUM_TIME_SAMPLES = 60

        solve(
            start_state=START_STATE,
            goal_position=GOAL_CENTER_OF_GRAVITY_POSITION,
            num_time_samples=NUM_TIME_SAMPLES,
            collision_sequence=COLLISION_SEQUENCE,
        )
    for i in range(30):
        print("## ONE COLLISION iter {}".format(i))
        sys.stdout.flush()
        # Start state for each car. See comment above `NUM_STATE_DIMENSIONS` for a
        # description of the state variables.
        START_STATE = STARTS[i]
        # Goal position for each car.
        GOAL_CENTER_OF_GRAVITY_POSITION = GOALS[i]
        # The collisions between cars must be specified up front. See function
        # comment of `solve()`.
        COLLISION_SEQUENCE = [(0, 1)]
        # See function comment of `solve()`.
        NUM_TIME_SAMPLES = 60

        solve(
            start_state=START_STATE,
            goal_position=GOAL_CENTER_OF_GRAVITY_POSITION,
            num_time_samples=NUM_TIME_SAMPLES,
            collision_sequence=COLLISION_SEQUENCE,
        )

def gen_pos():
    return np.random.uniform(-5, 5, 2)

def gen_state():
    x, y = gen_pos()
    head = np.random.uniform(- 2 * np.pi, 2 * np.pi, 1)[0]
    v = np.random.uniform(-5, 5, 1)[0]
    steer = np.random.uniform(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE, 1)[0]
    state = np.array([x, y, head, v, steer])
    return state

def gen_start_goal(seed=0):
    np.random.seed(seed=seed)
    start_state = np.array([gen_state(), gen_state()])
    goal_state = np.array([gen_pos(), gen_pos()])
    return start_state, goal_state

def is_good(start, goal):
    if (goal[1][0] - goal[0][0]) ** 2 + (goal[1][1] - goal[0][1]) ** 2 < (2 * CAR_RADIUS) ** 2:
        return False
    if squared_distance_of_centers_of_gravity(start, 0, 1) < (2 * CAR_RADIUS) ** 2:
        return False
    return True

def gen():
    success = 0
    seed = 0
    starts = []
    goals = []
    while success < 50:
        start, goal = gen_start_goal(seed=seed)
        if is_good(start, goal):
            success += 1
            starts.append(start.tolist())
            goals.append(goal.tolist())
        seed += 1
    print("STARTS=np.array([\n{}])".format(",\n".join(str(x) for x in starts)))
    print("GOALS=np.array([\n{}])".format(",\n".join(str(x) for x in goals)))

if __name__ == "__main__":
    run_main()
