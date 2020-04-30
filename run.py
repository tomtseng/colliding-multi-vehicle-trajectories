import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pydrake.math
import pydrake.solvers.mathematicalprogram

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
# Coefficient of restitution, determining the elasticity of car collisions
RESTITUTION = 0.1

EPSILON = 1e-2
SECONDS_TO_MILLISECONDS = 1000


def instantaneous_change_in_state(state, control):
    """Returns the instantaneous change in state with respect to time assuming
    no collisions.

    This uses a simple car model. See http://planning.cs.uiuc.edu/node658.html .

    Arguments:
      state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        [x position, y position, heading, linear speed, steering angle] for each
        car
      control: (NUM_CARS x NUM_CONTROL_DIMENSIONS)-dimensional array of floats
        [acceleration, steering velocity] for each car

    Returns:
      (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
    """
    # If the state and control are symbolic variables used for optimization,
    # then we need to call pydrake.symbolic functions rather than numpy
    # functions to represent mathematical operations.
    m = pydrake.symbolic if state.dtype == object else np

    change_in_state = np.empty_like(state)
    for i in range(NUM_CARS):
        heading = state[i, 2]
        speed = state[i, 3]
        change_in_state[i, 0] = speed * m.cos(heading)
        change_in_state[i, 1] = speed * m.sin(heading)
        change_in_state[i, 2] = speed * m.tan(state[i, 4])
    change_in_state[:, 3:] = control
    return change_in_state


def discrete_dynamics(state, control, time_step_size):
    """Returns the state after one time step.

    Arguments:
      state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array
        [x position, y position, heading, linear speed, steering angle] for each
        car
      control: (NUM_CARS x NUM_CONTROL_DIMENSIONS)-dimensional array
        [acceleration, steering velocity] for each car
      time_step_size: float
        Size of one time step in seconds.

    Returns:
      (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array
    """
    next_state = (
        state
        + instantaneous_change_in_state(state=state, control=control) * time_step_size
    )
    # Account for collisions.
    # TODO(tomtseng)
    # will need to change state to have 2D velocity to model this
    # properly
    """
    for i in range(NUM_CARS - 1):
        for j in range(i + 1, NUM_CARS):
            if (state[0][0] - state[1][0]) ** 2 + (state[0][1] - state[1][1]) ** 2 <= (2 * CAR_RADIUS) ** 2:
                # TODO(tomtseng) do something real here
                continue
    """
    return next_state


def plot_trajectory(solver_result, state_vars, goal_state, time_step_size):
    """Plots and animates a state trajectory.

    Arguments:
      solver_result: pydrake.solvers.mathematicalprogram.MathematicalProgramResult
        Result of solving for the trajectory.
      state_vars: (num_time_steps x NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array
        Solver variables representing State of cars at every time step.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired final state.
      time_step_size: float
        Size of one time step in seconds.
    """

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    # Set axis limits.
    xs = solver_result.GetSolution(state_vars[:, :, 0])
    ys = solver_result.GetSolution(state_vars[:, :, 1])
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
            x = solver_result.GetSolution(state_vars[t, i, 0])
            y = solver_result.GetSolution(state_vars[t, i, 1])
            cars[i].center = (x, y)

            heading = solver_result.GetSolution(state_vars[t, i, 2])
            car_headings[i].set_xdata([x, x + np.cos(heading) * CAR_RADIUS])
            car_headings[i].set_ydata([y, y + np.sin(heading) * CAR_RADIUS])
        return animated_objects

    SECONDS_TO_MILLISECONDS = 1000
    matplotlib.animation.FuncAnimation(
        fig,
        animate,
        init_func=lambda: [],
        frames=len(state_vars),
        interval=time_step_size * SECONDS_TO_MILLISECONDS,
        blit=True,
    )

    plt.show()


def solve(start_state, goal_state, time_step_size, num_time_steps):
    """Solves for vehicle trajectory from the start state to the goal state.

    Arguments:
      start_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of floats
        Desired start state of the cars.
      goal_state: (NUM_CARS x NUM_STATE_DIMENSIONS)-dimensional array of Optional[float]
        Desired final state of the cars.
      time_step_size: float
        Size of one time step in seconds.
      num_time_steps: int
        Number of time steps for the cars to get from start to goal.
    """

    solver = pydrake.solvers.mathematicalprogram.MathematicalProgram()
    state_vars = solver.NewContinuousVariables(
        (num_time_steps + 1) * NUM_CARS * NUM_STATE_DIMENSIONS
    ).reshape((num_time_steps + 1, NUM_CARS, NUM_STATE_DIMENSIONS))
    control_vars = solver.NewContinuousVariables(
        num_time_steps * NUM_CARS * NUM_CONTROL_DIMENSIONS
    ).reshape((num_time_steps, NUM_CARS, NUM_CONTROL_DIMENSIONS))

    solver.AddConstraint(pydrake.math.eq(state_vars[0], start_state))
    for c in range(NUM_CARS):
        for j in range(NUM_STATE_DIMENSIONS):
            if goal_state[c, j] is not None:
                difference_from_goal = state_vars[-1, c, j] - goal_state[c, j]
                solver.AddConstraint(difference_from_goal <= EPSILON)
                solver.AddConstraint(difference_from_goal >= -EPSILON)

    # Penalize solutions that use large accelerations.
    solver.AddCost(
        time_step_size * sum(accel ** 2 for accel in control_vars[:, :, 0].flatten())
    )

    # Enforce discrete dynamics.
    for t in range(num_time_steps):
        solver.AddConstraint(
            pydrake.math.eq(
                state_vars[t + 1],
                discrete_dynamics(
                    state=state_vars[t],
                    control=control_vars[t],
                    time_step_size=time_step_size,
                ),
            )
        )
    steering_angles = state_vars[:, :, 4]
    solver.AddConstraint(pydrake.math.le(steering_angles, MAX_STEERING_ANGLE))
    solver.AddConstraint(pydrake.math.ge(steering_angles, -MAX_STEERING_ANGLE))

    # Constrain control inputs.
    accelerations = control_vars[:, :, 0]
    steering_velocities = control_vars[:, :, 1]
    solver.AddConstraint(pydrake.math.le(accelerations, MAX_ACCELERATION))
    solver.AddConstraint(pydrake.math.ge(accelerations, -MAX_ACCELERATION))
    solver.AddConstraint(pydrake.math.le(steering_velocities, MAX_STEERING_VELOCITY))
    solver.AddConstraint(pydrake.math.ge(steering_velocities, -MAX_STEERING_VELOCITY))

    solver_result = pydrake.solvers.mathematicalprogram.Solve(solver)
    if solver_result.is_success():
        plot_trajectory(
            solver_result=solver_result,
            state_vars=state_vars,
            goal_state=goal_state,
            time_step_size=time_step_size,
        )
    else:
        infeasible_constraints = pydrake.solvers.mathematicalprogram.GetInfeasibleConstraints(
            solver, solver_result
        )
        print("Failed to solve.")
        print("Infeasible constraints:")
        for constraint in infeasible_constraints:
            print(constraint)


START_STATE = np.array([[0, 0, 0, 0, 0], [4, -3, np.pi / 4, 0, 0]])
# Don't enforce heading on goal state --- there are undesirable results if the
# cars turn more than 360 degrees in one direction.
GOAL_STATE = np.array([[6, 2, None, 0, 0], [3, 3, None, 0, 0]])
TIME_STEP_SIZE = 0.1  # seconds
NUM_TIME_STEPS = 50

solve(
    start_state=START_STATE,
    goal_state=GOAL_STATE,
    time_step_size=TIME_STEP_SIZE,
    num_time_steps=NUM_TIME_STEPS,
)

# TODO(tomtseng) progression:
# (1) [DONE] dummy implementation: skip collisions just use discrete dynamics
#   directly in direct transcription. figure out how to plot things and generate
#   animations
# (2) add collisions to discrete dynamics
# (3) continuous dynamics with collocation, with known mode sequence
# (4) floating-base coordinates to remove known mode sequence assumption
