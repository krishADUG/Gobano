## Create an robot agnostic IK solver.

1. The project should have a robot folder with 2 robots UR5 and Franka. each robot  in its own folder should have a urdfs, a dimension file that has all its joint limits, dh parameters and workspace values etc. 

2. Now we will have a model that uses the pinnochio library to compute the FK and Jacobians based onthe input robot selction from the robor folder

3. From this forward kinmatics and jacobians, we design an QP based solver using osqp solver; The constraints for this solver are supposed to be defined independently for each robot in the robot folder.

so the sover should take its inputs: 
solver (
    robot model (getting it from pinnochio lib), 
    goal cartesian pose
    start position
    constraints
) 

for the constraints
    consider joint limits of the robot
    have jump limit for each joint, for each iteration 
    also have a maximum jump limit, a max rotaion a joint can make from its start point to the final goal point.

the output of the solver is classified into 
    Success
    approximate
    invalid input
    non convergent


A solution from the solver is considered success if the final optimisaition is less than the error tolerance defined.

A solution from the solver is considered approximate if the final optimisation is more that the error tolerance defined but less than a (approx_multiplier * error tolerence) 

A solution from the solver is declared non convergent if the final optimisation is greater than the (approx_multiplier * error tolerence)

After the solution has been classified i want to post proscess the solution to add a note or tag.Specifically in the case of approxiamte and non convergent cases. so create dummy funtions named joint_limit, jump_limit, solver_stuck, outside_worspace.

The logic for each function to post process the solution will be done later.

All this will be in my solver file, anythign that is related to the configuration fo the solver like tolerance and weight have to be created in yaml file under the robot folder, a unique solver_config file for each robot.



## demonstration

Now lets plan test scenarios, create a seperate file to define the testing scenes.

scene1 reachable
lets define a home_postion for each robot, ideally a elbow up configuration. (define this is robot folder for each robot). now use this as the start postion and offset this postion by a small value and use the solver to find an reachable postion.


scene2 close to joint limit
choose a new start location that is already close to a joint limit and offset this start postion and define a new goal point that is actually reachable but not reachable from this start postion based on the set solver constraints

scene3 close to jump limit
choose a new start location and offset this start postion and define a new goal point that is actually reachable but not reachable from this start postion based on the set solver constraints. So in this case its not the joint limit but more emphasis on the joint jumps

scene4 exhuast the solver
choose a new start location and offset this start postion and define a new goal point that is actually reachable but not reachable from this start postion based on the set solver constraints. The solver solver should not cause any joint limit or jump limit but more like the solver cant not  find a better solution in the last consicutiove iterations (the iteration limit mentioned in the solve config file)

scene5 outside workspace
for this case choose the same home postion , but now choose a new goal point that is just outside the workspace of the robot. 





