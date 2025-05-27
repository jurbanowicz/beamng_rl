from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat
from training import Training

# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:25252
bng = BeamNGpy("192.168.1.20", 25252)
# Launch BeamNG.tech
print("Connecting to BeamNG...")
bng.open()
print("Connected!")
print("Loading scenario...")

scenario = Scenario("smallgrid", "rl_training")
vehicle = Vehicle("main_car", model="sunburst2", license="PYTHON")

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, 
        pos=(0, 0, 0),
        rot_quat=angle_to_quat((0, 0, 90))
        ) 
# Place files defining our scenario for the simulator to read
scenario.make(bng)

# Load and start our scenario
bng.scenario.load(scenario)
bng.scenario.start()
# Make the vehicle's AI span the map
trainer = Training(bng, scenario, vehicle)
trainer.start()

input('Hit Enter when done...')

# Disconnect BeamNG
bng.disconnect()
