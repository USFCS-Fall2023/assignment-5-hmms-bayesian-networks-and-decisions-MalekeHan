from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# New CPD for KeyPresent
cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)

# Updated CPD for Starts that includes KeyPresent
cpd_starts_updated = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01],  # Starts=yes
        [0.01, 0.99, 0.99, 0.99, 0.01, 0.99, 0.99, 0.99]   # Starts=no
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ["yes", "no"],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ["Full", "Empty"],
        "KeyPresent": ["yes", "no"]
    }
)

# Add the new node and CPD to the network
car_model.add_node("KeyPresent")
car_model.add_edge("KeyPresent", "Starts")
car_model.add_cpds(cpd_key_present, cpd_starts_updated)



# Associating the parameters with the model structure
car_model.add_cpds(cpd_battery, cpd_gas, cpd_radio, cpd_ignition, cpd_key_present, cpd_starts_updated, cpd_moves)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


# Query 1: Given that the car will not move, what is the probability that the battery is not working?
battery_not_working_given_moves_not = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
print("Probability of Battery not working given car does not move:")
print(battery_not_working_given_moves_not)

# Query 2: Given that the radio is not working, what is the probability that the car will not start?
car_not_start_given_radio_not = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
print("Probability of car not starting given radio doesn't turn on:")
print(car_not_start_given_radio_not)

# Query 3: Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
radio_working_given_battery_works = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
radio_working_given_battery_works_and_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
print("Probability of radio working given battery works:")
print(radio_working_given_battery_works)
print("Probability of radio working given battery works and car has gas:")
print(radio_working_given_battery_works_and_gas)

# Query 4: Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
ignition_fails_given_moves_not = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
ignition_fails_given_moves_not_and_gas_empty = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
print("Probability of ignition failing given car doesn't move:")
print(ignition_fails_given_moves_not)
print("Probability of ignition failing given car doesn't move and gas is empty:")
print(ignition_fails_given_moves_not_and_gas_empty)

# Query 5: What is the probability that the car starts if the radio works and it has gas in it?
car_starts_given_radio_works_and_gas = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
print("Probability of car starting given radio works and it has gas in it:")
print(car_starts_given_radio_works_and_gas)

# Query with KeyPresent evidence
print(car_infer.query(variables=["Starts"], evidence={"KeyPresent": "yes", "Gas": "Full", "Ignition": "Works"}))