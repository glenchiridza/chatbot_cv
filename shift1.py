import random

def generate_shift_schedule(employees, shifts_per_day, days_per_week):
    # Calculate the total number of shifts needed for the week
    total_shifts = shifts_per_day * days_per_week

    # Separate morning-only and night-only employees
    morning_only_employee = None
    night_only_employee = None
    general_employees = []

    for emp in employees:
        if emp['name'] == 'Employee1':
            morning_only_employee = emp
        elif emp['name'] == 'Employee2':
            night_only_employee = emp
        else:
            general_employees.append(emp)

    # Shuffle the list of general employees
    random.shuffle(general_employees)

    # Initialize shift schedule and employee indexes
    shift_schedule = []
    morning_shift_count = 0
    day_shift_count = 0
    night_shift_count = 0

    for shift in range(total_shifts):
        # Determine if it's a weekend shift
        is_weekend = shift % shifts_per_day >= 1

        # Assign the morning-only employee to morning shift
        if shift % shifts_per_day == 0:
            if random.choice([True, False]):
                assigned_employee = morning_only_employee
            else:
                assigned_employee = random.choice(general_employees)
            morning_shift_count += 1

        # Assign the night-only employee to night shift
        elif shift % shifts_per_day == 2:
            if random.choice([True, False]):
                assigned_employee = night_only_employee
            else:
                assigned_employee = random.choice(general_employees)
            night_shift_count += 1

        # Assign general employees to other shifts
        else:
            assigned_employee = random.choice(general_employees)
            day_shift_count += 1

        # Assign the employee to the shift
        shift_schedule.append(assigned_employee['name'])

    # Ensure exactly 3 employees are assigned to each shift per day
    enforce_shift_counts(shift_schedule, shifts_per_day, morning_shift_count, day_shift_count, night_shift_count)

# Rest of the code...

def enforce_shift_counts(shift_schedule, shifts_per_day, morning_shift_count, day_shift_count, night_shift_count):
    # Get the number of days
    total_days = len(shift_schedule) // shifts_per_day

    # Iterate over each day
    for day in range(total_days):
        morning_shift_index = day * shifts_per_day
        day_shift_index = morning_shift_index + 1
        night_shift_index = morning_shift_index + 2

        # Check the morning shift count and update if necessary
        if morning_shift_count < 3:
            while morning_shift_count < 3:
                random_employee = random.choice(shift_schedule)
                if random_employee != shift_schedule[morning_shift_index] and random_employee != shift_schedule[day_shift_index] and random_employee != shift_schedule[night_shift_index]:
                    shift_schedule[morning_shift_index] = random_employee
                    morning_shift_count += 1

        # Check the day shift count and update if necessary
        if day_shift_count < 3:
            while day_shift_count < 3:
                random_employee = random.choice(shift_schedule)
                if random_employee != shift_schedule[morning_shift_index] and random_employee != shift_schedule[day_shift_index] and random_employee != shift_schedule[night_shift_index]:
                    shift_schedule[day_shift_index] = random_employee
                    day_shift_count += 1

        # Check the night shift count and update if necessary
        if night_shift_count < 3:
            while night_shift_count < 3:
                random_employee = random.choice(shift_schedule)
                if random_employee != shift_schedule[morning_shift_index] and random_employee != shift_schedule[day_shift_index]:
                    shift_schedule[night_shift_index] = random_employee
                    night_shift_count += 1

        # Print the shift schedule
        for day in range(days_per_week):
            print(f"Day {day+1}:")
            for shift in range(shifts_per_day):
                index = (day * shifts_per_day) + shift
                print(f"Shift {shift+1}: {shift_schedule[index]}")
            print()

# Define the list of employees
employees = [
    {"name": "Employee1", "shift_availability": ["morning"]},
    {"name": "Employee2", "shift_availability": ["night"]},
    {"name": "Employee3", "shift_availability": ["morning", "day", "night"]},
    {"name": "Employee4", "shift_availability": ["morning", "day", "night"]},
    {"name": "Employee5", "shift_availability": ["morning", "day", "night"]},
    {"name": "Employee6", "shift_availability": ["morning", "day", "night"]},
    {"name": "Employee7", "shift_availability": ["morning", "day", "night"]}
]

# Define the number of shifts per day and days per week
shifts_per_day = 3
days_per_week = 7

# Generate the shift schedule
generate_shift_schedule(employees, shifts_per_day, days_per_week)

