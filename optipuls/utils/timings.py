from dolfin import timings, TimingClear, TimingType

time_table = timings(TimingClear.keep, [TimingType.wall])

def save_timings(filename='timings.log'):
    with open(filename, 'w') as outfile:
        outfile.write(time_table.str(True))
