import time

#HELPER FUNCTIONS
#NOTE: functions only see globals from the file they're defined in, so readings / args are passed in as arguments

#helper function that pulls the info out of a tt_smi.csv line
#NOTE: columns are TT_SMI_FIELDS from xla_yolo/hardware.py: time,power_w,board_power_w,aiclk_mhz,voltage_v,current_a,asic_temp_c
def parse(line):
    t, power_w, board_power_w, aiclk_mhz, voltage_v, current_a, asic_temp_c = (float(v) for v in line.strip().split(","))
    power = {"power": power_w, "board_power": board_power_w} #watts, tt-smi's power reading (TDP telemetry) and the whole card's input power
    temp = {"asic": asic_temp_c} #degrees C
    return power, temp, aiclk_mhz, voltage_v, current_a #returns power, temp, and the card's AI clock, core voltage and current

#helper function that prints average and max of every reading inside one window [start, end]
#NOTE: readings is the list of (time stamp, line) read from tt_smi.csv
def report(title, start, end, readings):
    inside = [line for t, line in readings if start <= t <= end] #metrics found within the window of the run
    print(f"\n    {title}: {time.strftime('%m-%d %H:%M:%S', time.localtime(start))}, {end - start:.1f} s, {len(inside)} readings    ")
    if not inside: #if no metrics found
        print("NO AVAILABLE LOG READINGS IN THIS TIME WINDOW")
        return
    powers, temps, clocks, voltages, currents = {}, {}, [], [], []
    #NOTE: powers and temps are dictionaries keyed by reading names (e.g. "board_power")
    for line in inside: #fill dictionaries and lists
        power, temp, clock, voltage, current = parse(line)
        for k, v in power.items():
            powers.setdefault(k, []).append(v)
        for k, v in temp.items():
            temps.setdefault(k, []).append(v)
        clocks.append(clock)
        voltages.append(voltage)
        currents.append(current)
    print(f"  {'POWER (W)':<16} {'avg':>7} {'max':>7}")
    for k in sorted(powers): #NOTE: loop over k reading names
        v = powers[k]
        print(f"  {k:<16} {sum(v) / len(v):>7.2f} {max(v):>7.2f}")
    print(f"  {'TEMP (C)':<16} {'avg':>7} {'max':>7}")
    for k in sorted(temps):
        v = temps[k]
        print(f"  {k:<16} {sum(v) / len(v):>7.1f} {max(v):>7.1f}")
    print(f"  {'AICLK (MHz)':<16} {sum(clocks) / len(clocks):>7.0f} {max(clocks):>7.0f}")
    print(f"  {'VCORE (V)':<16} {sum(voltages) / len(voltages):>7.3f} {max(voltages):>7.3f}")
    print(f"  {'CURRENT (A)':<16} {sum(currents) / len(currents):>7.2f} {max(currents):>7.2f}")

#helper function that is True when a row's batch size / step / optimization match the options given (None = any)
def wanted(batch_size, step, optimization, args):
    return ((args.batch_size is None or batch_size == args.batch_size) and (args.step is None or step == args.step)
            and (args.optimization is None or optimization == args.optimization))
