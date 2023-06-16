def multiple_depth_measurement(
        self,
        measurement,
        depth_init,
        depth_end,
        step = 10,
        resolution="1/4",
        thd_filter = True):
    m = measurement

    if depth_end is depth_init:
        print("You have to choose two different limits for measured depth.")
        return

    p = PiStepper(res=resolution)
    if depth_init is not self.position:
        pre_cycles = (self.position - depth_init) / self.wall_mm_per_cycle
        if pre_cycles < 0:
            pre_direction = False
        else:
            pre_direction = True
        p.on()
        p.enable()
        p.turn(abs(pre_cycles), pre_direction)
        p.disable()
        self.position = depth_init

    cycles = abs(step / self.wall_mm_per_cycle)
    if depth_init > depth_end:
        step = -step

    if step < 0:
        direction = True
    else:
        direction = False
    
    depth = np.arange(
        depth_init, 
        depth_end+step, 
        step).astype(int)
    

    for d in depth:
        running = True
        while running:
            print(d, depth[-1], d == depth[-1])
            for s in range(m.sub_measurements):
                f = os.path.join(self.trees[4][0], self.trees[1]+f"_wav_d{d}_{s}.wav")
                m.measure(f, thd_filter=thd_filter)
                sleep(1)
            if input(f"Repeat measurement for depth {d} mm? [y/N]").lower() == "y":
                continue
            else:
                running = False    
        if d is not depth[-1]:
            p.on()
            p.enable()
            p.turn(abs(cycles), direction)
            p.disable()