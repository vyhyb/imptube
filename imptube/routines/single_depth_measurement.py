def single_depth_measurement(
        sample,
        measurement,
        depth,
        resolution = "1/4",
        thd_filter = True):
    m = measurement
    if depth is not sample.position:
        pre_cycles = (sample.position - depth) / sample.wall_mm_per_cycle
        if pre_cycles < 0:
            pre_direction = False
        else:
            pre_direction = True
        p = PiStepper(res=resolution)
        p.on()
        p.enable()
        p.turn(abs(pre_cycles), pre_direction)
        p.disable()
        self.position = depth

    running = True
    while running:
        for s in range(m.sub_measurements):
            f = os.path.join(self.trees[4][0], self.trees[1]+f"_wav_d{depth}_{s}.wav")
            m.measure(f, thd_filter=thd_filter)
            sleep(0.5)
        if input("Repeat measurement? [y/N]").lower() == "y":
            continue
        else:
            running = False