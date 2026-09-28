# Jetson performance mode

The ZED camera is attached to the Jetson (`jetson-robotnik`: Jetson Nano
Developer Kit, tegra210, L4T R32.7.6), and `tools/record_data.py` runs there.
By default the Jetson scales its clocks dynamically, and while idle the GPU
sits at its minimum (76.8 MHz). The ZED computes depth on the GPU, so lock
the clocks at maximum before recording.

## Check the current mode

```bash
nvpmodel -q                  # power mode: MAXN (ID 0) = performance, 5W (ID 1) = low power
sudo jetson_clocks --show    # clocks: locked when MinFreq = MaxFreq = CurrentFreq
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq   # CPU frequency (kHz)
cat /sys/class/devfreq/57000000.gpu/cur_freq                # GPU frequency (Hz)
```

### Not locked (the default after boot)

The output below was taken on 2026-09-28. The power mode is already MAXN,
but the CPU and GPU scale down when idle:

```
cpu0: Online=1 Governor=schedutil MinFreq=102000 MaxFreq=1479000 CurrentFreq=825600 IdleStates: WFI=1 c7=1
GPU MinFreq=76800000 MaxFreq=921600000 CurrentFreq=76800000
EMC MinFreq=204000000 MaxFreq=1600000000 CurrentFreq=1600000000 FreqOverride=0
Fan: PWM=0
NV Power Mode: MAXN
```

### Locked (after `sudo jetson_clocks`)

```
cpu0: Online=1 Governor=schedutil MinFreq=1479000 MaxFreq=1479000 CurrentFreq=1479000 IdleStates: WFI=0 c7=0
GPU MinFreq=921600000 MaxFreq=921600000 CurrentFreq=921600000
EMC MinFreq=204000000 MaxFreq=1600000000 CurrentFreq=1600000000 FreqOverride=1
Fan: PWM=0
NV Power Mode: MAXN
```

| | Dynamic | Locked |
|---|---|---|
| CPU (4 cores) | 0.1–1.479 GHz, `schedutil` | 1.479 GHz, idle states off |
| GPU | 76.8–921.6 MHz | 921.6 MHz |
| EMC (memory) | up to 1.6 GHz | 1.6 GHz (`FreqOverride=1`) |

## Lock the clocks at maximum

```bash
sudo nvpmodel -m 0           # MAXN (it persists across reboots)
sudo jetson_clocks --store   # optional: save the current settings for --restore
sudo jetson_clocks           # lock CPU, GPU and EMC at maximum
sudo jetson_clocks --show    # check
```

- `jetson_clocks` does **not** persist across a reboot. To apply it at boot,
  add `/usr/bin/jetson_clocks` to a systemd service or to `/etc/rc.local`.
- To undo it without rebooting: `sudo jetson_clocks --restore`. This needs
  the earlier `--store`.

## Temperature and fan

`jetson_clocks` normally sets the fan to full speed, but here it stayed at
`PWM=0`. Either there is no fan, or it didn't respond. To turn a connected
fan on by hand:

```bash
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

Check the temperatures:

```bash
cat /sys/devices/virtual/thermal/thermal_zone*/type /sys/devices/virtual/thermal/thermal_zone*/temp   # millidegrees
sudo tegrastats --interval 2000   # live CPU/GPU load, frequencies and temperatures (Ctrl+C to stop)
```

These readings were taken idle with the clocks locked and the fan at 0
(2026-09-28):

| Sensor | °C |
|---|---|
| CPU | 37–38 |
| GPU | 38.5 |
| AO | 47.5–48 |
| PLL | 38–38.5 |
| thermal (fan estimate) | 37.75–38 |
| PMIC | 50 (a constant estimate on the Nano; ignore it) |

The board drew about 2.2 W (`POM_5V_IN`).

The Nano throttles at about 97 °C. Under load, keep CPU and GPU below
~80 °C, turning on the fan or adding a heatsink if needed. Run `tegrastats`
while recording and watch `CPU@`, `GPU@` and `GR3D_FREQ` (GPU load, which
should read 921 MHz).

## Effect on recording

Before locking the clocks, `record_data.py` video mode saved about
**2.1 fps** (`zed_highres_20260928_112333`: 23 frames in 10.4 s). Saving
the RGB, depth and depth-PNG files for every frame is the main cost. To
measure the rate with the clocks locked:

```bash
cd ~/renee_perception && python3 tools/record_data.py --headless --video --duration 30
```

The rate is printed in the `RGB video: … (X.XX fps, the capture rate)` line.
