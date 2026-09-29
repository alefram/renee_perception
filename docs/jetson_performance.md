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
cd ~/renee_ws/src/renee_perception && python3 tools/record_data.py --headless --video --duration 30
```

The rate is printed in the `RGB video: … (X.XX fps, the capture rate)` line.

## Power supply

On 2026-09-28 the ZED 2i kept dropping off the USB bus (`error -71`), and
its images came out torn (horizontal bands shifted out of place). The cause
is the Jetson's 5 V input: it sags under load. Measured with the INA3221
monitor (needs root):

```bash
d=/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0
sudo sh -c "while true; do echo \"\$(date +%H:%M:%S) \$(cat $d/in_voltage0_input) mV \$(cat $d/in_current0_input) mA\"; sleep 0.5; done" | tee /tmp/power.log
```

| Condition | `POM_5V_IN` |
|---|---|
| Idle, no ZED | 4.86–4.91 V at 0.35 A |
| MAXN, ZED SDK starting | **4.55 V at 1.5 A**: `OC ALARM` and `-71` in the same second, camera lost |
| 5 W mode (`sudo nvpmodel -m 1`), recording | 4.66 V minimum at 1.0 A: no disconnects, frames still partly torn |

That's about 0.27 Ω in the supply path; the USB spec needs at least 4.75 V at
the port, and the ZED sits behind a Genesys hub (`05e3:0620`) on top of it.
The ZED cable also ran through a second USB3 extension inside the Vogui;
without it the camera started working, so the plan is to replace that cable.
Until the cable/supply is fixed (no passive USB extensions; a 5 V converter
rated for 4 A or more, set to 5.1–5.2 V, with short thick wires; or a powered
USB3 hub for the ZED):

- Don't run `jetson_clocks`: it raises the load and makes things worse.
- 5 W mode is the most stable setting. It persists across reboots; go back
  to MAXN with `sudo nvpmodel -m 0`.
- `record_data.py` flags frames that look torn (`tear_seams` and `torn` in
  `frames.jsonl`) and prints a warning at the end of the session. The check
  catches about half of the torn frames and misses milder tears, so a
  session with any flagged frames has more.

## Checking the ZED and power (for the next test)

Run these on the Jetson (`ssh jetson`). Keep the power monitor from
[Power supply](#power-supply) running in a second terminal during the
recording; it writes `/tmp/power.log`.

**1. Is the ZED connected, and where?**

```bash
lsusb | grep 2b03          # ZED 2i = 2b03:f880; nothing printed = not detected
lsusb -t                   # the ZED should show 5000M (USB3) with Driver=uvcvideo
ls /dev/video*             # /dev/video0 should exist
```

**2. Watch USB errors live** (plug/unplug the camera while this runs):

```bash
journalctl -kf | grep -E "OC ALARM|usb|uvc"
```

| Message | Meaning |
|---|---|
| `New USB device found, idVendor=2b03` | ZED detected |
| `error -71`, `Device not responding to setup address` | Power or signal problem on the link |
| `Non-zero status (-71) in video completion handler` | Video data arriving corrupted |
| `unable to enumerate USB device` | The hub gave up; unplug and replug the ZED |
| `soctherm: OC ALARM` | The Jetson's over-current alarm: 5 V supply overloaded |

`Entity type for entity ... was not initialized!` is normal for the ZED.

**3. Power mode and clocks**

```bash
nvpmodel -q                  # MAXN (0) or 5W (1)
sudo jetson_clocks --show    # clocks locked when MinFreq = MaxFreq
```

**4. 30 s test recording**, then count errors since it started:

```bash
cd ~/renee_ws/src/renee_perception
T=$(date +%H:%M:%S)
python3 tools/record_data.py --headless --video --duration 30
echo "USB errors: $(journalctl -k --since "$T" | grep -v 'OC ALARM' | grep -cE 'disconnect|reset|-71')"
echo "OC alarms:  $(journalctl -k --since "$T" | grep -c 'OC ALARM')"
awk -v t="$T" '$1>=t {if(!m||$2<m){m=$2;i=$4}} END{print "min voltage:", m, "mV at", i, "mA"}' /tmp/power.log
```

The end of the `record_data.py` output also prints `WARNING: X/Y frames look
torn` if any frame was flagged. To list the flagged frames of the latest
session:

```bash
d=$(ls -td data/*/ | head -1)
grep -c '"torn":true' ${d}frames.jsonl
```

**A good result:** 0 USB errors, 0 OC alarms, minimum voltage ≥ 4.75 V, no
torn-frame warning. Then try MAXN (`sudo nvpmodel -m 0`) and repeat.

**5. The Vogui's battery** (from the laptop):

```bash
ssh vogui-board 'source /opt/ros/*/setup.bash; rostopic echo -n1 /robot/battery_estimator/data'
```

`voltage` is the battery (about 52–53 V), `level` the charge in %. The
Jetson's 5 V comes from this battery through a converter, so a low 5 V with
a healthy battery points at the converter or the wiring.
