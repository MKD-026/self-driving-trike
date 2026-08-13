# ESP steering backend

`run_autonomous_esp.sh` is the only hardware-actuating launcher. It starts with
the ESP centered and disarmed, performs a motion-free `STOP`/`STATUS` handshake,
and refuses to continue unless the ESP reports `cal=1`.

Steering mapping matches the calibrated trike:

```text
normalized -1 (left)  -> 100%
normalized  0         ->  50%
normalized +1 (right) ->   0%
```

Run:

```bash
./run_autonomous_esp.sh \
  --enable-esp \
  --esp-invert \
  --route-name route1 \
  --route-lookahead-m 5 \
  --no-tts
```

The process starts disarmed. `A` arms autonomous steering; `S`, `E`, or Space
disarms and centers; `Q` exits. Stale commands, serial failure, shutdown, stale
GPS, or an inactive route return steering to center. Throttle remains neutral.

`run_autonomous.sh` always injects `--dry-run` and cannot actuate hardware.
