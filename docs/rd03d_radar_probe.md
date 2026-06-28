# RD-03D V2 Radar Probe

The Ai-Thinker RD-03D V2 is a low-cost 24 GHz radar candidate for adding a
simple safety/target signal alongside the PiCam2 perception path.

Confirmed from the public datasheet:

- 24 GHz to 24.25 GHz FMCW radar
- target positioning and tracking firmware
- multiple-target reporting is described at the product level
- maximum sensing distance: 8 m in the normal direction
- azimuth range: +/-60 deg; elevation range: +/-30 deg
- 5 V supply, recommended peak current above 200 mA
- 4-pin connector: 5V, GND, UART_TX, UART_RX
- default serial baud rate: 256000 bps
- module IO is 3.3 V

The datasheet does not include the UART frame protocol. Before implementing a
real parser, capture the raw serial stream.

## Wiring

Use a USB-TTL adapter set to 3.3 V logic, or confirm logic levels before using
Pi GPIO UART directly.

```text
RD-03D 5V  -> 5V supply
RD-03D GND -> Pi/USB-TTL GND
RD-03D TX  -> USB-TTL RX
RD-03D RX  -> USB-TTL TX
```

## Raw Capture

Install the lightweight serial dependency:

```bash
source .venv/bin/activate
python -m pip install -r requirements-realworld.txt
```

Then capture:

```bash
python scripts/read_rd03d_uart.py --port /dev/ttyUSB0 --baud 256000 --duration 30 --csv-out rd03d_raw.csv
```

If using the Pi GPIO UART instead of USB-TTL, the port is often:

```bash
python scripts/read_rd03d_uart.py --port /dev/serial0 --baud 256000
```

Move one person at known points in front of the radar while logging. The raw CSV
lets us identify frame headers, target count, coordinate scale, and whether the
module reports `(x,y)`, `(distance,angle)`, speed, or status bits.
