"""Log raw UART output from an Ai-Thinker RD-03D V2 radar module.

The public RD-03D V2 datasheet documents UART and the default baud rate, but
not the binary frame format. This probe records timestamped raw chunks so the
actual x/y target protocol can be identified from hardware output before a
parser is wired into the robot policy.
"""

from __future__ import annotations

import argparse
import csv
import string
import time
from pathlib import Path


def bytes_to_hex(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def bytes_to_printable(data: bytes) -> str:
    printable = set(string.printable.encode("ascii"))
    return "".join(chr(b) if b in printable and b not in b"\r\n\t" else "." for b in data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port, e.g. /dev/ttyUSB0 or /dev/serial0")
    parser.add_argument("--baud", type=int, default=256000, help="RD-03D V2 datasheet default is 256000 bps")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to log; 0 means run until Ctrl-C")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--csv-out", default="rd03d_raw.csv")
    parser.add_argument("--no-print", action="store_true", help="Do not print chunks to stdout")
    args = parser.parse_args()

    import serial

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    out_path = Path(args.csv_out)
    start = time.monotonic()
    rows = 0
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        print(f"Listening on {args.port} @ {args.baud} bps")
        print(f"Writing raw log to {out_path}")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time_s", "len", "hex", "printable"])
            writer.writeheader()
            try:
                while True:
                    now = time.monotonic()
                    if args.duration > 0 and now - start >= args.duration:
                        break
                    data = ser.read(args.chunk_size)
                    if not data:
                        continue
                    elapsed = now - start
                    hex_text = bytes_to_hex(data)
                    printable = bytes_to_printable(data)
                    writer.writerow(
                        {
                            "time_s": f"{elapsed:.6f}",
                            "len": len(data),
                            "hex": hex_text,
                            "printable": printable,
                        }
                    )
                    f.flush()
                    rows += 1
                    if not args.no_print:
                        print(f"{elapsed:9.3f}s len={len(data):3d} hex={hex_text} ascii={printable}")
            except KeyboardInterrupt:
                print("\nStopped.")
    print(f"Captured {rows} chunks.")


if __name__ == "__main__":
    main()
