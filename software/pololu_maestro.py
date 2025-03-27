import time
import serial

class PololuMaestro:
    """
    Simple interface for sending commands to the Pololu Maestro
    using the compact protocol over a serial port.
    """
    def __init__(self, port="COM3", baudrate=9600):
        """
        - port: e.g. "COM3" on Windows, or "/dev/ttyACM0" on Linux/Mac.
        - baudrate: must match your Maestro configuration (often 9600, 57600, or 115200).
        """
        self.ser = None
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            # Wait a moment for the device to be ready
            time.sleep(2)
            print(f"Connected to Pololu Maestro on {port}.")
        except Exception as e:
            print(f"Warning: Could not open Maestro on {port}. No hardware connected.\n{e}")

    def set_target(self, channel, target):
        """
        Sets the servo target for 'channel' in quarter-microseconds.
          Example: 1500 µs -> 6000, 500 µs -> 2000, 2500 µs -> 10000, etc.
        - channel: integer (0..5 on a 6-channel Maestro).
        - target: integer in [2000..10000] for 500..2500 µs (multiplied by 4).
        """
        if self.ser is None or not self.ser.is_open:
            return

        # Compact protocol command: 0x84, channel, lowbits(target), highbits(target)
        cmd = bytearray([0x84, channel, target & 0x7F, (target >> 7) & 0x7F])
        self.ser.write(cmd)

    def close(self):
        """Close the serial port when done."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
